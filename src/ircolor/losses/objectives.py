"""Loss terms. Total = w_pix*L1 + w_adv*GAN + w_grad*Gradient + w_sem*Semantic.

Gradient-intensity loss penalizes blurry edges during super-resolution.
Semantic loss penalizes the colorizer when predicted RGB classifies (via the frozen
segmentation net) differently from the ground-truth RGB -- this is the anti-hallucination
guardrail described in the project blueprint.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def adversarial_loss(logits: torch.Tensor, is_real: bool) -> torch.Tensor:
    """Binary Cross-Entropy for Adversarial Training (PatchGAN)."""
    target = torch.ones_like(logits) if is_real else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, target)

def gradient_intensity_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize edge/gradient mismatch (Sobel-like finite differences)."""
    def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    pgx, pgy = grad(pred)
    tgx, tgy = grad(target)
    return F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)


def _ssim(x: torch.Tensor, y: torch.Tensor, C1: float = 0.01**2, C2: float = 0.03**2) -> torch.Tensor:
    """Self-contained 2D Structural Similarity Index (SSIM) helper."""
    mu_x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    mu_y = F.avg_pool2d(y, kernel_size=3, stride=1, padding=1)
    
    mu_x_sq = mu_x.pow(2)
    mu_y_sq = mu_y.pow(2)
    mu_xy = mu_x * mu_y
    
    sigma_x_sq = F.avg_pool2d(x * x, kernel_size=3, stride=1, padding=1) - mu_x_sq
    sigma_y_sq = F.avg_pool2d(y * y, kernel_size=3, stride=1, padding=1) - mu_y_sq
    sigma_xy = F.avg_pool2d(x * y, kernel_size=3, stride=1, padding=1) - mu_xy
    
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
    return ssim_map.mean()


def gradient_ssim_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Enforce edge sharpness via Structural Similarity in the gradient domain."""
    def grad(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    pgx, pgy = grad(pred)
    tgx, tgy = grad(target)
    
    loss_x = 1.0 - _ssim(pgx, tgx)
    loss_y = 1.0 - _ssim(pgy, tgy)
    return 0.5 * (loss_x + loss_y)


def direction_aligned_gradient_loss(pred: torch.Tensor, target: torch.Tensor, scales: int = 3) -> torch.Tensor:
    """Direction-Aligned Multi-Scale Gradient Loss.
    
    Supervises horizontal and vertical gradient components separately across
    multiple downsampled scales to eliminate directional blur.
    """
    def get_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        gx = x[..., :, 1:] - x[..., :, :-1]
        gy = x[..., 1:, :] - x[..., :-1, :]
        return gx, gy

    loss = 0.0
    for scale in range(scales):
        if scale > 0:
            pred = F.avg_pool2d(pred, kernel_size=2, stride=2)
            target = F.avg_pool2d(target, kernel_size=2, stride=2)
            
        pgx, pgy = get_gradients(pred)
        tgx, tgy = get_gradients(target)
        
        # Scale-normalized L1 loss on directed gradients
        loss += F.l1_loss(pgx, tgx) + F.l1_loss(pgy, tgy)
        
    return loss / scales


def semantic_consistency_loss(
    pred_rgb: torch.Tensor, target_rgb: torch.Tensor, frozen_seg: torch.nn.Module
) -> torch.Tensor:
    """KL/CE between land-cover logits of predicted vs. ground-truth RGB. `frozen_seg` is eval-only."""
    with torch.no_grad():
        target_out = frozen_seg(target_rgb)
        target_logits = target_out.logits if hasattr(target_out, "logits") else target_out
    
    pred_out = frozen_seg(pred_rgb)
    pred_logits = pred_out.logits if hasattr(pred_out, "logits") else pred_out
    
    return F.kl_div(
        F.log_softmax(pred_logits, dim=1), F.softmax(target_logits, dim=1), reduction="batchmean"
    )


class UncertaintyWeightedLoss(torch.nn.Module):
    """Dynamic Homoscedastic Uncertainty Weighting for multi-task loss auto-tuning.
    
    Balances the standard loss, gradient SSIM loss, and semantic consistency loss.
    """
    def __init__(self, num_tasks: int = 3, initial_log_vars: list[float] | None = None):
        super().__init__()
        if initial_log_vars is None:
            initial_log_vars = [0.0] * num_tasks
        self.log_vars = torch.nn.Parameter(torch.tensor(initial_log_vars, dtype=torch.float32))

    def forward(self, losses: list[torch.Tensor]) -> tuple[torch.Tensor, list[float]]:
        """Computes the weighted sum of multi-task losses.
        
        L_total = sum( 0.5 * exp(-log_var_i) * L_i + 0.5 * log_var_i )
        """
        total_loss = 0.0
        weights = []
        for i, loss in enumerate(losses):
            log_var = self.log_vars[i]
            precision = torch.exp(-log_var)
            weighted_loss = 0.5 * precision * loss + 0.5 * log_var
            total_loss = total_loss + weighted_loss
            weights.append(precision.item())
        return total_loss, weights

