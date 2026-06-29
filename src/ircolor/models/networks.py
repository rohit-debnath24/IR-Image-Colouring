import torch
import torch.nn as nn
import numpy as np

# --- 1. Wavelet / FEM Tools ---
class HaarDownsample(nn.Module):
    """Discrete Wavelet Transform (Haar) for Frequency Decoupling."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = in_channels * 4
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        x00 = x[..., 0::2, 0::2]
        x01 = x[..., 0::2, 1::2]
        x10 = x[..., 1::2, 0::2]
        x11 = x[..., 1::2, 1::2]
        
        LL = (x00 + x01 + x10 + x11) / 4.0
        HL = (x00 - x01 + x10 - x11) / 4.0
        LH = (x00 + x01 - x10 - x11) / 4.0
        HH = (x00 - x01 - x10 + x11) / 4.0
        
        return torch.cat([LL, HL, LH, HH], dim=1)

# --- 2. VSSM / Mamba Backbone Proxy ---
class SSMLinearAttention(nn.Module):
    """Proxy for Visual State-Space Model block (Linear Complexity)."""
    def __init__(self, dim: int):
        super().__init__()
        self.proj_in = nn.Linear(dim, dim * 2)
        self.conv1d = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act = nn.SiLU()
        self.proj_out = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        seq = x.view(B, C, -1).transpose(1, 2)
        
        z = self.proj_in(seq)
        x_proj, gate = z.chunk(2, dim=-1)
        
        x_proj = x_proj.transpose(1, 2)
        x_proj = self.conv1d(x_proj).transpose(1, 2)
        
        out = x_proj * self.act(gate)
        out = self.proj_out(out)
        
        return out.transpose(1, 2).view(B, C, H, W) + x

class VSSMNet(nn.Module):
    """Dual-Stream, Frequency-Decoupled State-Space Backbone."""
    def __init__(self, in_channels: int = 2, scale: int = 4, dim: int = 64):
        super().__init__()
        self.scale = scale
        
        self.stem = nn.Conv2d(in_channels, dim, kernel_size=3, padding=1)
        
        # VSSM Blocks
        self.vssm_blocks = nn.Sequential(*[SSMLinearAttention(dim) for _ in range(4)])
        
        # FEM (Frequency Decoupling)
        self.wavelet_down = HaarDownsample(dim)
        self.fem_proj = nn.Conv2d(dim * 4, dim, kernel_size=1)
        
        # Upsampling via Sub-Pixel
        upsample_blocks = []
        num_upsamples = int(np.log2(scale)) if scale > 1 else 0
        for _ in range(num_upsamples):
            upsample_blocks.append(nn.Conv2d(dim, dim * 4, kernel_size=3, padding=1))
            upsample_blocks.append(nn.PixelShuffle(upscale_factor=2))
            upsample_blocks.append(nn.SiLU())
            
        self.upsample = nn.Sequential(*upsample_blocks)
        self.conv_out = nn.Conv2d(dim, in_channels, kernel_size=3, padding=1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hr, _ = self.forward_with_features(x)
        return hr
        
    def forward_with_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feat = self.stem(x)
        
        vssm_feat = self.vssm_blocks(feat)
        
        freq_feat = self.wavelet_down(feat)
        freq_feat = self.fem_proj(freq_feat)
        freq_feat = nn.functional.interpolate(freq_feat, scale_factor=2, mode="bilinear", align_corners=False)
        
        combined_feat = vssm_feat + freq_feat
        hr_feat = self.upsample(combined_feat)
        hr_ir = self.conv_out(hr_feat)
        
        return hr_ir, hr_feat

# --- 3. Cross-Attention Skip Connections ---
class CrossAttentionBlock(nn.Module):
    def __init__(self, q_dim: int, kv_dim: int, heads: int = 4):
        super().__init__()
        self.heads = heads
        self.q_proj = nn.Conv2d(q_dim, q_dim, kernel_size=1)
        self.k_proj = nn.Conv2d(kv_dim, q_dim, kernel_size=1)
        self.v_proj = nn.Conv2d(kv_dim, q_dim, kernel_size=1)
        self.out_proj = nn.Conv2d(q_dim, q_dim, kernel_size=1)
        self.scale = (q_dim // heads) ** -0.5

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        B, C, H, W = q.shape
        if kv.shape[-2:] != q.shape[-2:]:
            kv = nn.functional.interpolate(kv, size=(H, W), mode="bilinear", align_corners=False)
            
        Q = self.q_proj(q).view(B, self.heads, C // self.heads, -1).transpose(-1, -2) # B, heads, N, D
        K = self.k_proj(kv).view(B, self.heads, C // self.heads, -1).transpose(-1, -2) # B, heads, N, D
        V = self.v_proj(kv).view(B, self.heads, C // self.heads, -1).transpose(-1, -2) # B, heads, N, D
        
        # Use PyTorch memory-efficient attention
        attn_out = torch.nn.functional.scaled_dot_product_attention(Q, K, V)
        
        out = attn_out.transpose(-1, -2).reshape(B, C, H, W)
        return self.out_proj(out) + q

# --- 4. ControlNet-Guided Generative Colorizer ---
class ControlNetColorizer(nn.Module):
    """U-Net with Cross-Attention structural guidance."""
    def __init__(self, in_channels: int = 2, out_channels: int = 3, bridge_dim: int = 64):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(nn.Conv2d(in_channels, 32, 3, 1, 1), nn.SiLU(), nn.Conv2d(32, 32, 3, 1, 1), nn.SiLU())
        self.pool1 = nn.MaxPool2d(2)
        
        self.enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, 1, 1), nn.SiLU(), nn.Conv2d(64, 64, 3, 1, 1), nn.SiLU())
        self.pool2 = nn.MaxPool2d(2)
        
        # Cross-Attention Bridges
        self.attn_bridge1 = CrossAttentionBlock(q_dim=32, kv_dim=bridge_dim)
        self.attn_bridge2 = CrossAttentionBlock(q_dim=64, kv_dim=bridge_dim)
        
        self.bottleneck = nn.Sequential(nn.Conv2d(64, 128, 3, 1, 1), nn.SiLU(), nn.Conv2d(128, 128, 3, 1, 1), nn.SiLU())
        
        self.attn_bridge_bn = CrossAttentionBlock(q_dim=128, kv_dim=bridge_dim)
        
        # Decoder
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(128, 64, 3, 1, 1), nn.SiLU())
        
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(64, 32, 3, 1, 1), nn.SiLU())
        
        self.final = nn.Conv2d(32, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b_feat = torch.zeros(x.size(0), 64, x.size(2), x.size(3), device=x.device, dtype=x.dtype)
        return self.forward_conditioned(x, b_feat)
        
    def forward_conditioned(self, x: torch.Tensor, bridge_features: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x1_cond = self.attn_bridge1(x1, bridge_features)
        p1 = self.pool1(x1_cond)
        
        x2 = self.enc2(p1)
        x2_cond = self.attn_bridge2(x2, bridge_features)
        p2 = self.pool2(x2_cond)
        
        b = self.bottleneck(p2)
        b_cond = self.attn_bridge_bn(b, bridge_features)
        
        d2 = self.up2(b_cond)
        d2 = torch.cat([d2, x2_cond], dim=1)
        x_dec2 = self.dec2(d2)
        
        d1 = self.up1(x_dec2)
        d1 = torch.cat([d1, x1_cond], dim=1)
        x_dec1 = self.dec1(d1)
        
        return torch.sigmoid(self.final(x_dec1))
