import torch

from ircolor.losses.objectives import (
    gradient_intensity_loss,
    gradient_ssim_loss,
    UncertaintyWeightedLoss,
)


def test_gradient_loss_zero_on_identical():
    x = torch.rand(2, 3, 16, 16)
    assert gradient_intensity_loss(x, x).item() == 0.0


def test_gradient_loss_positive_on_blur():
    x = torch.rand(2, 3, 16, 16)
    blurred = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
    assert gradient_intensity_loss(blurred, x).item() > 0.0


def test_gradient_ssim_loss_zero_on_identical():
    x = torch.rand(2, 3, 16, 16)
    assert abs(gradient_ssim_loss(x, x).item()) < 1e-5


def test_gradient_ssim_loss_positive_on_blur():
    x = torch.rand(2, 3, 16, 16)
    blurred = torch.nn.functional.avg_pool2d(x, 3, 1, 1)
    assert gradient_ssim_loss(blurred, x).item() > 0.0


def test_uncertainty_weighted_loss_gradient_flow():
    loss_module = UncertaintyWeightedLoss(num_tasks=3, initial_log_vars=[0.1, 0.2, 0.3])
    loss1 = torch.tensor(1.5, requires_grad=True)
    loss2 = torch.tensor(0.8, requires_grad=True)
    loss3 = torch.tensor(0.5, requires_grad=True)
    
    total_loss, weights = loss_module([loss1, loss2, loss3])
    
    assert total_loss.item() > 0.0
    assert len(weights) == 3
    assert all(w > 0.0 for w in weights)
    
    # Verify we can backprop through the combined weighted loss field
    total_loss.backward()
    assert loss1.grad is not None
    assert loss_module.log_vars.grad is not None

