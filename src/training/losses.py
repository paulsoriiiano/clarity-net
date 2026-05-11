"""
Loss functions for training.
"""

import torch.nn.functional as F


def mag_l1_loss(pred_mag, target_mag):
    """
    L1 loss on magnitude.

    Args:
        pred_mag:   [B, F, T]
        target_mag: [B, F, T]
    """
    if pred_mag.shape != target_mag.shape:
        raise ValueError(f"Shape mismatch: pred {pred_mag.shape}, target {target_mag.shape}")
    if pred_mag.dim() != 3:
        raise ValueError(f"Expected [B, F, T], got {pred_mag.shape}")

    return F.l1_loss(pred_mag, target_mag)


def complex_mse_loss(pred, target):
    """
    MSE on real and imaginary parts.

    Args:
        pred:   [B, 2, F, T]
        target: [B, 2, F, T]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    if pred.dim() != 4 or pred.size(1) != 2:
        raise ValueError(f"Expected [B, 2, F, T], got {pred.shape}")

    loss_real = F.mse_loss(pred[:, 0], target[:, 0])
    loss_imag = F.mse_loss(pred[:, 1], target[:, 1])
    return loss_real + loss_imag


def complex_l1_loss(pred, target):
    """
    L1 loss on real and imaginary parts.

    Args:
        pred:   [B, 2, F, T]
        target: [B, 2, F, T]
    """
    if pred.shape != target.shape:
        raise ValueError(f"Shape mismatch: pred {pred.shape}, target {target.shape}")
    if pred.dim() != 4 or pred.size(1) != 2:
        raise ValueError(f"Expected [B, 2, F, T], got {pred.shape}")

    loss_real = F.l1_loss(pred[:, 0], target[:, 0])
    loss_imag = F.l1_loss(pred[:, 1], target[:, 1])
    return loss_real + loss_imag


def fullsubnet_plus_l1_loss(
    enh_real,
    enh_imag,
    clean_real,
    clean_imag,
    enhanced_mag=None,
    clean_mag=None,
    pred_mask=None,
):
    loss = (
        F.l1_loss(enh_real, clean_real) +
        F.l1_loss(enh_imag, clean_imag)
    )

    if enhanced_mag is not None and clean_mag is not None:
        loss = loss + 0.5 * F.l1_loss(enhanced_mag, clean_mag)

    return loss


def get_loss_fn(loss_name):
    if loss_name == "mag_l1":
        return mag_l1_loss
    elif loss_name == "complex_mse":
        return complex_mse_loss
    elif loss_name == "fullsubnet_plus_l1":
        return fullsubnet_plus_l1_loss
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")