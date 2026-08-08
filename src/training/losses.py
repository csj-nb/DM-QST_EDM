"""
Loss functions for diffusion model training.

The primary loss is the DDPM noise prediction loss, but additional
auxiliary losses can be added for physics constraints.
"""

import torch
import torch.nn.functional as F
from typing import Optional
import numpy as np


def ddpm_loss(
    noise_pred: torch.Tensor,
    noise: torch.Tensor,
    loss_type: str = "l2",
) -> torch.Tensor:
    """
    Standard DDPM noise prediction loss.

    L = E[ || epsilon - epsilon_theta(x_t, t) ||^p ]

    Args:
        noise_pred: Predicted noise of shape (B, D).
        noise: True noise of shape (B, D).
        loss_type: "l2" (MSE), "l1" (MAE), or "huber".

    Returns:
        Scalar loss.
    """
    if loss_type == "l2":
        return F.mse_loss(noise_pred, noise)
    elif loss_type == "l1":
        return F.l1_loss(noise_pred, noise)
    elif loss_type == "huber":
        return F.smooth_l1_loss(noise_pred, noise)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def purity_loss(
    rho: torch.Tensor,
    target_purity: float = 1.0,
) -> torch.Tensor:
    """
    Auxiliary loss to encourage a target purity level.

    Args:
        rho: Density matrix of shape (B, d, d).
        target_purity: Target purity (1.0 for pure states).

    Returns:
        Scalar loss encouraging Tr(rho^2) = target_purity.
    """
    # Compute purity: Tr(rho^2)
    # For complex rho: rho is (B, d, d), rho^2 = rho @ rho
    purity = torch.einsum("bij,bji->b", rho, rho)  # (B,)
    purity = torch.real(purity)

    return F.mse_loss(purity, torch.full_like(purity, target_purity))


def trace_norm_loss(rho: torch.Tensor) -> torch.Tensor:
    """
    Penalize deviation from unit trace.

    Args:
        rho: Density matrix of shape (B, d, d).

    Returns:
        Scalar penalty for non-unit trace.
    """
    trace = torch.einsum("bii->b", rho)  # (B,)
    trace = torch.real(trace)
    return F.mse_loss(trace, torch.ones_like(trace))
