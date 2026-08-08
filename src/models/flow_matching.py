"""
Conditional Flow Matching for Quantum State Tomography.

Replaces DDPM's noisy curved diffusion path with a straight-line
probability path (Rectified Flow / Conditional Flow Matching).

Key differences from DDPM:
    Forward:  x_t = (1-t)*x_0 + t*ε              (straight line, not noisy SDE)
    Target:   v = ε - x_0                          (velocity field, not noise)
    Loss:     MSE(v_θ(x_t, t, cond), ε - x_0)
    Sampling: ODE dx/dt = v_θ, solved via Euler    (2-10 steps, not 1000)

References:
    - Liu et al. "Flow Straight and Fast" (ICLR 2023)
    - Lipman et al. "Flow Matching" (ICLR 2023)
    - Albergo et al. "Stochastic Interpolants" (2023)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
from tqdm import tqdm

from .unet import CholeskyUNet
from .conditioning import ConditioningNetwork


class FlowMatching(nn.Module):
    """
    Conditional Flow Matching for quantum state tomography.

    Usage (identical API to DDPM):
        model = FlowMatching(d=4, cond_input_dim=36)
        loss = model.training_loss(x_0, measurement)
        x_0_pred = model.sample(measurement, n_steps=4)
    """

    def __init__(
        self,
        d: int,
        cond_input_dim: int = 36,
        base_channels: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 128,
        cond_dropout_prob: float = 0.1,
        loss_type: str = "l2",
        sigma_min: float = 0.001,  # Small noise at t=0 for stability
    ):
        """
        Args:
            d: Hilbert space dimension (2^n).
            cond_input_dim: Dimension of measurement vector (6^n).
            base_channels: Base channel count for UNet.
            dim_mults: UNet channel multipliers.
            num_res_blocks: Number of ResNet blocks per UNet level.
            cond_dim: Conditioning embedding dimension.
            cond_dropout_prob: Conditioning dropout probability.
            loss_type: "l2" or "l1" loss.
            sigma_min: Minimum noise std at t=0 (prevents singularity).
        """
        super().__init__()
        self.d = d
        self.cholesky_dim = d * d
        self.loss_type = loss_type
        self.sigma_min = sigma_min

        # --- Denoising/Velocity network (same UNet as DDPM) ---
        self.denoise_fn = CholeskyUNet(
            d=d,
            base_channels=base_channels,
            dim_mults=dim_mults,
            num_res_blocks=num_res_blocks,
            in_channels=2,
            time_emb_dim=256,
            cond_dim=cond_dim,
        )

        # --- Conditioning network (same as DDPM) ---
        self.conditioning = ConditioningNetwork(
            input_dim=cond_input_dim,
            hidden_dim=256,
            cond_dim=cond_dim,
            num_resolutions=self.denoise_fn.num_resolutions,
            base_channels=base_channels,
            dim_mults=dim_mults,
            cond_dropout_prob=cond_dropout_prob,
        )

    # ------------------------------------------------------------------
    # Forward Process
    # ------------------------------------------------------------------

    def interpolate(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Straight-line interpolation: x_t = (1-t) * x_0 + t * epsilon.

        Args:
            x_0: Clean Cholesky vector of shape (B, D).
            t: Continuous time in [0, 1], shape (B,).
            noise: Target noise. If None, sample from N(0, I).

        Returns:
            x_t of shape (B, D).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        # Reshape t for broadcasting
        t = t.view(-1, *([1] * (x_0.dim() - 1)))

        # Add small noise at t=0 (stability)
        sigma = self.sigma_min
        return (1.0 - (1.0 - sigma) * t) * x_0 + t * noise

    def true_velocity(
        self,
        x_0: torch.Tensor,
        noise: torch.Tensor,
    ) -> torch.Tensor:
        """
        True velocity field: v = (1 - sigma_min) * (epsilon - x_0).

        The velocity is constant along the straight-line path from x_0 to epsilon.
        """
        return (1.0 - self.sigma_min) * (noise - x_0)

    # ------------------------------------------------------------------
    # Training Loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Flow Matching training loss.

        L = E_{t ~ Uniform(0,1)} [ || v_θ(x_t, t, cond) - (ε - x_0) ||^2 ]

        Args:
            x_0: Clean Cholesky vector of shape (B, D).
            condition: Measurement vector of shape (B, cond_input_dim).
            noise: Target noise. If None, sampled from N(0, I).

        Returns:
            Scalar loss.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random continuous time in [0, 1]
        t = torch.rand(B, device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward interpolation
        x_t = self.interpolate(x_0, t, noise=noise)

        # True velocity
        v_true = self.true_velocity(x_0, noise)

        # Encode condition
        cond_emb, _ = self.conditioning(condition)

        # Map t from [0,1] to integer-like index for UNet's sinusoidal embedding
        # UNet expects integer timesteps; we map t -> t * 1000
        t_int = (t * 999).long()

        # Predict velocity
        v_pred = self.denoise_fn(x_t, t_int, cond_emb)

        # Loss
        if self.loss_type == "l2":
            loss = F.mse_loss(v_pred, v_true)
        elif self.loss_type == "l1":
            loss = F.l1_loss(v_pred, v_true)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    # ------------------------------------------------------------------
    # Sampling (ODE Integration)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_steps: int = 4,
        method: str = "euler",
        batch_size: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate x_0 from noise via ODE integration.

        Args:
            condition: Measurement vector of shape (batch_size, cond_input_dim).
            n_steps: Number of ODE integration steps (2-10 is usually enough).
            method: Integration method - "euler" or "midpoint" or "rk4".
            batch_size: Override batch size.
            progress: Show progress bar.

        Returns:
            Predicted clean Cholesky vector of shape (batch_size, D).
        """
        if batch_size is not None:
            B = batch_size
            if condition.shape[0] != B:
                condition = condition[:B]
        else:
            B = condition.shape[0]

        device = next(self.parameters()).device
        condition = condition.to(device)

        # Encode condition once
        cond_emb, _ = self.conditioning(condition)

        # Start from pure noise at t=1
        x = torch.randn(B, self.cholesky_dim, device=device)

        # Time steps from t=1 to t=0
        dt = 1.0 / n_steps
        times = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        iterator = zip(times[:-1], times[1:])
        if progress:
            iterator = tqdm(list(iterator), desc=f"Flow Sampling ({n_steps} steps)",
                           total=n_steps)

        for t_start, t_end in iterator:
            t_mid = (t_start + t_end) / 2.0
            t_tensor = (t_start * 999).long().unsqueeze(0).expand(B)

            if method == "euler":
                # x_{t-dt} = x_t - v(x_t, t) * dt
                v = self.denoise_fn(x, t_tensor, cond_emb)
                x = x - v * dt

            elif method == "midpoint":
                # Midpoint method (2nd order)
                v1 = self.denoise_fn(x, t_tensor, cond_emb)
                x_mid = x - v1 * dt / 2.0
                t_mid_tensor = (t_mid * 999).long().unsqueeze(0).expand(B)
                v2 = self.denoise_fn(x_mid, t_mid_tensor, cond_emb)
                x = x - v2 * dt

            elif method == "rk4":
                # Classical 4th-order Runge-Kutta
                def v_fn(xx, tt):
                    t_tt = (tt * 999).long().unsqueeze(0).expand(B)
                    return self.denoise_fn(xx, t_tt, cond_emb)

                k1 = v_fn(x, t_start)
                k2 = v_fn(x - k1 * dt / 2.0, t_mid)
                k3 = v_fn(x - k2 * dt / 2.0, t_mid)
                k4 = v_fn(x - k3 * dt, t_end)
                x = x - (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6.0

        return x

    @torch.no_grad()
    def unconditional_sample(
        self,
        batch_size: int = 1,
        n_steps: int = 4,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate states unconditionally."""
        device = next(self.parameters()).device
        dummy_cond = torch.zeros(
            batch_size, self.conditioning.encoder[0].in_features, device=device
        )
        cond_emb, _ = self.conditioning(dummy_cond, force_dropout=True)

        x = torch.randn(batch_size, self.cholesky_dim, device=device)
        dt = 1.0 / n_steps

        iterator = range(n_steps)
        if progress:
            iterator = tqdm(iterator, desc=f"Uncond Flow ({n_steps} steps)")

        for step in iterator:
            t = 1.0 - step * dt
            t_tensor = (t * 999).long().unsqueeze(0).expand(batch_size)
            v = self.denoise_fn(x, t_tensor, cond_emb)
            x = x - v * dt

        return x
