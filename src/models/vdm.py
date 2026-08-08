"""
Variational Diffusion Model (VDM) for Quantum State Tomography.

Based on Kingma et al., "Variational Diffusion Models" (NeurIPS 2021).

Key differences from DDPM:
    - Continuous time t ∈ [0,1] (not discrete 1..T)
    - Learnable SNR schedule gamma(t) via monotonic neural network
    - x_0-parameterization: directly predicts clean state
    - ELBO loss with automatic SNR derivative weighting
    - ODE-based deterministic sampling

Physics constraint:
    - Low-rank auxiliary loss via purity: L_rank = 1 - Tr(rho^2)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from .unet import CholeskyUNet
from .conditioning import ConditioningNetwork
from .vdm_snr import MonotonicSNR


def _vector_to_dm_pytorch(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Differentiable Cholesky vector → density matrix conversion.

    Layout matches `cholesky_to_dm` in numpy but uses PyTorch ops.
    Purely functional (no in-place writes): gradients flow back to x.

    Args:
        x: Cholesky vector of shape (B, d*d).
        d: Hilbert space dimension.
    Returns:
        Density matrix of shape (B, d, d), complex-valued.
    """
    B = x.shape[0]
    device = x.device
    dtype = x.dtype

    # Cholesky layout: [diag(d), strictly-lower-triangular (interleaved
    # real, imag)].
    tril_off = torch.tril_indices(d, d, offset=-1, device=device)  # (2, n_off)
    n_off = tril_off.shape[1]
    diag_flat_idx = torch.arange(d, device=device) * (d + 1)  # (d,)
    off_flat_idx = tril_off[0] * d + tril_off[1]  # (n_off,)

    off = x[:, d:]  # (B, 2*n_off)
    off_real = off[:, 0::2]
    off_imag = off[:, 1::2]

    # Build L via functional scatter (returns a new tensor; differentiable
    # in `x`). reshape to (B, d, d): rows 0..d-1.
    L_real = torch.zeros(B, d * d, device=device, dtype=dtype)
    L_real = torch.scatter(
        L_real, 1,
        diag_flat_idx.expand(B, d),
        x[:, :d],
    )
    L_real = torch.scatter(
        L_real, 1,
        off_flat_idx.expand(B, n_off),
        off_real,
    )
    L_imag = torch.zeros(B, d * d, device=device, dtype=dtype)
    L_imag = torch.scatter(
        L_imag, 1,
        off_flat_idx.expand(B, n_off),
        off_imag,
    )

    L = torch.complex(L_real, L_imag).reshape(B, d, d)

    # rho = L L^H / Tr(L L^H)
    Lh = L.transpose(-2, -1).conj()
    M = torch.bmm(L, Lh)  # (B, d, d)
    trace = torch.real(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1))  # (B,)
    rho = M / (trace.unsqueeze(-1).unsqueeze(-1) + 1e-10)

    # Force Hermitian
    rho = (rho + rho.transpose(-2, -1).conj()) / 2.0
    return rho


def _purity(rho: torch.Tensor) -> torch.Tensor:
    """Compute purity Tr(rho^2) for batch of density matrices."""
    # rho: (B, d, d) complex
    rho_sq = torch.bmm(rho, rho)  # (B, d, d)
    return torch.real(torch.diagonal(rho_sq, dim1=-2, dim2=-1).sum(-1))  # (B,)


class VDM(nn.Module):
    """
    Variational Diffusion Model with learnable noise schedule.

    Usage:
        model = VDM(d=4, cond_input_dim=36)
        loss = model.training_loss(x_0, measurement)
        x_0_pred = model.sample(measurement, n_steps=100)
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
        # VDM-specific
        gamma_min: float = -15.0,
        gamma_max: float = 15.0,
        # Low-rank constraint
        lambda_rank: float = 0.0,
        lambda_rank_warmup: int = 1000,
    ):
        """
        Args:
            d: Hilbert space dimension (2^n).
            cond_input_dim: Measurement vector dimension (6^n).
            base_channels, dim_mults, num_res_blocks: UNet architecture.
            cond_dim: Conditioning embedding dimension.
            cond_dropout_prob: Conditioning dropout for CFG.
            loss_type: "l2" or "l1".
            gamma_min: Minimum log-SNR at t=1.
            gamma_max: Maximum log-SNR at t=0.
            snr_hidden_dim: Hidden dim of monotonic SNR network.
            snr_n_layers: Number of layers in SNR network.
            lambda_rank: Weight of low-rank auxiliary loss (0 = disabled).
            lambda_rank_warmup: Steps to linearly warm up lambda_rank.
        """
        super().__init__()
        self.d = d
        self.cholesky_dim = d * d
        self.loss_type = loss_type
        self.lambda_rank = lambda_rank
        self.lambda_rank_warmup = lambda_rank_warmup
        self.register_buffer("global_step", torch.tensor(0))

        # --- Learnable SNR schedule ---
        self.snr_schedule = MonotonicSNR(
            gamma_min=gamma_min,
            gamma_max=gamma_max,
        )

        # --- Denoising network (UNet, same as DDPM/FlowMatching) ---
        self.denoise_fn = CholeskyUNet(
            d=d,
            base_channels=base_channels,
            dim_mults=dim_mults,
            num_res_blocks=num_res_blocks,
            in_channels=2,
            time_emb_dim=256,
            cond_dim=cond_dim,
        )

        # --- Conditioning network ---
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

    def q_sample(
        self,
        x_0: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward diffusion: z_t = alpha_t * x_0 + sigma_t * epsilon.

        Args:
            x_0: Clean Cholesky vector (B, D).
            t: Continuous time in [0,1] (B,).
            noise: Optional pre-sampled noise.

        Returns:
            Noisy sample z_t (B, D).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        alpha_t, sigma_t = self.snr_schedule.get_coefficients(t)

        # Reshape for broadcasting
        while alpha_t.dim() < x_0.dim():
            alpha_t = alpha_t.unsqueeze(-1)
            sigma_t = sigma_t.unsqueeze(-1)

        return alpha_t * x_0 + sigma_t * noise

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
        VDM ELBO training loss with optional low-rank constraint.

        L = 1/2 * E_t [-SNR'(t) * ||x_0 - x̂_0||²] + lambda * L_rank

        Args:
            x_0: Clean Cholesky vector (B, D).
            condition: Measurement vector (B, cond_input_dim).
            noise: Optional pre-sampled noise.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample continuous time
        t = torch.rand(B, device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion
        alpha_t, sigma_t = self.snr_schedule.get_coefficients(t)
        z_t = self._broadcast_mul(alpha_t, x_0) + self._broadcast_mul(sigma_t, noise)

        # Encode condition
        cond_emb, _ = self.conditioning(condition)

        # Map t to integer-like input for UNet
        t_int = (t * 999).long()

        # Predict clean x_0
        x_0_pred = self.denoise_fn(z_t, t_int, cond_emb)

        # --- ELBO loss (weighted by SNR derivative) ---
        snr_deriv = self.snr_schedule.get_snr_derivative(t)  # (B,) = SNR'(t)
        # Negative because SNR decreases (SNR'(t) < 0), we want positive loss
        # L = 1/2 * E[-SNR'(t) * ||x_0 - x̂||²]
        weight = -snr_deriv  # (B,), should be positive
        weight = weight / (weight.detach().mean() + 1e-8)  # Normalize for stability

        if self.loss_type == "l2":
            diff = (x_0 - x_0_pred).pow(2).mean(dim=-1)  # (B,)
        elif self.loss_type == "l1":
            diff = (x_0 - x_0_pred).abs().mean(dim=-1)  # (B,)
        else:
            diff = (x_0 - x_0_pred).pow(2).mean(dim=-1)

        elbo_loss = 0.5 * (weight * diff).mean()

        # --- Low-rank auxiliary loss ---
        total_loss = elbo_loss

        if self.lambda_rank > 0 and self.training:
            # Warmup: linearly increase lambda from 0 to lambda_rank
            step = self.global_step.item()
            current_lambda = self.lambda_rank * min(1.0, step / max(1, self.lambda_rank_warmup))

            # Convert predicted Cholesky to density matrix (differentiable)
            rho_pred = _vector_to_dm_pytorch(x_0_pred, self.d)

            # Purity-based rank penalty: 1 - Tr(rho^2)
            pur = _purity(rho_pred)  # (B,)
            rank_loss = (1.0 - pur).mean()

            total_loss = elbo_loss + current_lambda * rank_loss

        # Only advance the internal step counter during training (validation
        # calls training_loss with model.eval() and must not skew warmup).
        if self.training:
            self.global_step += 1
        return total_loss

    # ------------------------------------------------------------------
    # Sampling (ODE Integration)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_steps: int = 100,
        method: str = "euler",
        batch_size: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate x_0 via ODE integration.

        The reverse ODE:
            dz/dt = -1/2 * d(log SNR)/dt * x̂_θ(z, t)
                 = -1/2 * gamma'(t) * x̂_θ(z, t)

        Args:
            condition: Measurement vector (B, cond_input_dim).
            n_steps: Number of integration steps (50-200 for good quality).
            method: "euler" or "midpoint".
            batch_size: Override batch size.
            progress: Show progress bar.

        Returns:
            Clean Cholesky vector (B, D).
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
        _, sigma_1 = self.snr_schedule.get_coefficients(
            torch.ones(1, device=device)
        )
        z = sigma_1.item() * torch.randn(B, self.cholesky_dim, device=device)

        # Integration steps from t=1 to t=0
        dt = -1.0 / n_steps  # negative because going backward
        times = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        iterator = zip(times[:-1], times[1:])
        if progress:
            iterator = tqdm(list(iterator), desc=f"VDM Sampling ({n_steps} steps)",
                           total=n_steps)

        for t_start, t_end in iterator:
            t_mid = (t_start + t_end) / 2.0
            t_tensor = t_start.unsqueeze(0).expand(B)
            t_int = (t_tensor * 999).long()

            if method == "euler":
                # dz = -0.5 * gamma'(t) * x̂ * dt
                _, gamma_prime = self.snr_schedule.forward(t_tensor)  # (B,)
                x_0_pred = self.denoise_fn(z, t_int, cond_emb)
                dz = -0.5 * self._broadcast_mul(gamma_prime, x_0_pred)
                z = z + dz * dt

            elif method == "midpoint":
                # Midpoint (2nd order)
                _, gamma_prime_1 = self.snr_schedule.forward(t_tensor)
                x_0_pred_1 = self.denoise_fn(z, t_int, cond_emb)
                dz_1 = -0.5 * self._broadcast_mul(gamma_prime_1, x_0_pred_1)
                z_mid = z + dz_1 * dt / 2.0

                t_mid_tensor = t_mid.unsqueeze(0).expand(B)
                t_mid_int = (t_mid_tensor * 999).long()
                _, gamma_prime_2 = self.snr_schedule.forward(t_mid_tensor)
                x_0_pred_2 = self.denoise_fn(z_mid, t_mid_int, cond_emb)
                dz_2 = -0.5 * self._broadcast_mul(gamma_prime_2, x_0_pred_2)
                z = z + dz_2 * dt

        return z  # x_0 prediction at t=0

    @torch.no_grad()
    def unconditional_sample(
        self,
        batch_size: int = 1,
        n_steps: int = 100,
        progress: bool = True,
    ) -> torch.Tensor:
        """Generate unconditionally (no measurement conditioning)."""
        device = next(self.parameters()).device
        dummy_cond = torch.zeros(
            batch_size, self.conditioning.encoder[0].in_features, device=device
        )
        cond_emb, _ = self.conditioning(dummy_cond, force_dropout=True)

        _, sigma_1 = self.snr_schedule.get_coefficients(torch.ones(1, device=device))
        z = sigma_1.item() * torch.randn(batch_size, self.cholesky_dim, device=device)

        dt = -1.0 / n_steps
        times = torch.linspace(1.0, 0.0, n_steps + 1, device=device)

        iterator = zip(times[:-1], times[1:])
        if progress:
            iterator = tqdm(list(iterator), desc="VDM Uncond", total=n_steps)

        for t_start, t_end in iterator:
            t_tensor = t_start.unsqueeze(0).expand(batch_size)
            t_int = (t_tensor * 999).long()
            _, gamma_prime = self.snr_schedule.forward(t_tensor)
            x_0_pred = self.denoise_fn(z, t_int, cond_emb)
            dz = -0.5 * self._broadcast_mul(gamma_prime, x_0_pred)
            z = z + dz * dt

        return z

    @staticmethod
    def _broadcast_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Multiply a (B,) or scalar with b (B, D), broadcasting correctly."""
        while a.dim() < b.dim():
            a = a.unsqueeze(-1)
        return a * b
