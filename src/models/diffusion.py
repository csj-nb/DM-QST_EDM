"""
Denoising Diffusion Probabilistic Model (DDPM) for quantum state tomography.

Implements:
    - Forward diffusion process: q(x_t | x_0)
    - Training loss: noise prediction loss
    - DDPM sampling (reverse process)
    - DDIM accelerated sampling

The diffusion operates in the Cholesky vector space. The denoising network
predicts the noise epsilon that was added to x_0 to obtain x_t.

Conditioning is provided through the measurement data, encoded by the
ConditioningNetwork into FiLM parameters that modulate the UNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from .noise_schedules import make_beta_schedule, compute_diffusion_parameters
from .unet import CholeskyUNet
from .conditioning import ConditioningNetwork, TimeEmbedding


class DDPM(nn.Module):
    """
    Denoising Diffusion Probabilistic Model.

    Usage:
        # Training
        model = DDPM(d=4, timesteps=1000, cond_input_dim=36)
        loss = model.training_loss(x_0, measurement)
        loss.backward()

        # Sampling
        x_0_pred = model.sample(measurement, batch_size=1)

    Args:
        d: Hilbert space dimension (2^n).
        timesteps: Number of diffusion steps T.
        cond_input_dim: Dimension of measurement vector (6^n).
        base_channels: Base channel count for UNet.
        dim_mults: UNet channel multipliers per resolution level.
        cond_dim: Conditioning embedding dimension.
        cond_dropout_prob: Conditioning dropout probability (for CFG).
        beta_schedule: Noise schedule type ("cosine" or "linear").
        beta_start: Starting beta for linear schedule.
        beta_end: Ending beta for linear schedule.
        loss_type: Loss function type ("l2", "l1", or "huber").
        self_conditioning: Whether to use self-conditioning (x_0 prediction fed back).
    """

    def __init__(
        self,
        d: int,
        timesteps: int = 1000,
        cond_input_dim: int = 36,
        base_channels: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 128,
        cond_dropout_prob: float = 0.1,
        beta_schedule: str = "cosine",
        beta_start: float = 1e-4,
        beta_end: float = 0.02,
        loss_type: str = "l2",
        self_conditioning: bool = False,
    ):
        super().__init__()
        self.d = d
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.self_conditioning = self_conditioning
        self.cholesky_dim = d * d

        # --- Noise schedule ---
        betas = make_beta_schedule(
            timesteps=timesteps,
            schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
        )
        self.register_buffer("betas", betas)

        params = compute_diffusion_parameters(betas)
        for key, val in params.items():
            self.register_buffer(key, val)

        # --- Denoising network (UNet) ---
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
        Forward diffusion: sample x_t given x_0.

        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon

        Args:
            x_0: Clean Cholesky vector of shape (B, d*d).
            t: Time steps of shape (B,).
            noise: Pre-sampled noise. If None, sample from N(0, I).

        Returns:
            Noisy sample x_t of shape (B, d*d).
        """
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_alpha_bar = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar = self.sqrt_one_minus_alphas_cumprod[t]

        # Reshape for broadcasting
        while sqrt_alpha_bar.dim() < x_0.dim():
            sqrt_alpha_bar = sqrt_alpha_bar.unsqueeze(-1)
            sqrt_one_minus_alpha_bar = sqrt_one_minus_alpha_bar.unsqueeze(-1)

        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise

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
        Compute the DDPM training loss.

        L = E_{t, x_0, epsilon} [ || epsilon - epsilon_theta(x_t, t, cond) ||^2 ]

        Args:
            x_0: Clean Cholesky vector of shape (B, d*d).
            condition: Measurement vector of shape (B, cond_input_dim).
            noise: Pre-sampled noise. If None, sampled from N(0, I).

        Returns:
            Scalar loss.
        """
        B = x_0.shape[0]
        device = x_0.device

        # Sample random time steps
        t = torch.randint(0, self.timesteps, (B,), device=device)

        # Sample noise
        if noise is None:
            noise = torch.randn_like(x_0)

        # Forward diffusion
        x_t = self.q_sample(x_0, t, noise=noise)

        # Get conditioning embedding
        cond_emb, _ = self.conditioning(condition)

        # Predict noise
        noise_pred = self.denoise_fn(x_t, t, cond_emb)

        # Compute loss
        if self.loss_type == "l2":
            loss = F.mse_loss(noise_pred, noise)
        elif self.loss_type == "l1":
            loss = F.l1_loss(noise_pred, noise)
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(noise_pred, noise)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        return loss

    # ------------------------------------------------------------------
    # Reverse Process (Sampling)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(
        self,
        x_t: torch.Tensor,
        t: int,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Single DDPM reverse step: sample x_{t-1} given x_t.

        Args:
            x_t: Noisy sample at step t, shape (B, d*d).
            t: Current time step (scalar int).
            cond_emb: Conditioning embedding, shape (B, cond_dim).

        Returns:
            x_{t-1} of shape (B, d*d).
        """
        B = x_t.shape[0]
        device = x_t.device
        t_tensor = torch.full((B,), t, device=device, dtype=torch.long)

        # Predict noise
        noise_pred = self.denoise_fn(x_t, t_tensor, cond_emb)

        # Compute x_0 prediction
        alpha_bar_t = self.alphas_cumprod[t]
        sqrt_alpha_bar_t = self.sqrt_alphas_cumprod[t]
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]

        x_0_pred = (x_t - sqrt_one_minus_alpha_bar_t * noise_pred) / sqrt_alpha_bar_t

        if t == 0:
            return x_0_pred

        # Posterior mean: mu_t(x_t, x_0_pred)
        alpha_t = self.alphas[t]
        alpha_bar_prev = self.alphas_cumprod_prev[t]
        beta_t = self.betas[t]

        # mu = sqrt(alpha_bar_{t-1}) * beta_t / (1 - alpha_bar_t) * x_0_pred
        #    + sqrt(alpha_t) * (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t) * x_t
        coef_x0 = torch.sqrt(alpha_bar_prev) * beta_t / (1.0 - alpha_bar_t)
        coef_xt = torch.sqrt(alpha_t) * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
        mu = coef_x0 * x_0_pred + coef_xt * x_t

        # Posterior variance
        if t > 0:
            var = self.posterior_variance[t]
            noise = torch.randn_like(x_t)
            return mu + torch.sqrt(var) * noise
        else:
            return mu

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        batch_size: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Full DDPM reverse process: sample x_0 from noise conditioned on measurements.

        Args:
            condition: Measurement vector of shape (batch_size, cond_input_dim).
            batch_size: Override batch size. If None, use condition.shape[0].
            progress: Whether to show a progress bar.

        Returns:
            Predicted clean Cholesky vector of shape (batch_size, d*d).
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

        # Start from pure noise
        x_t = torch.randn(B, self.cholesky_dim, device=device)

        # Iteratively denoise
        iterator = reversed(range(self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc="DDPM Sampling", total=self.timesteps)

        for t in iterator:
            x_t = self.p_sample(x_t, t, cond_emb)

        return x_t

    # ------------------------------------------------------------------
    # DDIM Sampling (Accelerated)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def ddim_sample(
        self,
        condition: torch.Tensor,
        ddim_steps: int = 100,
        ddim_eta: float = 0.0,
        batch_size: Optional[int] = None,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        DDIM accelerated sampling.

        Args:
            condition: Measurement vector of shape (batch_size, cond_input_dim).
            ddim_steps: Number of DDIM steps (fewer = faster, slightly less quality).
            ddim_eta: Stochasticity parameter. 0 = fully deterministic, 1 = DDPM-like.
            batch_size: Override batch size.
            progress: Whether to show a progress bar.

        Returns:
            Predicted clean Cholesky vector of shape (batch_size, d*d).
        """
        if batch_size is not None:
            B = batch_size
            if condition.shape[0] != B:
                condition = condition[:B]
        else:
            B = condition.shape[0]

        device = next(self.parameters()).device
        condition = condition.to(device)

        # Encode condition
        cond_emb, _ = self.conditioning(condition)

        # Compute DDIM time steps (uniformly spaced)
        step_indices = torch.linspace(
            self.timesteps - 1, 0, ddim_steps, device=device
        ).long()

        # Start from pure noise
        x_t = torch.randn(B, self.cholesky_dim, device=device)

        iterator = zip(step_indices[:-1], step_indices[1:])
        if progress:
            iterator = tqdm(
                list(iterator), desc="DDIM Sampling", total=len(step_indices) - 1
            )

        for t, t_prev in iterator:
            t_tensor = torch.full((B,), t.item(), device=device, dtype=torch.long)

            # Predict noise
            noise_pred = self.denoise_fn(x_t, t_tensor, cond_emb)

            # Predict x_0
            alpha_bar_t = self.alphas_cumprod[t]
            sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alphas_cumprod[t]
            x_0_pred = (
                x_t - sqrt_one_minus_alpha_bar_t * noise_pred
            ) / torch.sqrt(alpha_bar_t)

            # Clamp x_0 prediction for stability
            x_0_pred = torch.clamp(x_0_pred, -10.0, 10.0)

            # DDIM update
            alpha_bar_prev = self.alphas_cumprod[t_prev]
            sqrt_alpha_bar_prev = torch.sqrt(alpha_bar_prev)
            pred_dir = torch.sqrt(1.0 - alpha_bar_prev - ddim_eta ** 2) * noise_pred

            x_t = sqrt_alpha_bar_prev * x_0_pred + pred_dir

            if ddim_eta > 0:
                noise = torch.randn_like(x_t)
                x_t = x_t + ddim_eta * torch.sqrt(
                    (1.0 - alpha_bar_prev) / (1.0 - alpha_bar_t)
                    * (1.0 - alpha_bar_t / alpha_bar_prev)
                ) * noise

        return x_t

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @torch.no_grad()
    def unconditional_sample(
        self,
        batch_size: int = 1,
        progress: bool = True,
    ) -> torch.Tensor:
        """
        Generate states unconditionally (without measurement conditioning).

        Uses the conditioning network with forced dropout.

        Args:
            batch_size: Number of states to generate.
            progress: Whether to show a progress bar.

        Returns:
            Cholesky vectors of shape (batch_size, d*d).
        """
        device = next(self.parameters()).device

        # Create dummy measurement (all zeros) with forced dropout
        dummy_cond = torch.zeros(
            batch_size, self.conditioning.encoder[0].in_features, device=device
        )
        cond_emb, _ = self.conditioning(dummy_cond, force_dropout=True)

        # Start from noise
        x_t = torch.randn(batch_size, self.cholesky_dim, device=device)

        iterator = reversed(range(self.timesteps))
        if progress:
            iterator = tqdm(iterator, desc="Uncond Sampling", total=self.timesteps)

        for t in iterator:
            x_t = self.p_sample(x_t, t, cond_emb)

        return x_t
