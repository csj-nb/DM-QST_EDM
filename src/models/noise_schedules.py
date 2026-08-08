"""
Noise schedules for DDPM.

Supports linear and cosine beta schedules. The schedule controls how
quickly information is destroyed during the forward diffusion process.

Key quantities:
    beta_t: variance of noise added at step t
    alpha_t = 1 - beta_t
    alpha_bar_t = prod_{s=1}^t alpha_s  (cumulative product)

    x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * epsilon
    where epsilon ~ N(0, I)

Reference:
    - "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal, 2021)
      for the cosine schedule.
"""

import torch
import torch.nn.functional as F
import numpy as np


def make_beta_schedule(
    timesteps: int,
    schedule: str = "cosine",
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
    s: float = 0.008,
) -> torch.Tensor:
    """
    Create a beta schedule for the diffusion process.

    Args:
        timesteps: Total number of diffusion steps T.
        schedule: Type of schedule, "linear" or "cosine".
        beta_start: Starting beta value (for linear schedule).
        beta_end: Ending beta value (for linear schedule).
        s: Offset for cosine schedule (controls endpoint noise level).

    Returns:
        Tensor of shape (timesteps,) with beta values.
    """
    if schedule == "linear":
        return _linear_beta_schedule(timesteps, beta_start, beta_end)
    elif schedule == "cosine":
        return _cosine_beta_schedule(timesteps, s)
    elif schedule == "sigmoid":
        return _sigmoid_beta_schedule(timesteps, beta_start, beta_end)
    else:
        raise ValueError(f"Unknown schedule: {schedule}")


def _linear_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Linear interpolation between beta_start and beta_end."""
    return torch.linspace(beta_start, beta_end, timesteps)


def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Cosine schedule as proposed in Nichol & Dhariwal (2021).

    alpha_bar_t = cos((t/T + s) / (1 + s) * pi/2)^2 / cos(s / (1 + s) * pi/2)^2
    beta_t = 1 - alpha_bar_t / alpha_bar_{t-1}

    The cosine schedule adds noise more slowly at the beginning, which is
    beneficial for learning fine details.
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps)

    # Compute alpha_bar
    ft = torch.cos(((t / timesteps + s) / (1 + s)) * (np.pi / 2)) ** 2
    # Normalize so alpha_bar_0 = 1
    alpha_bar = ft / ft[0]

    # Compute beta from alpha_bar
    beta = 1 - alpha_bar[1:] / alpha_bar[:-1]
    beta = torch.clamp(beta, max=0.999)
    return beta


def _sigmoid_beta_schedule(
    timesteps: int,
    beta_start: float = 1e-4,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """Sigmoid (geometric) schedule."""
    betas = torch.linspace(-6, 6, timesteps)
    return torch.sigmoid(betas) * (beta_end - beta_start) + beta_start


def compute_diffusion_parameters(
    betas: torch.Tensor,
) -> dict:
    """
    Compute all derived diffusion parameters from beta schedule.

    Args:
        betas: Beta schedule of shape (T,).

    Returns:
        Dict containing:
            - betas: (T,)
            - alphas: (T,) = 1 - betas
            - alphas_cumprod: (T,) cumulative product of alphas
            - alphas_cumprod_prev: (T,) shifted cumulative product
            - sqrt_alphas_cumprod: (T,) = sqrt(alphas_cumprod)
            - sqrt_one_minus_alphas_cumprod: (T,) = sqrt(1 - alphas_cumprod)
            - sqrt_recip_alphas_cumprod: (T,)
            - sqrt_recipm1_alphas_cumprod: (T,)
            - posterior_variance: (T,) variance of q(x_{t-1} | x_t, x_0)
    """
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

    # For q(x_t | x_0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)

    # For q(x_{t-1} | x_t, x_0) posterior
    sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
    sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1.0)

    # Posterior variance: beta_t * (1 - alpha_{t-1}) / (1 - alpha_t)
    posterior_variance = (
        betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
    )

    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "alphas_cumprod_prev": alphas_cumprod_prev,
        "sqrt_alphas_cumprod": sqrt_alphas_cumprod,
        "sqrt_one_minus_alphas_cumprod": sqrt_one_minus_alphas_cumprod,
        "sqrt_recip_alphas_cumprod": sqrt_recip_alphas_cumprod,
        "sqrt_recipm1_alphas_cumprod": sqrt_recipm1_alphas_cumprod,
        "posterior_variance": posterior_variance,
    }
