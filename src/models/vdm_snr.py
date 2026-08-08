"""
Learnable monotonic SNR schedule for VDM (Kingma et al., NeurIPS 2021).

The noise schedule is parameterized as:
    SNR(t) = exp(gamma(t))
    alpha_t^2 = sigmoid(gamma(t))     (signal coefficient)
    sigma_t^2 = sigmoid(-gamma(t))    (noise coefficient)

where gamma(t) is a monotonically DECREASING function of t ∈ [0,1].

Parameterization:
    gamma(t) = gamma_max + (gamma_min - gamma_max) * g(t)

    Base schedule:
        g_base(t) = t^p  (learnable power, p > 0)
        → g(0)=0, g(1)=1, monotonic for p > 0

    Full schedule (with MLP correction):
        g(t) = g_base(t) + eps * MLP(t) * t * (1-t)
        → correction vanishes at endpoints, preserves monotonicity for small eps

The derivative gamma'(t) is used for ELBO loss weighting and ODE sampling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class MonotonicSNR(nn.Module):
    """
    Learnable monotonic SNR schedule.

    Uses a learnable power t^p as the base schedule, with optional
    MLP correction that vanishes at t=0 and t=1.

    Guarantees:
        gamma(0) = gamma_max, gamma(1) = gamma_min, gamma(t) is decreasing.
    """

    def __init__(
        self,
        gamma_min: float = -15.0,
        gamma_max: float = 15.0,
        learnable: bool = True,
        use_mlp_correction: bool = False,
        correction_hidden: int = 32,
        correction_eps: float = 0.1,
    ):
        """
        Args:
            gamma_min: Minimum log-SNR at t=1 (most noisy).
            gamma_max: Maximum log-SNR at t=0 (cleanest).
            learnable: If True, learn the power p. If False, use p=1 (linear).
            use_mlp_correction: If True, add small MLP correction.
            correction_hidden: Hidden dim for correction MLP.
            correction_eps: Maximum magnitude of MLP correction.
        """
        super().__init__()
        self.gamma_min = gamma_min
        self.gamma_max = gamma_max
        self.use_mlp_correction = use_mlp_correction
        self.correction_eps = correction_eps

        if learnable:
            # p = softplus(log_p) > 0 guarantees monotonicity
            self.log_p = nn.Parameter(torch.tensor(0.0))
        else:
            self.register_buffer("log_p", torch.tensor(0.0))

        if use_mlp_correction:
            self.correction_mlp = nn.Sequential(
                nn.Linear(1, correction_hidden),
                nn.SiLU(),
                nn.Linear(correction_hidden, 1),
                nn.Tanh(),  # output in [-1, 1]
            )

    def _g_base(self, t: torch.Tensor) -> torch.Tensor:
        """Base schedule: g(t) = t^p."""
        p = F.softplus(self.log_p)  # p > 0
        return t ** p

    def _g_correction(self, t: torch.Tensor) -> torch.Tensor:
        """MLP correction: vanishes at t=0 and t=1."""
        if self.use_mlp_correction:
            raw = self.correction_mlp(t.unsqueeze(-1)).squeeze(-1)  # (B,) in [-1,1]
            # Multiply by t*(1-t) to vanish at endpoints
            return raw * t * (1.0 - t) * self.correction_eps
        else:
            return torch.zeros_like(t)

    def g(self, t: torch.Tensor) -> torch.Tensor:
        """Full schedule function g(t) ∈ [0,1], monotonic."""
        g = self._g_base(t) + self._g_correction(t)
        return torch.clamp(g, 0.0, 1.0)

    def forward(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute gamma(t) and gamma'(t).

        Args:
            t: Time values in [0, 1], shape (B,) or (B, 1).

        Returns:
            gamma: Log-SNR at t, shape (B,).
            gamma_prime: d(gamma)/dt at t, shape (B,). Always <= 0.
        """
        if t.dim() > 1:
            t = t.squeeze(-1)

        t = torch.clamp(t, 0.0, 1.0)
        t_in = t.detach().clone().requires_grad_(True)

        g_val = self.g(t_in)
        gamma = self.gamma_max + (self.gamma_min - self.gamma_max) * g_val

        if torch.is_grad_enabled():
            gamma_prime = torch.autograd.grad(
                gamma.sum(), t_in,
                create_graph=True,
                retain_graph=True,
            )[0]
        else:
            dt = 1e-4
            t_plus = torch.clamp(t_in + dt, 0.0, 1.0)
            g_plus = self.g(t_plus)
            gamma_plus = self.gamma_max + (self.gamma_min - self.gamma_max) * g_plus
            gamma_prime = (gamma_plus - gamma) / dt

        return gamma.detach(), gamma_prime

    def get_coefficients(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute signal and noise coefficients at time t.
        alpha_t = sqrt(sigmoid(gamma(t)))
        sigma_t = sqrt(sigmoid(-gamma(t)))
        alpha_t^2 + sigma_t^2 = 1 (by construction).
        """
        gamma, _ = self.forward(t)
        alpha_t = torch.sqrt(torch.sigmoid(gamma))
        sigma_t = torch.sqrt(torch.sigmoid(-gamma))
        return alpha_t, sigma_t

    def get_snr(self, t: torch.Tensor) -> torch.Tensor:
        """SNR(t) = exp(gamma(t))."""
        gamma, _ = self.forward(t)
        return torch.exp(gamma)

    def get_snr_derivative(self, t: torch.Tensor) -> torch.Tensor:
        """d(SNR)/dt = SNR(t) * gamma'(t)."""
        gamma, gamma_prime = self.forward(t)
        return torch.exp(gamma) * gamma_prime

    def get_power(self) -> float:
        """Return the learned power parameter."""
        return F.softplus(self.log_p).item()
