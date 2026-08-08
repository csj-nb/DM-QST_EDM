"""
Physics-informed auxiliary losses for VDM-QST.

Implements differentiable quantum state constraints that can be
added to the VDM training objective:

    L_total = L_ELBO + lambda_rank * L_rank + ...

All losses operate on the Cholesky vector prediction (B, d*d) and
use differentiable PyTorch operations for end-to-end training.
"""

import torch
import torch.nn.functional as F
from typing import Optional


def vector_to_dm(x: torch.Tensor, d: int) -> torch.Tensor:
    """
    Differentiable Cholesky vector → density matrix conversion.

    Layout: first d values = real diagonal, remaining = interleaved (re, im)
            of strictly lower-triangular entries, column-major.
    """
    B = x.shape[0]
    device = x.device

    L_real = torch.zeros(B, d, d, device=device)
    L_imag = torch.zeros(B, d, d, device=device)

    # Diagonal
    L_real[:, torch.arange(d), torch.arange(d)] = x[:, :d]

    # Strictly lower-triangular
    idx = d
    for col in range(d):
        for row in range(col + 1, d):
            L_real[:, row, col] = x[:, idx]
            L_imag[:, row, col] = x[:, idx + 1]
            idx += 2

    L = torch.complex(L_real, L_imag)
    Lh = L.transpose(-2, -1).conj()
    M = torch.bmm(L, Lh)
    trace = torch.real(torch.diagonal(M, dim1=-2, dim2=-1).sum(-1))
    rho = M / (trace.unsqueeze(-1).unsqueeze(-1) + 1e-10)
    rho = (rho + rho.transpose(-2, -1).conj()) / 2.0
    return rho


def rank_penalty(x_pred: torch.Tensor, d: int, method: str = "purity") -> torch.Tensor:
    """
    Low-rank penalty on predicted density matrix.

    Args:
        x_pred: Predicted Cholesky vector (B, d*d).
        d: Hilbert space dimension.
        method: Penalty type.
            - "purity": 1 - Tr(rho^2), penalizes high-rank states
            - "entropy": von Neumann entropy (negative, so maximize)

    Returns:
        Scalar penalty (lower = more low-rank).
    """
    rho = vector_to_dm(x_pred, d)

    if method == "purity":
        # Tr(rho^2) ∈ [1/d, 1]. Penalize deviation from 1 (pure).
        rho_sq = torch.bmm(rho, rho)
        purity = torch.real(torch.diagonal(rho_sq, dim1=-2, dim2=-1).sum(-1))
        return (1.0 - purity).mean()

    elif method == "entropy":
        # S(rho) = -Tr(rho log rho). Penalize high entropy.
        # Use eigenvalue decomposition (stable for small d)
        # Convert complex to 2*d real representation for eigh
        rho_real = torch.cat([
            torch.cat([rho.real, -rho.imag], dim=-1),
            torch.cat([rho.imag, rho.real], dim=-1),
        ], dim=-2)  # (B, 2d, 2d) real representation

        eigvals = torch.linalg.eigvalsh(rho_real)  # (B, 2d) — each repeated twice
        eigvals = eigvals[:, d:]  # Take unique eigenvalues (last d)
        eigvals = torch.clamp(eigvals, min=1e-10)
        entropy = -(eigvals * torch.log(eigvals)).sum(-1)  # (B,)
        return entropy.mean()  # Penalize: lower entropy = lower rank

    else:
        raise ValueError(f"Unknown rank penalty method: {method}")


def trace_norm_penalty(x_pred: torch.Tensor, d: int) -> torch.Tensor:
    """
    Penalize deviation from unit trace (should already be close, but a safeguard).
    """
    rho = vector_to_dm(x_pred, d)
    trace = torch.real(torch.diagonal(rho, dim1=-2, dim2=-1).sum(-1))
    return F.mse_loss(trace, torch.ones_like(trace))


def hermiticity_penalty(x_pred: torch.Tensor, d: int) -> torch.Tensor:
    """Penalize non-Hermitian density matrices (numerical safeguard)."""
    rho = vector_to_dm(x_pred, d)
    diff = rho - rho.transpose(-2, -1).conj()
    return torch.abs(diff).mean()
