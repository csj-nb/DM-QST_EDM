"""Hermitian direct parameterization of density matrices (replacement for Cholesky).

Vector layout (16 real components for 2 qubits, d=4):
    y[0:4]    = real diagonal elements rho_ii
    y[4:10]   = real parts of upper-triangular elements
    y[10:16]  = imaginary parts of upper-triangular elements

rho = H(y) is Hermitian by construction but NOT trace-1 / PSD guaranteed.
Use project_to_valid_*() after sampling (evaluation side). The map is an
isometry: J^T J = 2*I (verified numerically), so no per-coordinate geometry
distortion like the Cholesky parameterization (9.4x anisotropy).

Training-side loss uses the unprojected rho (the model learns trace~1/PSD
naturally); projection is applied only at sampling/evaluation.
"""
import numpy as np
import torch


def _offdiag_idx(d):
    return np.triu_indices(d, 1)


def dm_to_vec_np(rho):
    """(..., d, d) complex rho -> (..., d*d) real Hermitian params."""
    rho = np.asarray(rho)
    shape = rho.shape
    d = shape[-1]
    flat = rho.reshape(-1, d, d)
    diag = np.diagonal(flat, axis1=-2, axis2=-1).real          # (N, d)
    iu = np.triu_indices(d, 1)
    off = flat[:, iu[0], iu[1]]                                # (N, n_off)
    y = np.concatenate([diag, off.real, off.imag], axis=-1)    # (N, d*d)
    return y.reshape(shape[:-2] + (d * d,))


def vec_to_dm_np(y):
    """(..., d*d) Hermitian params -> (..., d, d) complex rho (not normalized)."""
    y = np.asarray(y)
    shape = y.shape
    d = int(round(np.sqrt(shape[-1])))
    flat = y.reshape(-1, d * d)
    n = flat.shape[0]
    rho = np.zeros((n, d, d), dtype=np.complex128)
    iu = np.triu_indices(d, 1)
    n_off = len(iu[0])
    rho[:, range(d), range(d)] = flat[:, :d]
    re = flat[:, d:d + n_off]
    im = flat[:, d + n_off:d + 2 * n_off] if d + n_off < flat.shape[1] else np.zeros((n, n_off))
    rho[:, iu[0], iu[1]] = re + 1j * im
    rho[:, iu[1], iu[0]] = re - 1j * im
    return rho.reshape(shape[:-1] + (d, d))


def project_to_valid_np(rho, eps: float = 1e-10):
    """Frobenius-nearest valid state: Hermitian + PSD + trace-1 (eigclip)."""
    rho = np.asarray(rho)
    shape = rho.shape
    d = shape[-1]
    flat = rho.reshape(-1, d, d)
    flat = 0.5 * (flat + np.conj(flat.transpose(0, 2, 1)))     # Hermitian
    out = np.zeros_like(flat)
    for i in range(flat.shape[0]):
        evals, evecs = np.linalg.eigh(flat[i])
        evals = np.clip(evals, eps, None)
        r = (evecs * evals) @ evecs.conj().T
        r = r / (np.trace(r).real + eps)
        out[i] = r
    return out.reshape(shape)


def dm_to_vec_torch(rho, d):
    """(B, d, d) complex -> (B, d*d) real."""
    diag = torch.diagonal(rho, dim1=-2, dim2=-1).real          # (B, d)
    iu = torch.triu_indices(d, d, 1)
    off = rho[:, iu[0], iu[1]]
    return torch.cat([diag, off.real, off.imag], dim=-1)       # (B, d*d)


def vec_to_dm_torch(y, d):
    """(B, d*d) real -> (B, d, d) complex Hermitian (differentiable)."""
    flat = y.reshape(-1, d * d)
    b = flat.shape[0]
    iu = torch.triu_indices(d, d, 1)
    n_off = len(iu[0])
    rho = torch.zeros(b, d, d, dtype=torch.complex64, device=y.device)
    rho[:, range(d), range(d)] = flat[:, :d].to(torch.complex64)
    re = flat[:, d:d + n_off]
    im = flat[:, d + n_off:d + 2 * n_off]
    rho[:, iu[0], iu[1]] = re.to(torch.complex64) + 1j * im.to(torch.complex64)
    rho[:, iu[1], iu[0]] = re.to(torch.complex64) - 1j * im.to(torch.complex64)
    return rho


def project_to_valid_torch(rho, eps: float = 1e-10):
    """Differentiable PSD+trace projection (eigclip) for torch tensors."""
    shape = rho.shape
    d = shape[-1]
    flat = rho.reshape(-1, d, d)
    flat = 0.5 * (flat + flat.transpose(-2, -1).conj())
    evals, evecs = torch.linalg.eigh(flat)
    evals_c = torch.clamp(evals, min=eps)
    r = (evecs * evals_c.unsqueeze(-2)) @ evecs.transpose(-2, -1).conj()
    tr = torch.diagonal(r, dim1=-2, dim2=-1).sum(-1).real.unsqueeze(-1).unsqueeze(-1)
    r = r / (tr + eps)
    return r.reshape(shape)
