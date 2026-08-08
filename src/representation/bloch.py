"""Isometric Bloch parameterization of density matrices.

Vector layout (15 real components for 2 qubits, d=4):
    y[i] = (d/2) * Tr(rho lam_i),  i = 1..d^2-1
where {lam_i} are the generalized Gell-Mann generators (SU(d) basis),
Tr(lam_i lam_j) = 2 delta_ij.

rho(y) = I/d + sum_i y_i lam_i / d

Properties (numerically verified, see herm_geometry_verify.json):
    - Perfect isometry: J^T J = (2/d) * I  ->  diag ratio 1.00x, spectral 1.00x
      (every coordinate direction is equally sensitive; no per-coordinate
       sigma_data correction needed)
    - Trace-1 by construction (lam_i are traceless)
    - NOT PSD by construction: use project_to_valid_*() after sampling
      (same post-processing as the Hermitian direct parameterization)
"""
import numpy as np

# torch is imported lazily inside the torch functions so that numpy-only
# environments (e.g. server-side sigma calibration) can use this module.
_LAM_NP = None
_LAM_TORCH = None


def _gellmann_basis(d):
    """Generalized Gell-Mann generators: Tr(lam_i lam_j) = 2 delta_ij."""
    mats = []
    for i in range(d):            # symmetric (i<j)
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=np.complex128)
            M[i, j] = 1.0
            M[j, i] = 1.0
            mats.append(M)
    for i in range(d):            # antisymmetric (i<j)
        for j in range(i + 1, d):
            M = np.zeros((d, d), dtype=np.complex128)
            M[i, j] = -1j
            M[j, i] = 1j
            mats.append(M)
    for k in range(1, d):         # diagonal (k=1..d-1)
        M = np.zeros((d, d), dtype=np.complex128)
        c = np.sqrt(2.0 / (k * (k + 1)))
        for m in range(k):
            M[m, m] = c
        M[k, k] = -k * c
        mats.append(M)
    return np.stack(mats)         # (d^2-1, d, d)


def _lams(d):
    global _LAM_NP
    if _LAM_NP is None or _LAM_NP.shape[-1] != d:
        _LAM_NP = _gellmann_basis(d)
    return _LAM_NP


def dm_to_vec_np(rho):
    """(..., d, d) complex rho -> (..., d*d-1) Bloch coefficients."""
    rho = np.asarray(rho)
    shape = rho.shape
    d = shape[-1]
    lams = _lams(d)
    flat = rho.reshape(-1, d, d)
    # r_i = (d/2) * Re(Tr(rho lam_i)) = (d/2) * Re(sum_ij rho_ij lam_ji)
    coeffs = (d / 2.0) * np.einsum("bij,kji->bk", flat, lams).real
    return coeffs.reshape(shape[:-2] + (d * d - 1,))


def vec_to_dm_np(y):
    """(..., d*d-1) Bloch coeffs -> (..., d, d) rho (trace-1, not PSD-guaranteed)."""
    y = np.asarray(y)
    shape = y.shape
    d = int(round(np.sqrt(shape[-1] + 1)))
    lams = _lams(d)
    flat = y.reshape(-1, d * d - 1)
    rho = (np.eye(d, dtype=np.complex128) / d).reshape(1, d, d) \
        + np.einsum("bk,kij->bij", flat, lams) / d
    return rho.reshape(shape[:-1] + (d, d))


def _lams_torch(d, device, dtype):
    global _LAM_TORCH
    import torch
    if _LAM_TORCH is None or _LAM_TORCH.device != device or _LAM_TORCH.dtype != dtype:
        _LAM_TORCH = torch.from_numpy(_lams(d)).to(device=device, dtype=dtype)
    return _LAM_TORCH


def dm_to_vec_torch(rho, d):
    """(B, d, d) complex -> (B, d*d-1) real Bloch coeffs (differentiable)."""
    import torch
    lams = _lams_torch(d, rho.device, rho.dtype)
    coeffs = (d / 2.0) * torch.einsum("bij,kji->bk", rho, lams).real
    return coeffs


def vec_to_dm_torch(y, d):
    """(B, d*d-1) real -> (B, d, d) complex rho (differentiable)."""
    import torch
    lams = _lams_torch(d, y.device, torch.complex64 if y.dtype == torch.float32 else torch.complex128)
    b = y.shape[0]
    eye = torch.eye(d, dtype=lams.dtype, device=y.device) / d
    yc = y.to(lams.dtype)  # real -> complex to match lams
    return eye.unsqueeze(0).expand(b, d, d) + torch.einsum("bk,kij->bij", yc, lams) / d


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
