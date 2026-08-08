"""
Metrics for evaluating quantum state tomography quality.

Key metrics:
    - Fidelity: F(rho, sigma) = [Tr sqrt(sqrt(rho) * sigma * sqrt(rho))]^2
    - Trace distance: T(rho, sigma) = 0.5 * Tr |rho - sigma|
    - Purity: Tr(rho^2)
    - von Neumann entropy: -Tr(rho * log(rho))
"""

import numpy as np
from scipy.linalg import sqrtm
from typing import Optional, Tuple


def fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the Uhlmann fidelity between two density matrices.

    F(rho, sigma) = [Tr sqrt(sqrt(rho) * sigma * sqrt(rho))]^2

    For pure states, this reduces to |<psi|phi>|^2.

    Args:
        rho: First density matrix of shape (d, d) or batch (..., d, d).
        sigma: Second density matrix of shape (d, d) or batch (..., d, d).

    Returns:
        Fidelity value(s) in [0, 1].
    """
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)

    if rho.ndim == 2:
        return _fidelity_single(rho, sigma)
    else:
        batch_shape = rho.shape[:-2]
        rho_flat = rho.reshape(-1, rho.shape[-2], rho.shape[-1])
        sigma_flat = sigma.reshape(-1, sigma.shape[-2], sigma.shape[-1])
        fids = np.array([
            _fidelity_single(r, s) for r, s in zip(rho_flat, sigma_flat)
        ])
        return fids.reshape(batch_shape)


def _fidelity_single(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute fidelity for a single pair of density matrices."""
    # Ensure Hermitian
    rho = (rho + rho.T.conj()) / 2.0
    sigma = (sigma + sigma.T.conj()) / 2.0

    # Compute eigenvalues of rho
    eigvals, eigvecs = np.linalg.eigh(rho)
    # Clamp negative eigenvalues to zero (numerical noise)
    eigvals = np.clip(eigvals, 0, None)

    # sqrt(rho)
    sqrt_eigvals = np.sqrt(eigvals)
    sqrt_rho = eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T.conj()

    # sqrt(sqrt(rho) * sigma * sqrt(rho))
    M = sqrt_rho @ sigma @ sqrt_rho
    M = (M + M.T.conj()) / 2.0

    # Compute sqrt of M
    eigvals_M = np.linalg.eigvalsh(M)
    eigvals_M = np.clip(eigvals_M, 0, None)
    sqrt_eigvals_M = np.sqrt(eigvals_M)

    fid = np.sum(sqrt_eigvals_M) ** 2
    return np.clip(np.real(fid), 0.0, 1.0)


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Compute the trace distance between two density matrices.

    T(rho, sigma) = 0.5 * Tr |rho - sigma|

    Args:
        rho: First density matrix of shape (d, d) or batch (..., d, d).
        sigma: Second density matrix of shape (d, d) or batch (..., d, d).

    Returns:
        Trace distance(s) in [0, 1].
    """
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)

    diff = rho - sigma
    if diff.ndim == 2:
        return _trace_distance_single(diff)
    else:
        batch_shape = diff.shape[:-2]
        diff_flat = diff.reshape(-1, diff.shape[-2], diff.shape[-1])
        tds = np.array([_trace_distance_single(d) for d in diff_flat])
        return tds.reshape(batch_shape)


def _trace_distance_single(diff: np.ndarray) -> float:
    """Compute trace distance for a single difference matrix."""
    # Ensure Hermitian
    diff = (diff + diff.T.conj()) / 2.0
    eigvals = np.linalg.eigvalsh(diff)
    return 0.5 * np.sum(np.abs(eigvals))


def purity(rho: np.ndarray) -> float:
    """
    Compute the purity of a density matrix.

    P(rho) = Tr(rho^2)

    For pure states, P = 1. For maximally mixed states, P = 1/d.

    Args:
        rho: Density matrix of shape (d, d) or batch (..., d, d).

    Returns:
        Purity value(s) in [1/d, 1].
    """
    rho = np.asarray(rho)
    if rho.ndim == 2:
        return np.real(np.trace(rho @ rho))
    else:
        batch_shape = rho.shape[:-2]
        rho_flat = rho.reshape(-1, rho.shape[-2], rho.shape[-1])
        purities = np.array([np.real(np.trace(r @ r)) for r in rho_flat])
        return purities.reshape(batch_shape)


def von_neumann_entropy(rho: np.ndarray) -> float:
    """
    Compute the von Neumann entropy of a density matrix.

    S(rho) = -Tr(rho * log2(rho))

    Args:
        rho: Density matrix of shape (d, d) or batch (..., d, d).

    Returns:
        Entropy value(s) in [0, log2(d)].
    """
    rho = np.asarray(rho)
    if rho.ndim == 2:
        return _entropy_single(rho)
    else:
        batch_shape = rho.shape[:-2]
        rho_flat = rho.reshape(-1, rho.shape[-2], rho.shape[-1])
        ents = np.array([_entropy_single(r) for r in rho_flat])
        return ents.reshape(batch_shape)


def _entropy_single(rho: np.ndarray) -> float:
    """Compute von Neumann entropy for a single density matrix."""
    eigvals = np.linalg.eigvalsh(rho)
    eigvals = np.clip(eigvals, 1e-15, None)  # Avoid log(0)
    return -np.sum(eigvals * np.log2(eigvals))


def average_fidelity(
    rhos_pred: np.ndarray,
    rhos_true: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute average fidelity and standard deviation across a batch.

    Args:
        rhos_pred: Predicted density matrices of shape (N, d, d).
        rhos_true: True density matrices of shape (N, d, d).

    Returns:
        Tuple of (mean_fidelity, std_fidelity).
    """
    fids = fidelity(rhos_pred, rhos_true)
    return float(np.mean(fids)), float(np.std(fids))


def hilbert_schmidt_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """
    Hilbert--Schmidt distance between two density matrices.

    d_HS(rho, sigma) = ||rho - sigma||_F
                     = sqrt(Tr[(rho - sigma)^dag (rho - sigma)])

    This is the distance whose expected SQUARED value the MMSE estimator
    (posterior mean) minimizes -- the theory-side metric of the Bayesian
    claim (see sections/theory.tex). Evaluate it alongside fidelity so the
    theoretical commitment (MMSE optimality in HS loss) can be checked
    empirically, not just the fidelity outcome.

    For d-dimensional states, d_HS <= sqrt(2) (attained by orthogonal pure
    states). Batch support mirrors fidelity(): pass (..., d, d) arrays.

    Args:
        rho: Density matrix of shape (d, d) or batch (..., d, d).
        sigma: Density matrix of shape (d, d) or batch (..., d, d).

    Returns:
        HS distance (float) or array of distances.
    """
    rho = np.asarray(rho)
    sigma = np.asarray(sigma)
    if rho.ndim == 2:
        return _hs_distance_single(rho, sigma)
    batch_shape = rho.shape[:-2]
    rho_flat = rho.reshape(-1, rho.shape[-2], rho.shape[-1])
    sigma_flat = sigma.reshape(-1, sigma.shape[-2], sigma.shape[-1])
    dists = np.array([
        _hs_distance_single(r, s) for r, s in zip(rho_flat, sigma_flat)
    ])
    return dists.reshape(batch_shape)


def _hs_distance_single(rho: np.ndarray, sigma: np.ndarray) -> float:
    """Compute the Hilbert--Schmidt distance for a single pair."""
    diff = rho - sigma
    # Frobenius norm: sqrt(Tr[(rho-sigma)^dag (rho-sigma)])
    return float(np.sqrt(np.real(np.trace(diff.conj().T @ diff))))
