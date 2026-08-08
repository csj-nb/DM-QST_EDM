"""
Validation functions for density matrices.

Checks that a matrix satisfies the required properties:
    1. Hermiticity: rho = rho^dagger
    2. Positive semi-definiteness: all eigenvalues >= 0
    3. Unit trace: Tr(rho) = 1
"""

import numpy as np


def is_valid_dm(
    rho: np.ndarray,
    atol_herm: float = 1e-10,
    atol_psd: float = 1e-10,
    atol_trace: float = 1e-10,
) -> bool:
    """
    Check if a matrix is a valid density matrix.

    Args:
        rho: Matrix of shape (d, d) or batch of shape (..., d, d).
        atol_herm: Absolute tolerance for Hermiticity check.
        atol_psd: Absolute tolerance for PSD check.
        atol_trace: Absolute tolerance for trace-1 check.

    Returns:
        True if the matrix satisfies all density matrix constraints.
    """
    rho = np.asarray(rho)
    if rho.ndim == 2:
        return _single_is_valid_dm(rho, atol_herm, atol_psd, atol_trace)
    else:
        batch_shape = rho.shape[:-2]
        rho_flat = rho.reshape(-1, rho.shape[-2], rho.shape[-1])
        return all(
            _single_is_valid_dm(r, atol_herm, atol_psd, atol_trace)
            for r in rho_flat
        )


def _single_is_valid_dm(
    rho: np.ndarray,
    atol_herm: float,
    atol_psd: float,
    atol_trace: float,
) -> bool:
    """Check a single density matrix."""
    d = rho.shape[0]

    # Check square
    if rho.shape != (d, d):
        return False

    # Check Hermiticity
    if not np.allclose(rho, rho.T.conj(), atol=atol_herm):
        return False

    # Check trace = 1
    if not np.isclose(np.trace(rho), 1.0, atol=atol_trace):
        return False

    # Check PSD: all eigenvalues >= -atol_psd
    eigvals = np.linalg.eigvalsh(rho)
    if np.any(eigvals < -atol_psd):
        return False

    return True


def check_dm_constraints(rho: np.ndarray) -> dict:
    """
    Detailed constraint checking with diagnostic information.

    Args:
        rho: Density matrix of shape (d, d).

    Returns:
        Dictionary with constraint check results and values.
    """
    d = rho.shape[0]
    result = {
        "shape": rho.shape,
        "is_square": rho.shape == (d, d),
    }

    # Hermiticity
    herm_error = np.max(np.abs(rho - rho.T.conj()))
    result["hermiticity_error"] = herm_error
    result["is_hermitian"] = np.allclose(rho, rho.T.conj(), atol=1e-10)

    # Trace
    trace_val = np.real(np.trace(rho))
    result["trace"] = trace_val
    result["trace_is_one"] = np.isclose(trace_val, 1.0, atol=1e-10)

    # Eigenvalues
    eigvals = np.linalg.eigvalsh(rho)
    result["eigenvalues"] = eigvals
    result["min_eigenvalue"] = np.min(eigvals)
    result["max_eigenvalue"] = np.max(eigvals)
    result["is_psd"] = np.all(eigvals >= -1e-10)

    # Derived properties
    result["purity"] = np.real(np.trace(rho @ rho))
    result["rank"] = np.sum(eigvals > 1e-10)
    result["is_valid"] = (
        result["is_hermitian"] and result["trace_is_one"] and result["is_psd"]
    )

    return result
