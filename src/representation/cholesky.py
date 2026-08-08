"""
Cholesky representation of density matrices.

A density matrix rho (d x d, Hermitian, PSD, trace-1) is parameterized as:
    rho = L L^dagger / Tr(L L^dagger)
where L is a d x d lower-triangular complex matrix.

This parameterization automatically enforces:
    - Hermiticity (by construction of L L^dagger)
    - Positive semi-definiteness (by construction)
    - Unit trace (by normalization)

The Cholesky vector x in R^{d^2} is formed by flattening L:
    - First d elements: real diagonal entries of L
    - Remaining d(d-1) elements: real and imaginary parts of strictly
      lower-triangular entries, ordered column-wise, alternating (re, im).

For a d-dimensional Hilbert space (n qubits, d = 2^n):
    - d^2 real parameters
    - This matches the d^2 - 1 degrees of freedom of a density matrix
      (one parameter is redundant due to trace normalization).
"""

import numpy as np
from scipy.linalg import cholesky as scipy_cholesky
from scipy.linalg import LinAlgError


def dm_to_cholesky(
    rho: np.ndarray,
    eps: float = 1e-6,
    normalize: bool = False,
) -> np.ndarray:
    """
    Convert a density matrix to its Cholesky vector representation.

    Args:
        rho: Density matrix of shape (..., d, d), Hermitian, PSD, trace-1.
        eps: Regularization constant for numerical stability.
             rho_reg = (1 - eps) * rho + eps * I/d
        normalize: If True, divide the vector by its L2 norm (||x||_2 = 1).
            Because rho = L L^dagger / Tr(L L^dagger) is invariant under the
            global scaling L -> c L (equivalently x -> c x), the scaling
            direction is a 1-dimensional redundancy of the Cholesky
            parameterization (the Jacobian d rho / d x has a null direction
            exactly along x, verified numerically). Normalizing removes this
            redundant degree of freedom: the data then lives on the unit
            sphere of the d^2-1 effective dimensions, so the diffusion model
            no longer spends capacity or noise energy on the scaling
            direction. Decoding is unaffected: cholesky_to_dm is scale-
            invariant by construction.

    Returns:
        Cholesky vector of shape (..., d*d). Real-valued.
    """
    # Ensure real diagonal (Hermitian matrices should have real diagonal)
    rho = np.asarray(rho)
    d = rho.shape[-1]

    # Regularize to avoid singular Cholesky
    identity = np.eye(d, dtype=rho.dtype)
    if eps > 0:
        rho_reg = (1.0 - eps) * rho + eps * identity / d
    else:
        rho_reg = rho

    # Handle batch dimensions
    if rho.ndim == 2:
        x = _single_dm_to_cholesky(rho_reg, d)
        return _normalize(x) if normalize else x
    else:
        batch_shape = rho.shape[:-2]
        rho_reg = rho_reg.reshape(-1, d, d)
        results = np.array([_single_dm_to_cholesky(r, d) for r in rho_reg])
        results = results.reshape(*batch_shape, d * d)
        if normalize:
            results = _normalize(results)
        return results


def _normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Divide the Cholesky vector by its L2 norm (scale-invariant direction)."""
    x = np.asarray(x, dtype=np.float64)
    norm = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / (norm + eps)


def _single_dm_to_cholesky(rho: np.ndarray, d: int) -> np.ndarray:
    """
    Convert a single density matrix to Cholesky vector.

    Steps:
    1. Compute Cholesky decomposition: rho = L L^dagger
       (scipy returns lower-triangular L)
    2. Extract real parameters from L into a vector.
    """
    try:
        L = scipy_cholesky(rho, lower=True)
    except LinAlgError:
        # Fallback: force Hermitian and try again with stronger regularization
        rho_fixed = (rho + rho.T.conj()) / 2.0
        eigvals = np.linalg.eigvalsh(rho_fixed)
        min_eig = np.min(eigvals)
        if min_eig < 0:
            rho_fixed = rho_fixed - min_eig * np.eye(d, dtype=rho.dtype)
            rho_fixed = rho_fixed / np.trace(rho_fixed)
        L = scipy_cholesky(rho_fixed, lower=True)

    return _cholesky_matrix_to_vector(L, d)


def _cholesky_matrix_to_vector(L: np.ndarray, d: int) -> np.ndarray:
    """
    Flatten a lower-triangular Cholesky matrix L into a real vector.

    Layout (COLUMN-MAJOR, matching unet.py / vdm.py / vdm_losses.py):
        x[0:d]                 = real(L[i, i])   for i = 0, ..., d-1
        x[d:]                  = real and imag parts of strictly lower-triangular
                                 entries, ordered column by column:
                                 for each column j, for each row i > j:
                                     x[.] = real(L[i, j])
                                     x[.] = imag(L[i, j])

    NOTE: we must NOT use np.tril_indices here: it returns row-major order,
    which only coincides with this layout for d <= 3. For d >= 4 (2+ qubits)
    the two orders differ, and mixing them scrambles the 2D image that the
    UNet sees. This was the root cause of the numpy/torch layout mismatch.
    """
    # Extract real diagonal
    diag_real = np.real(np.diag(L))  # shape (d,)

    # Strictly lower-triangular entries (i > j), COLUMN-major:
    # for col in range(d): for row in range(col+1, d)
    n_lower = d * (d - 1) // 2
    lower_flat = np.empty(2 * n_lower, dtype=np.float64)
    idx = 0
    for col in range(d):
        for row in range(col + 1, d):
            lower_flat[idx] = np.real(L[row, col])
            lower_flat[idx + 1] = np.imag(L[row, col])
            idx += 2

    return np.concatenate([diag_real, lower_flat])


def cholesky_to_dm(x: np.ndarray) -> np.ndarray:
    """
    Convert a Cholesky vector back to a valid density matrix.

    Args:
        x: Cholesky vector of shape (..., d*d). Real-valued.

    Returns:
        Density matrix of shape (..., d, d). Guaranteed Hermitian, PSD, trace-1.
    """
    x = np.asarray(np.real(x))  # Ensure real
    if x.ndim == 1:
        return _single_cholesky_to_dm(x)
    else:
        batch_shape = x.shape[:-1]
        n_batch = int(np.prod(batch_shape))
        x_flat = x.reshape(n_batch, -1)
        results = np.array([_single_cholesky_to_dm(xi) for xi in x_flat])
        return results.reshape(*batch_shape, results.shape[-2], results.shape[-1])


def _single_cholesky_to_dm(x: np.ndarray) -> np.ndarray:
    """
    Convert a single Cholesky vector to density matrix.

    1. Reconstruct lower-triangular L from the vector.
    2. rho = L L^dagger / Tr(L L^dagger)
    """
    d = int(np.sqrt(len(x)))
    L = _vector_to_cholesky_matrix(x, d)
    M = L @ L.T.conj()
    rho = M / np.trace(M)
    # Ensure exact Hermiticity
    rho = (rho + rho.T.conj()) / 2.0
    return rho


def _vector_to_cholesky_matrix(x: np.ndarray, d: int) -> np.ndarray:
    """
    Reconstruct lower-triangular Cholesky matrix L from the real vector.

    Inverse of _cholesky_matrix_to_vector (COLUMN-major layout).
    """
    # Diagonal
    diag_real = x[:d]

    # Strictly lower-triangular entries (interleaved real, imag),
    # column-major: for col in range(d): for row in range(col+1, d)
    L = np.zeros((d, d), dtype=np.complex128)
    np.fill_diagonal(L, diag_real)

    idx = d
    for col in range(d):
        for row in range(col + 1, d):
            L[row, col] = x[idx] + 1j * x[idx + 1]
            idx += 2

    return L


def cholesky_dim(n_qubits: int) -> int:
    """Return the dimension of the Cholesky vector for n qubits."""
    d = 2 ** n_qubits
    return d * d


def density_matrix_dim(n_qubits: int) -> int:
    """Return the Hilbert space dimension for n qubits."""
    return 2 ** n_qubits
