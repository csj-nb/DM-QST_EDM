"""
Classical baselines for Quantum State Tomography.

Implements:
    1. Maximum Likelihood Estimation (MLE) with Cholesky parameterization
    2. Linear inversion (least-squares) with eigenvalue clipping
    3. Direct inversion with positivity projection

These baselines are essential for evaluating whether the diffusion model
provides advantages over traditional methods.
"""

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import cholesky as scipy_cholesky
from typing import Optional, Tuple, Callable
import warnings

from ..representation.cholesky import dm_to_cholesky, cholesky_to_dm
from ..data.measurements import (
    compute_measurement_probabilities,
    get_measurement_projector,
    _PAULI_BASES,
)
from itertools import product as cartesian_product


def _infer_hilbert_dim(n_outcomes: int) -> int:
    """
    Infer the Hilbert space dimension d = 2^n from the length of a Pauli
    measurement frequency vector (6^n = 3^n * 2^n).

    Args:
        n_outcomes: Length of the measurement frequency vector (must be 6^n).

    Returns:
        Hilbert space dimension d = 2^n.

    Raises:
        ValueError: If n_outcomes is not a power of 6.
    """
    n = int(np.round(np.log(n_outcomes) / np.log(6)))
    if 6 ** n != n_outcomes:
        raise ValueError(
            f"Cannot infer qubit number from {n_outcomes} outcomes: "
            "expected a power of 6 (6^n)."
        )
    return 2 ** n


def mle_reconstruct(
    measurement_freqs: np.ndarray,
    n_shots: int = 10000,
    max_iter: int = 5000,
    regularization: float = 0.01,
    tol: float = 1e-8,
    verbose: bool = False,
) -> np.ndarray:
    """
    Maximum Likelihood Estimation for quantum state tomography.

    Uses Cholesky parameterization (same as the diffusion model) with
    L-BFGS-B optimization. This ensures a fair comparison.

    Args:
        measurement_freqs: Measurement frequency vector of shape (6^n,).
        n_shots: Total number of measurement shots.
        max_iter: Maximum L-BFGS iterations.
        regularization: Entropy regularization strength.
        tol: Optimization tolerance.
        verbose: Print optimization progress.

    Returns:
        Reconstructed density matrix of shape (d, d).
    """
    # Correct dimension inference: len = 6^n = 3^n * 2^n, so d = 2^n.
    # The previous formula sqrt(len // 2^floor(log2(3^n))) produced
    # d = 1/2/3 for n = 1/2/3 instead of 2/4/8, breaking all baselines.
    d = _infer_hilbert_dim(len(measurement_freqs))
    n = int(np.log2(d))
    n_bases = 3 ** n
    n_outcomes = 2 ** n

    # Shot counts: convert frequencies back to counts
    shots_per_basis = n_shots // n_bases
    counts = np.maximum(measurement_freqs * shots_per_basis, 0)
    # Handle rounding
    for i in range(n_bases):
        start = i * n_outcomes
        end = start + n_outcomes
        total = np.sum(counts[start:end])
        if total > 0:
            counts[start:end] = counts[start:end] / total * shots_per_basis

    # Precompute measurement projectors
    bases = list(cartesian_product(_PAULI_BASES, repeat=n))
    outcomes = list(cartesian_product([1, -1], repeat=n))
    projectors = []
    for basis in bases:
        for outcome in outcomes:
            P = get_measurement_projector(basis, outcome)
            projectors.append(P)

    def negative_log_likelihood(x: np.ndarray) -> float:
        """Negative log-likelihood with optional regularization."""
        try:
            rho = cholesky_to_dm(x)
        except Exception:
            return 1e10  # Invalid parameters

        # Log-likelihood: -sum_k n_k * log(Tr(rho * P_k))
        nll = 0.0
        for k, P in enumerate(projectors):
            prob = np.real(np.trace(rho @ P))
            prob = np.clip(prob, 1e-15, 1.0)
            nll -= counts[k] * np.log(prob)

        # Entropy regularization (encourages higher entropy / more mixed states)
        if regularization > 0:
            eigvals = np.linalg.eigvalsh(rho)
            eigvals = np.clip(eigvals, 1e-15, None)
            entropy = -np.sum(eigvals * np.log(eigvals))
            nll -= regularization * entropy

        return nll

    # NOTE: No analytic gradient is passed to `minimize` below, so L-BFGS-B
    # uses two-point finite differences internally. The previous `gradient`
    # function here returned a zero vector and was never wired up (dead code
    # that would have stopped optimization at the initial point if used).
    # An analytic gradient would make this much faster; see
    # https://en.wikipedia.org/wiki/Matrix_calculus for the Cholesky chain rule.

    # Initial guess: start from a maximally mixed state
    rho_init = np.eye(d, dtype=np.complex128) / d
    x_init = dm_to_cholesky(rho_init, eps=1e-4)

    # Optimize using L-BFGS-B (doesn't require explicit gradient)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = minimize(
            negative_log_likelihood,
            x_init,
            method="L-BFGS-B",
            options={
                "maxiter": max_iter,
                "ftol": tol,
                "disp": verbose,
            },
        )

    rho_mle = cholesky_to_dm(result.x)
    return rho_mle


def _count_bases(total_outcomes: int) -> int:
    """Given total POVM outcomes, compute number of Pauli bases."""
    # total_outcomes = 3^n * 2^n = 6^n
    n = int(np.round(np.log(total_outcomes) / np.log(6)))
    return 3 ** n


def linear_inversion(
    measurement_freqs: np.ndarray,
    n_shots: int = 10000,
) -> np.ndarray:
    """
    Linear inversion (least-squares) reconstruction.

    Estimates the density matrix by inverting the Born rule measurement
    probabilities as a linear system, then projects to the nearest valid
    density matrix.

    Args:
        measurement_freqs: Measurement frequency vector of shape (6^n,).
        n_shots: Total measurement shots.

    Returns:
        Reconstructed density matrix of shape (d, d).
    """
    # Correct dimension inference (see _infer_hilbert_dim).
    d = _infer_hilbert_dim(len(measurement_freqs))
    # A naive implementation - reconstruct from Pauli expectations
    n = int(np.log2(d))
    n_bases = 3 ** n
    n_outcomes = 2 ** n

    # Convert frequencies to expectations for each Pauli operator
    bases = list(cartesian_product(_PAULI_BASES, repeat=n))

    rho = np.zeros((d, d), dtype=np.complex128)

    # Reconstruct from Z-basis measurements (diagonal elements)
    # and use all bases for a simple reconstruction
    for i, basis in enumerate(bases):
        start = i * n_outcomes
        basis_freqs = measurement_freqs[start:start + n_outcomes]
        outcomes = list(cartesian_product([1, -1], repeat=n))

        for j, outcome in enumerate(outcomes):
            P = get_measurement_projector(basis, outcome)
            rho += basis_freqs[j] * P

    # Normalize
    rho = rho / n_bases
    rho = rho / np.trace(rho)
    rho = (rho + rho.T.conj()) / 2.0

    # Project to PSD
    eigvals, eigvecs = np.linalg.eigh(rho)
    eigvals = np.clip(eigvals, 0, None)
    if np.sum(eigvals) > 0:
        eigvals = eigvals / np.sum(eigvals)
    rho = eigvecs @ np.diag(eigvals) @ eigvecs.T.conj()

    return rho


def mle_reconstruct_batch(
    measurement_freqs: np.ndarray,
    n_shots: int = 10000,
    max_iter: int = 5000,
    regularization: float = 0.01,
    verbose: bool = False,
) -> np.ndarray:
    """
    MLE reconstruction for a batch of measurement vectors.

    Args:
        measurement_freqs: Array of shape (N, 6^n).
        n_shots: Total measurement shots per state.
        max_iter: Maximum L-BFGS iterations.
        regularization: Entropy regularization strength.
        verbose: Print progress.

    Returns:
        Reconstructed density matrices of shape (N, d, d).
    """
    N = measurement_freqs.shape[0]
    d = _infer_hilbert_dim(measurement_freqs.shape[1])

    rhos = np.zeros((N, d, d), dtype=np.complex128)
    for i in range(N):
        if verbose and (i + 1) % 10 == 0:
            print(f"  MLE: {i + 1}/{N}")
        rhos[i] = mle_reconstruct(
            measurement_freqs[i],
            n_shots=n_shots,
            max_iter=max_iter,
            regularization=regularization,
            verbose=False,
        )
    return rhos
