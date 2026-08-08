"""
Pauli basis measurement simulation.

For n qubits, the informationally complete set of Pauli measurements
consists of 3^n measurement bases, where each qubit is measured in
either the X, Y, or Z basis. In each basis, there are 2^n possible
outcomes, giving a total of 6^n outcome probabilities.

The measurement data is the frequency distribution estimated from
a finite number of shots, optionally with Fisher z-transformation
for variance stabilization before feeding into neural networks.
"""

import numpy as np
from itertools import product as cartesian_product
from typing import Optional, Tuple


# Pauli eigenstates: eigenvectors of X, Y, Z
# X eigenstates: |+> = (|0> + |1>)/sqrt(2), |-> = (|0> - |1>)/sqrt(2)
# Y eigenstates: |+i> = (|0> + i|1>)/sqrt(2), |-i> = (|0> - i|1>)/sqrt(2)
# Z eigenstates: |0>, |1>

def _pauli_eigenstates() -> dict:
    """Return the eigenstate projectors for each Pauli operator."""
    # Z basis eigenstates
    z0 = np.array([[1, 0], [0, 0]], dtype=np.complex128)   # |0><0|
    z1 = np.array([[0, 0], [0, 1]], dtype=np.complex128)   # |1><1|

    # X basis eigenstates
    x_plus = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.complex128)      # |+><+|
    x_minus = np.array([[0.5, -0.5], [-0.5, 0.5]], dtype=np.complex128)   # |-><-|

    # Y basis eigenstates
    y_plus = np.array([[0.5, -0.5j], [0.5j, 0.5]], dtype=np.complex128)    # |+i><+i|
    y_minus = np.array([[0.5, 0.5j], [-0.5j, 0.5]], dtype=np.complex128)   # |-i><-i|

    return {
        "Z": {"+1": z0, "-1": z1},
        "X": {"+1": x_plus, "-1": x_minus},
        "Y": {"+1": y_plus, "-1": y_minus},
    }


# Cache the projectors
_PAULI_PROJECTORS = _pauli_eigenstates()

# All Pauli bases
_PAULI_BASES = ["X", "Y", "Z"]


def get_measurement_projector(basis: Tuple[str, ...], outcome: Tuple[int, ...]) -> np.ndarray:
    """
    Build the tensor-product projector for a given measurement basis and outcome.

    Args:
        basis: Tuple of n strings, each in {"X", "Y", "Z"}.
        outcome: Tuple of n ints, each in {+1, -1}.

    Returns:
        Projector matrix of shape (d, d) where d = 2^n.
    """
    n = len(basis)
    # Build the projector for the first qubit
    sign = "+1" if outcome[0] == 1 else "-1"
    P = _PAULI_PROJECTORS[basis[0]][sign]
    for i in range(1, n):
        sign = "+1" if outcome[i] == 1 else "-1"
        P = np.kron(P, _PAULI_PROJECTORS[basis[i]][sign])
    return P


def compute_measurement_probabilities(rho: np.ndarray) -> np.ndarray:
    """
    Compute the full probability distribution over all Pauli measurement
    outcomes.

    For n qubits, returns a vector of length 6^n containing the probability
    of each (basis, outcome) pair, ordered lexicographically by basis then
    by outcome.

    Args:
        rho: Density matrix of shape (d, d) or batch (..., d, d).

    Returns:
        Probability vector of shape (..., 6^n).
    """
    rho = np.asarray(rho)
    d = rho.shape[-1]
    n = int(np.log2(d))

    if rho.ndim == 2:
        return _compute_probs_single(rho, n)
    else:
        batch_shape = rho.shape[:-2]
        rho_flat = rho.reshape(-1, d, d)
        results = np.array([_compute_probs_single(r, n) for r in rho_flat])
        return results.reshape(*batch_shape, 6 ** n)


def _compute_probs_single(rho: np.ndarray, n: int) -> np.ndarray:
    """Compute measurement probabilities for a single density matrix."""
    bases = list(cartesian_product(_PAULI_BASES, repeat=n))
    outcomes = list(cartesian_product([1, -1], repeat=n))

    probs = []
    for basis in bases:
        for outcome in outcomes:
            P = get_measurement_projector(basis, outcome)
            prob = np.real(np.trace(rho @ P))
            # Clamp to [0, 1] to handle numerical issues
            prob = np.clip(prob, 0.0, 1.0)
            probs.append(prob)

    probs = np.array(probs)
    # Ensure normalization (within each basis group)
    for i in range(len(bases)):
        start = i * len(outcomes)
        end = start + len(outcomes)
        total = np.sum(probs[start:end])
        if total > 0:
            probs[start:end] /= total
    return probs


def simulate_measurements(
    rho: np.ndarray,
    n_shots: int,
    seed: Optional[int] = None,
    return_counts: bool = False,
) -> np.ndarray:
    """
    Simulate finite-shot Pauli measurements.

    Shots are distributed evenly across all 3^n measurement bases.
    Within each basis, outcomes are sampled according to Born rule probabilities.

    Args:
        rho: Density matrix of shape (d, d) or batch (..., d, d).
        n_shots: Total number of measurement shots (split across all bases).
        seed: Random seed.
        return_counts: If True, also return the raw per-outcome counts
            (the sufficient statistic of the multinomial measurement model).
            Returns (freqs, counts); counts have the same shape as freqs.

    Returns:
        Frequency vector of shape (..., 6^n). Normalized counts summing to 1
        within each basis group. If return_counts=True, returns a tuple
        (freqs, counts).
    """
    rho = np.asarray(rho)
    d = rho.shape[-1]
    n = int(np.log2(d))

    if rho.ndim == 2:
        return _simulate_single(rho, n, n_shots, seed, return_counts=return_counts)
    else:
        batch_shape = rho.shape[:-2]
        rho_flat = rho.reshape(-1, d, d)
        rng = np.random.default_rng(seed)
        seeds = rng.integers(0, 2 ** 31, size=len(rho_flat))
        results = [
            _simulate_single(r, n, n_shots, int(s), return_counts=return_counts)
            for r, s in zip(rho_flat, seeds)
        ]
        if return_counts:
            freqs = np.array([res[0] for res in results])
            counts = np.array([res[1] for res in results])
            return (
                freqs.reshape(*batch_shape, 6 ** n),
                counts.reshape(*batch_shape, 6 ** n),
            )
        return np.array(results).reshape(*batch_shape, 6 ** n)


def _simulate_single(
    rho: np.ndarray,
    n: int,
    n_shots: int,
    seed: Optional[int] = None,
    return_counts: bool = False,
) -> np.ndarray:
    """Simulate measurements for a single density matrix."""
    rng = np.random.default_rng(seed)
    bases = list(cartesian_product(_PAULI_BASES, repeat=n))
    outcomes = list(cartesian_product([1, -1], repeat=n))
    n_bases = len(bases)  # 3^n
    n_outcomes = len(outcomes)  # 2^n

    # Distribute shots evenly across bases
    shots_per_basis = n_shots // n_bases
    remaining = n_shots % n_bases

    freqs = np.zeros(n_bases * n_outcomes)
    counts_all = np.zeros(n_bases * n_outcomes, dtype=np.float32)

    for i, basis in enumerate(bases):
        # Some bases get one extra shot if n_shots not divisible by n_bases
        n_base_shots = shots_per_basis + (1 if i < remaining else 0)
        if n_base_shots == 0:
            continue

        # Compute true probabilities for this basis
        basis_probs = np.zeros(n_outcomes)
        for j, outcome in enumerate(outcomes):
            P = get_measurement_projector(basis, outcome)
            prob = np.real(np.trace(rho @ P))
            basis_probs[j] = np.clip(prob, 0.0, 1.0)
        basis_probs /= np.sum(basis_probs)

        # Sample outcomes
        counts = rng.multinomial(n_base_shots, basis_probs)
        start_idx = i * n_outcomes
        freqs[start_idx:start_idx + n_outcomes] = counts / n_base_shots
        counts_all[start_idx:start_idx + n_outcomes] = counts

    if return_counts:
        return freqs, counts_all
    return freqs


def fisher_z_transform(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Apply Fisher z-transformation to stabilize variance of probability estimates.

    z = 0.5 * log(p / (1 - p))
    or for bounded support: z = arcsin(sqrt(p))  (variance-stabilizing for binomial)

    We use a modified version: z = log(p + eps) - mean over the dataset

    Args:
        p: Probability/frequency vector.
        eps: Small constant to avoid log(0).

    Returns:
        Transformed vector with more uniform variance.
    """
    p = np.asarray(p)
    # Clip to avoid log(0) and log(negative)
    p_safe = np.clip(p, eps, 1.0 - eps)
    # logit transform: log(p / (1 - p))
    z = np.log(p_safe / (1.0 - p_safe))
    return z


def inverse_fisher_z_transform(z: np.ndarray) -> np.ndarray:
    """Inverse of Fisher z-transformation."""
    return 1.0 / (1.0 + np.exp(-z))


def arcsin_sqrt_transform(p: np.ndarray, eps: float = 0.0) -> np.ndarray:
    """
    Variance-stabilizing arcsine-square-root transform for frequencies.

    z = arcsin(sqrt(p)),  z in [0, pi/2].

    For a binomial proportion p_hat ~ Binomial(N, p)/N, the delta method
    gives Var[arcsin(sqrt(p_hat))] ~ 1/(4N), which is CONSTANT in p — the
    optimal variance stabilization for binomial/multinomial data.

    Crucially, unlike the logit (Fisher z), arcsin-sqrt is well-defined at
    p = 0 and p = 1 (arcsin(0) = 0, arcsin(1) = pi/2), so no clipping
    distortion is introduced at the boundary. This matters in the low-shot
    regime where 0/1 frequencies are common.

    Args:
        p: Frequency/probability vector in [0, 1].
        eps: Unused, kept for API compatibility with fisher_z_transform.

    Returns:
        Transformed vector with approximately constant variance.
    """
    p = np.asarray(p)
    p_safe = np.clip(p, 0.0, 1.0)  # numerical guard only; 0 and 1 are valid
    return np.arcsin(np.sqrt(p_safe))


def inverse_arcsin_sqrt_transform(z: np.ndarray) -> np.ndarray:
    """Inverse of the arcsine-square-root transform: p = sin(z)^2."""
    return np.sin(np.asarray(z)) ** 2
