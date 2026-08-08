"""Tests for the classical baselines (MLE, linear inversion).

These cover the dimension-inference logic that previously produced
d = 1/2/3 for n = 1/2/3 qubits (instead of 2/4/8), which made the
evaluation script crash in fidelity() and silently discarded most of
the measurement data. Regression tests are added here so the bug
cannot come back unnoticed.
"""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.evaluation.baselines import (
    _infer_hilbert_dim,
    mle_reconstruct,
    linear_inversion,
)


def test_infer_hilbert_dim_correct():
    """6^n outcomes must map to d = 2^n."""
    for n in range(1, 5):
        d = _infer_hilbert_dim(6 ** n)
        assert d == 2 ** n, f"n={n}: got d={d}, want {2**n}"


def test_infer_hilbert_dim_rejects_invalid():
    """Non-6^n lengths must raise (previously produced wrong d silently)."""
    for bad in [10, 100, 37]:
        try:
            _infer_hilbert_dim(bad)
            assert False, f"Expected ValueError for {bad}"
        except ValueError:
            pass


def test_mle_reconstruct_dimension_2qubit():
    """MLE on 2-qubit measurements must return a 4x4 density matrix.

    Regression: the old dimension formula gave d=2, so the returned
    matrix was 2x2 and fidelity() crashed.
    """
    rng = np.random.default_rng(0)
    d = 4
    rho = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = rho @ rho.T.conj()
    rho = rho / np.trace(rho)

    from src.data.measurements import compute_measurement_probabilities, simulate_measurements
    probs = compute_measurement_probabilities(rho)
    freqs = simulate_measurements(rho, n_shots=1000, seed=1)

    rho_mle = mle_reconstruct(freqs, n_shots=1000, max_iter=200)
    assert rho_mle.shape == (4, 4), f"MLE returned wrong shape: {rho_mle.shape}"
    # Reconstructed state must be physical.
    eigvals = np.linalg.eigvalsh(rho_mle)
    assert np.all(eigvals > -1e-6), "MLE result has negative eigenvalues"
    assert np.isclose(np.trace(rho_mle), 1.0, atol=1e-6), "MLE result not trace-1"


def test_linear_inversion_dimension_3qubit():
    """Linear inversion on 3-qubit measurements must return 8x8.

    Regression: the old formula gave d=7 for n=3.
    """
    rng = np.random.default_rng(1)
    d = 8
    rho = rng.normal(size=(d, d)) + 1j * rng.normal(size=(d, d))
    rho = rho @ rho.T.conj()
    rho = rho / np.trace(rho)

    from src.data.measurements import simulate_measurements
    freqs = simulate_measurements(rho, n_shots=10000, seed=2)

    rho_lin = linear_inversion(freqs, n_shots=10000)
    assert rho_lin.shape == (8, 8), f"Linear inversion returned wrong shape: {rho_lin.shape}"


if __name__ == "__main__":
    test_infer_hilbert_dim_correct()
    test_infer_hilbert_dim_rejects_invalid()
    test_mle_reconstruct_dimension_2qubit()
    test_linear_inversion_dimension_3qubit()
    print("All baseline tests passed!")
