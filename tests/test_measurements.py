"""Tests for the measurement simulation module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.measurements import (
    compute_measurement_probabilities,
    simulate_measurements,
    fisher_z_transform,
    inverse_fisher_z_transform,
)
from src.data.states import haar_random_pure


def test_probability_normalization():
    """Test that measurement probabilities sum to 1 within each basis."""
    for n in [1, 2]:
        rho = haar_random_pure(n, seed=42)
        probs = compute_measurement_probabilities(rho)
        d = 2 ** n
        n_bases = 3 ** n
        n_outcomes = 2 ** n

        for i in range(n_bases):
            start = i * n_outcomes
            end = start + n_outcomes
            total = np.sum(probs[start:end])
            assert np.isclose(total, 1.0, atol=1e-10), \
                f"Basis {i} probabilities sum to {total}, not 1.0"


def test_born_rule():
    """Test that measurement probabilities match the Born rule."""
    # For |0> state, Z measurement gives 100% |0>
    rho = np.array([[1, 0], [0, 0]], dtype=np.complex128)
    probs = compute_measurement_probabilities(rho)

    # Z basis: outcomes [|0>, |1>]
    # Layout: bases order X,Y,Z; outcomes order [+1,-1]
    # Z basis +1 (|0>) = probs[4], Z basis -1 (|1>) = probs[5]
    assert np.isclose(probs[4], 1.0, atol=1e-10), f"Expected 1.0, got {probs[4]}"
    assert np.isclose(probs[5], 0.0, atol=1e-10), f"Expected 0.0, got {probs[5]}"


def test_simulated_measurements_shape():
    """Test that simulated measurement output has the right shape."""
    for n in [1, 2]:
        rho = haar_random_pure(n, seed=42)
        freqs = simulate_measurements(rho, n_shots=10000, seed=123)
        assert len(freqs) == 6 ** n, f"Wrong shape for n={n}: {len(freqs)} != {6**n}"


def test_simulated_measurements_normalization():
    """Test that simulated frequencies sum to 1 within each basis."""
    rho = haar_random_pure(2, seed=42)
    freqs = simulate_measurements(rho, n_shots=100000, seed=123)
    n_bases = 9  # 3^2
    n_outcomes = 4  # 2^2

    for i in range(n_bases):
        start = i * n_outcomes
        end = start + n_outcomes
        total = np.sum(freqs[start:end])
        assert np.isclose(total, 1.0, atol=1e-10), \
            f"Basis {i}: sum = {total}"


def test_fisher_z_roundtrip():
    """Test Fisher z-transform round-trip."""
    rho = haar_random_pure(2, seed=42)
    probs = compute_measurement_probabilities(rho)
    z = fisher_z_transform(probs, eps=1e-6)
    probs_recon = inverse_fisher_z_transform(z)

    assert np.allclose(probs, probs_recon, atol=1e-6), \
        "Fisher z-transform round-trip failed"


def test_batch_measurements():
    """Test batch measurement computation."""
    rhos = np.array([haar_random_pure(1, seed=i) for i in range(5)])
    probs = compute_measurement_probabilities(rhos)
    assert probs.shape == (5, 6), f"Wrong shape: {probs.shape}"

    freqs = simulate_measurements(rhos, n_shots=5000, seed=42)
    assert freqs.shape == (5, 6), f"Wrong shape: {freqs.shape}"


if __name__ == "__main__":
    test_probability_normalization()
    test_born_rule()
    test_simulated_measurements_shape()
    test_simulated_measurements_normalization()
    test_fisher_z_roundtrip()
    test_batch_measurements()
    print("All measurement tests passed!")
