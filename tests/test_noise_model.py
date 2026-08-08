"""Tests for the realistic noise model module.

Run: python tests/test_noise_model.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch


def test_lognormal_pdf():
    """Test lognormal PDF computation (used in ERDM reweighting)."""
    from src.models.edm import EDM

    sigma = torch.tensor([0.01, 0.1, 0.3, 0.5, 1.0])
    P_mean, P_std = -1.2, 1.2

    pdf = EDM.lognormal_pdf(sigma, P_mean, P_std)

    # Shape check
    assert pdf.shape == sigma.shape, f"Wrong shape: {pdf.shape}"

    # All positive
    assert torch.all(pdf > 0), "PDF should be positive"

    # No NaN/Inf
    assert not torch.isnan(pdf).any()
    assert not torch.isinf(pdf).any()

    # Peak near exp(P_mean)
    peak_sigma = torch.exp(torch.tensor(float(P_mean)))
    peak_pdf = EDM.lognormal_pdf(peak_sigma.unsqueeze(0), P_mean, P_std)
    far_pdf = EDM.lognormal_pdf(torch.tensor([0.001]), P_mean, P_std)
    assert peak_pdf.item() > far_pdf.item(), "PDF should peak near exp(P_mean)"

    print("[PASS] test_lognormal_pdf")


def test_readout_error_application():
    """Test that readout error changes frequencies."""
    from src.data.noise_model import apply_readout_error_to_frequencies

    # Create a simple frequency vector (1 qubit, 3 bases, 2 outcomes each)
    n_qubits = 1
    n_bases = 3
    n_outcomes = 2
    n_total = n_bases * n_outcomes

    # Perfect frequencies (ideal case)
    freqs = np.array([
        1.0, 0.0,  # Z basis: always |0>
        0.5, 0.5,  # X basis: random
        0.5, 0.5,  # Y basis: random
    ], dtype=np.float32)

    noisy_freqs = apply_readout_error_to_frequencies(
        freqs, n_qubits, readout_error=0.01
    )

    # Shape preserved
    assert noisy_freqs.shape == freqs.shape

    # The perfect Z measurement (1.0, 0.0) should now have some leakage
    assert noisy_freqs[0] < 1.0, "Perfect |0> should have some leakage to |1>"
    assert noisy_freqs[1] > 0.0, "Perfect |0> should have some leakage to |1>"

    # Frequencies should still sum to 1 within each basis
    for b in range(n_bases):
        start = b * n_outcomes
        end = start + n_outcomes
        basis_sum = noisy_freqs[start:end].sum()
        assert abs(basis_sum - 1.0) < 1e-5, f"Basis {b} sums to {basis_sum}"

    print("[PASS] test_readout_error_application")


def test_readout_error_2qubit():
    """Test readout error on 2-qubit system."""
    from src.data.noise_model import apply_readout_error_to_frequencies

    n_qubits = 2
    n_bases = 9  # 3^2
    n_outcomes = 4  # 2^2
    n_total = n_bases * n_outcomes

    # Random normalized frequencies
    rng = np.random.default_rng(42)
    freqs = rng.random(n_total).astype(np.float32)
    # Normalize within each basis
    for b in range(n_bases):
        start = b * n_outcomes
        end = start + n_outcomes
        freqs[start:end] /= freqs[start:end].sum()

    noisy_freqs = apply_readout_error_to_frequencies(
        freqs, n_qubits, readout_error=0.02
    )

    # Shape preserved
    assert noisy_freqs.shape == freqs.shape

    # Should be different from original
    assert not np.allclose(noisy_freqs, freqs), "Noise should change frequencies"

    # Still normalized within each basis
    for b in range(n_bases):
        start = b * n_outcomes
        end = start + n_outcomes
        basis_sum = noisy_freqs[start:end].sum()
        assert abs(basis_sum - 1.0) < 1e-4, f"Basis {b} sums to {basis_sum}"

    print("[PASS] test_readout_error_2qubit")


def test_noise_preserves_statistics():
    """Test that readout error preserves expected statistical properties."""
    from src.data.noise_model import apply_readout_error_to_frequencies

    # Symmetric case: equal probabilities should stay equal
    n_qubits = 1
    freqs = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float32)

    noisy_freqs = apply_readout_error_to_frequencies(
        freqs, n_qubits, readout_error=0.05
    )

    # Symmetric input should give symmetric output
    for b in range(3):
        start = b * 2
        assert abs(noisy_freqs[start] - noisy_freqs[start + 1]) < 1e-5, \
            "Symmetric input should preserve symmetry"

    print("[PASS] test_noise_preserves_statistics")


def test_realistic_noise_model_creation():
    """Test that realistic noise model can be created."""
    try:
        from src.data.noise_model import get_realistic_noise_model
        noise_model = get_realistic_noise_model()
        assert noise_model is not None
        print("[PASS] test_realistic_noise_model_creation")
    except ImportError:
        print("[SKIP] test_realistic_noise_model_creation (qiskit-aer not installed)")


if __name__ == "__main__":
    import torch

    test_lognormal_pdf()
    test_readout_error_application()
    test_readout_error_2qubit()
    test_noise_preserves_statistics()
    test_realistic_noise_model_creation()

    print()
    print("=" * 40)
    print("All noise model tests passed!")
    print("=" * 40)
