"""Tests for the low-shot representation (scheme 1): arcsin-sqrt transform
+ raw counts channel.

Run: python tests/test_lowshot.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np


def test_arcsin_sqrt_boundary():
    """arcsin-sqrt must be finite and invertible at 0 and 1 (no clipping)."""
    from src.data.measurements import (
        arcsin_sqrt_transform,
        inverse_arcsin_sqrt_transform,
    )

    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    z = arcsin_sqrt_transform(p)
    assert np.all(np.isfinite(z)), "must be finite at 0 and 1"
    assert np.allclose(z, [0.0, np.pi / 6, np.pi / 4, np.pi / 3, np.pi / 2])
    assert np.allclose(inverse_arcsin_sqrt_transform(z), p), "round-trip failed"

    print("[PASS] test_arcsin_sqrt_boundary")


def test_arcsin_sqrt_variance_stabilization():
    """Empirical variance of arcsin(sqrt(p_hat)) should be ~constant in p."""
    from src.data.measurements import arcsin_sqrt_transform

    rng = np.random.default_rng(0)
    N = 200
    vars_ = []
    for p_true in [0.05, 0.3, 0.5, 0.9]:
        samples = rng.binomial(N, p_true, size=20000) / N
        vars_.append(np.var(arcsin_sqrt_transform(samples)))
    # Theoretical value: 1/(4N) = 0.00125; allow slack for empirical noise
    assert max(vars_) - min(vars_) < 0.0015, f"variance not constant: {vars_}"

    print("[PASS] test_arcsin_sqrt_variance_stabilization")


def test_simulate_measurements_with_counts():
    """simulate_measurements(return_counts=True) must return (freqs, counts)
    with freqs unchanged and counts recoverable per basis."""
    from src.data.measurements import simulate_measurements
    from src.data.states import haar_random_pure

    rho = haar_random_pure(2, seed=42)
    freqs, counts = simulate_measurements(rho, n_shots=1000, seed=7, return_counts=True)
    freqs_only = simulate_measurements(rho, n_shots=1000, seed=7)

    assert np.allclose(freqs, freqs_only), "return_counts changed freqs"
    assert counts.shape == (36,) and counts.dtype == np.float32

    n_bases, n_out, N = 9, 4, 1000
    spb, rem = N // n_bases, N % n_bases
    for b in range(n_bases):
        n_bs = spb + (1 if b < rem else 0)
        s, e = b * n_out, b * n_out + 4
        assert np.allclose(freqs[s:e], counts[s:e] / n_bs), f"basis {b}"

    print("[PASS] test_simulate_measurements_with_counts")


def test_dataset_counts_channel():
    """QSTDataset with use_arcsin_sqrt + use_counts_channel must produce a
    2*6^n-dim condition: arcsin-sqrt frequencies then log1p-normalized counts."""
    from src.data.dataset import QSTDataset

    ds = QSTDataset(
        n_qubits=2,
        n_states=30,
        n_shots=1000,
        state_types={"mixed_hs": 1.0},
        use_arcsin_sqrt=True,
        use_counts_channel=True,
        seed=42,
    )

    cond = ds.measurement_condition
    assert cond.shape == (30, 72), f"cond shape: {cond.shape}"
    assert ds.cond_input_dim == 72
    assert ds.measurement_counts.shape == (30, 36)

    first, second = cond[:, :36], cond[:, 36:]
    assert first.min() >= 0 and first.max() <= np.pi / 2 + 1e-6
    assert second.min() >= 0 and second.max() <= 1.0 + 1e-6

    item = ds.__getitem__(0)
    assert item["condition"].shape == (72,), "getitem cond must be 72-dim"

    print("[PASS] test_dataset_counts_channel")


def test_dataset_variable_shot_counts_channel():
    """Variable-shot re-sampling must also append the counts channel."""
    from src.data.dataset import QSTDataset

    ds = QSTDataset(
        n_qubits=2,
        n_states=20,
        n_shots=1000,
        state_types={"mixed_hs": 1.0},
        use_arcsin_sqrt=True,
        use_counts_channel=True,
        seed=1,
        is_train=True,
        n_shots_min=100,
        n_shots_max=5000,
    )
    for i in range(5):
        item = ds.__getitem__(i)
        assert item["condition"].shape == (72,), item["condition"].shape

    print("[PASS] test_dataset_variable_shot_counts_channel")


def test_dataset_backward_compat():
    """Default settings (Fisher z, no counts channel) keep the 6^n-dim cond."""
    from src.data.dataset import QSTDataset

    ds = QSTDataset(
        n_qubits=2,
        n_states=10,
        n_shots=1000,
        state_types={"mixed_hs": 1.0},
        seed=7,
    )
    assert ds.measurement_condition.shape == (10, 36)
    assert ds.__getitem__(0)["condition"].shape == (36,)

    print("[PASS] test_dataset_backward_compat")


if __name__ == "__main__":
    test_arcsin_sqrt_boundary()
    test_arcsin_sqrt_variance_stabilization()
    test_simulate_measurements_with_counts()
    test_dataset_counts_channel()
    test_dataset_variable_shot_counts_channel()
    test_dataset_backward_compat()
    print("\nAll low-shot representation tests passed.")
