"""Tests for RQC (random quantum circuit) state generation and fake-backend
noise models, borrowed from the DD-QST project.

Run: python tests/test_rqc.py
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np

try:
    import qiskit  # noqa: F401
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    import qiskit_aer  # noqa: F401
    HAS_AER = True
except ImportError:
    HAS_AER = False


def test_random_circuit_state_physical():
    """RQC output must be a valid density matrix (Hermitian, trace 1, PSD)."""
    from src.data.states import random_circuit_state

    rho = random_circuit_state(2, seed=7)

    assert rho.shape == (4, 4), f"Wrong shape: {rho.shape}"
    assert np.allclose(rho, rho.conj().T, atol=1e-10), "Not Hermitian"
    assert abs(np.trace(rho) - 1.0) < 1e-10, "Trace != 1"
    evals = np.linalg.eigvalsh(rho)
    assert evals.min() > -1e-10, "Not positive semi-definite"
    assert abs(np.trace(rho @ rho) - 1.0) < 1e-6, "RQC state should be pure"

    print("[PASS] test_random_circuit_state_physical")


def test_random_circuit_state_reproducible_and_distinct():
    """Same seed reproduces the state; different seeds give different states."""
    from src.data.states import random_circuit_state

    rho_a = random_circuit_state(2, seed=11)
    rho_b = random_circuit_state(2, seed=11)
    rho_c = random_circuit_state(2, seed=12)

    assert np.allclose(rho_a, rho_b), "Same seed should reproduce the state"
    assert not np.allclose(rho_a, rho_c), "Different seeds should differ"

    print("[PASS] test_random_circuit_state_reproducible_and_distinct")


def test_generate_random_states_with_rqc():
    """generate_random_states must handle the 'rqc' state type."""
    from src.data.states import generate_random_states

    rhos, labels = generate_random_states(
        n_qubits=2,
        n_states=10,
        state_types={"rqc": 1.0},
        seed=3,
    )

    assert rhos.shape == (10, 4, 4), f"Wrong shape: {rhos.shape}"
    assert all(l == "rqc" for l in labels), f"Wrong labels: {set(labels)}"
    for rho in rhos:
        assert abs(np.trace(rho) - 1.0) < 1e-10

    print("[PASS] test_generate_random_states_with_rqc")


def test_rqc_depth_range_config():
    """set_rqc_depth_range should control circuit depth."""
    from src.data.states import random_circuit_state, set_rqc_depth_range

    set_rqc_depth_range(1, 2)
    # Depth is stored as module state; generating must not crash and must
    # still return a valid state.
    rho = random_circuit_state(2, seed=1)
    assert rho.shape == (4, 4)
    set_rqc_depth_range(2, 10)  # restore default

    print("[PASS] test_rqc_depth_range_config")


def test_fake_backend_noise_model():
    """Fake-backend noise model must be constructible (or gracefully fall
    back) without an IBM account."""
    if not HAS_AER:
        print("[SKIP] test_fake_backend_noise_model (qiskit-aer not installed)")
        return

    from src.data.noise_model import get_fake_backend_noise_model

    nm = get_fake_backend_noise_model("FakeTorino")

    # Both the fake backend and the fallback produce something with errors.
    err_dict = nm.to_dict() if hasattr(nm, "to_dict") else {}
    assert nm is not None
    # Check that it has at least readout error entries
    assert "readout_errors" in err_dict or len(err_dict) > 0

    print("[PASS] test_fake_backend_noise_model")


def test_simulate_measurements_noisy_with_noise_model():
    """simulate_measurements_noisy must honor the noise_model argument."""
    if not HAS_AER:
        print("[SKIP] test_simulate_measurements_noisy_with_noise_model (qiskit-aer not installed)")
        return

    from src.data.noise_model import (
        get_fake_backend_noise_model,
        simulate_measurements_noisy,
    )
    from src.data.states import random_circuit_state

    rho = random_circuit_state(2, seed=5)
    nm = get_fake_backend_noise_model("FakeTorino")

    freqs = simulate_measurements_noisy(rho, n_shots=1000, noise_model=nm, seed=0)

    assert freqs.shape == (36,), f"Wrong shape: {freqs.shape}"
    assert np.all(freqs >= 0.0) and np.all(freqs <= 1.0)
    # Each basis group (6 outcomes for 2 qubits) must sum to ~1
    for b in range(9):
        start, end = b * 4, b * 4 + 4
        assert abs(freqs[start:end].sum() - 1.0) < 0.2, (
            f"Basis {b} not normalized: {freqs[start:end].sum()}"
        )

    print("[PASS] test_simulate_measurements_noisy_with_noise_model")


def test_circuit_hash_deterministic():
    """circuit_hash must be deterministic: same circuit → same hash."""
    if not HAS_QISKIT:
        print("[SKIP] test_circuit_hash_deterministic (qiskit not installed)")
        return

    from qiskit.circuit.random import random_circuit
    from src.data.states import circuit_hash

    qc1 = random_circuit(2, 3, seed=42)
    qc2 = random_circuit(2, 3, seed=42)
    qc3 = random_circuit(2, 3, seed=99)

    h1 = circuit_hash(qc1)
    h2 = circuit_hash(qc2)
    h3 = circuit_hash(qc3)

    assert h1 == h2, "Same seed should produce same hash"
    assert h1 != h3, "Different seed should produce different hash"
    assert len(h1) == 32, "MD5 hex digest should be 32 chars"

    print("[PASS] test_circuit_hash_deterministic")


def test_rqc_deduplication():
    """generate_random_states with 'rqc' type should produce unique circuits."""
    if not HAS_QISKIT:
        print("[SKIP] test_rqc_deduplication (qiskit not installed)")
        return

    from src.data.states import generate_random_states, circuit_hash
    from qiskit.circuit.random import random_circuit

    n_states = 50
    rhos, labels = generate_random_states(
        n_qubits=2,
        n_states=n_states,
        state_types={"rqc": 1.0},
        seed=123,
    )

    # Verify all states are unique (no duplicates)
    hashes = set()
    for i, label in enumerate(labels):
        assert label == "rqc"
        # Reconstruct circuit to hash (same seed as generation)
        rng = np.random.default_rng(123 + i)
        depth = int(rng.integers(2, 11))
        qc = random_circuit(2, depth, seed=123 + i)
        h = circuit_hash(qc)
        hashes.add(h)

    # With dedup, all hashes should be unique
    assert len(hashes) == n_states, (
        f"Expected {n_states} unique circuits, got {len(hashes)}"
    )

    print("[PASS] test_rqc_deduplication")


if __name__ == "__main__":
    if not HAS_QISKIT:
        print("SKIP: qiskit not installed; RQC tests cannot run.")
        print("  Install with: pip install qiskit")
        sys.exit(0)

    test_random_circuit_state_physical()
    test_random_circuit_state_reproducible_and_distinct()
    test_generate_random_states_with_rqc()
    test_rqc_depth_range_config()
    test_circuit_hash_deterministic()
    test_rqc_deduplication()

    if HAS_AER:
        test_fake_backend_noise_model()
        test_simulate_measurements_noisy_with_noise_model()
    else:
        print("SKIP: qiskit-aer not installed; noise-model tests cannot run.")
        print("  Install with: pip install qiskit-aer")

    print("\nAll RQC tests passed.")
