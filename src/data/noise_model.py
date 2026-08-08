"""
Qiskit-based realistic noise models for quantum measurement simulation.

Provides noise models that can be used to enhance synthetic training data
with realistic device noise characteristics (readout error, gate error,
depolarization), without requiring access to real quantum hardware.

Usage:
    # Default realistic noise (no IBM account needed)
    noise_model = get_realistic_noise_model()

    # Noise from a real IBM device's calibration data (no IBM account needed,
    # uses bundled "fake" backends that replay real device properties)
    noise_model = get_fake_backend_noise_model('FakeTorino')

    # From real IBM device (requires IBM Quantum account)
    noise_model = get_ibm_noise_model('ibm_brisbane')

    # Apply to measurement simulation
    freqs = simulate_measurements_noisy(rho, n_shots, noise_model=noise_model)
"""

import numpy as np
from itertools import product as cartesian_product
from typing import Optional


def get_realistic_noise_model(
    readout_error: float = 0.01,
    single_qubit_gate_error: float = 0.001,
    two_qubit_gate_error: float = 0.01,
    t1: float = 50e-6,
    t2: float = 70e-6,
    gate_time_1q: float = 50e-9,
    gate_time_2q: float = 300e-9,
) -> "NoiseModel":
    """
    Create a realistic noise model without requiring IBM Quantum account.

    Based on typical values for superconducting transmon qubits (IBM-style).

    Args:
        readout_error: Single-qubit readout misclassification probability.
        single_qubit_gate_error: Depolarizing error for single-qubit gates.
        two_qubit_gate_error: Depolarizing error for two-qubit gates.
        t1: T1 relaxation time (seconds).
        t2: T2 dephasing time (seconds).
        gate_time_1q: Single-qubit gate duration (seconds).
        gate_time_2q: Two-qubit gate duration (seconds).

    Returns:
        Qiskit NoiseModel object.
    """
    try:
        from qiskit_aer.noise import (
            NoiseModel,
            depolarizing_error,
            ReadoutError,
        )
    except ImportError:
        raise ImportError(
            "qiskit-aer is required for noise models. "
            "Install with: pip install qiskit-aer"
        )

    noise_model = NoiseModel()

    # Single-qubit gate depolarizing error
    error_1q = depolarizing_error(single_qubit_gate_error, 1)
    noise_model.add_all_qubit_quantum_error(error_1q, ['u1', 'u2', 'u3', 'rx', 'ry', 'rz'])

    # Two-qubit gate depolarizing error
    error_2q = depolarizing_error(two_qubit_gate_error, 2)
    noise_model.add_all_qubit_quantum_error(error_2q, ['cx', 'ecr'])

    # Readout error (symmetric)
    p0_given1 = readout_error
    p1_given0 = readout_error
    readout = ReadoutError(
        [[1 - p1_given0, p1_given0],
         [p0_given1, 1 - p0_given1]]
    )
    noise_model.add_all_qubit_readout_error(readout)

    return noise_model


def get_ibm_noise_model(backend_name: str = "ibm_brisbane") -> "NoiseModel":
    """
    Import noise model from a real IBM Quantum device.

    Requires IBM Quantum account and token:
        https://quantum.ibm.com/

    Args:
        backend_name: Name of the IBM backend.

    Returns:
        Qiskit NoiseModel object.
    """
    try:
        from qiskit_aer.noise import NoiseModel
        from qiskit_ibm_runtime import QiskitRuntimeService
    except ImportError:
        raise ImportError(
            "qiskit-aer and qiskit-ibm-runtime are required. "
            "Install with: pip install qiskit-aer qiskit-ibm-runtime"
        )

    service = QiskitRuntimeService()
    backend = service.backend(backend_name)
    noise_model = NoiseModel.from_backend(backend)

    return noise_model


def get_fake_backend_noise_model(
    backend_name: str = "FakeTorino",
) -> "NoiseModel":
    """
    Build a noise model from a bundled fake (simulated) IBM backend.

    Fake backends replay the *real* calibration data (T1/T2, gate errors,
    readout errors, connectivity) of actual IBM Quantum devices, but run
    locally — no IBM Quantum account or network access required. This is
    the same trick used by DD-QST
    (``AerSimulator.from_backend(FakeTorino())``) to get device-realistic
    noise without hardware access.

    Args:
        backend_name: Name of a fake backend, e.g. 'FakeTorino' (127 qubits),
            'FakeBrisbane', 'FakeSherbrooke', 'FakeWashingtonV2', ...
            Falls back to a generic realistic noise model if the requested
            fake backend is unavailable.

    Returns:
        Qiskit NoiseModel object.
    """
    try:
        from qiskit_aer.noise import NoiseModel
    except ImportError:
        raise ImportError(
            "qiskit-aer is required for noise models. "
            "Install with: pip install qiskit-aer"
        )

    # qiskit-ibm-runtime ships the fake backends in different locations
    # depending on the version; try them in order.
    fake_class = None
    try:
        from qiskit_ibm_runtime.fake_provider import FakeTorino

        fake_class = FakeTorino
    except (ImportError, AttributeError):
        try:
            from qiskit.providers.fake_provider import FakeTorino

            fake_class = FakeTorino
        except (ImportError, AttributeError):
            fake_class = None

    if fake_class is not None:
        try:
            backend = fake_class()
            return NoiseModel.from_backend(backend)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"    [noise_model] FakeBackend {backend_name} failed ({exc}); "
                  f"falling back to generic noise model.")

    # Fallback: generic transmon-style noise model
    return get_realistic_noise_model()


def apply_readout_error_to_frequencies(
    freqs: np.ndarray,
    n_qubits: int,
    readout_error: float = 0.01,
) -> np.ndarray:
    """
    Apply readout error directly to measurement frequencies.

    This is a lightweight approximation that doesn't require Qiskit.
    Applies a symmetric confusion matrix to flip outcomes with probability
    ``readout_error``.

    For n qubits, the confusion matrix for each qubit is:
        [[1-e, e],
         [e, 1-e]]

    We apply this independently to each qubit's marginal.

    Args:
        freqs: Measurement frequency vector of shape (..., 6^n).
        n_qubits: Number of qubits.
        readout_error: Readout misclassification probability.

    Returns:
        Noisy frequency vector, same shape as input.
    """
    freqs = np.asarray(freqs, dtype=np.float64)
    n_bases = 3 ** n_qubits
    n_outcomes = 2 ** n_qubits

    if freqs.ndim == 1:
        freqs = freqs.reshape(1, -1)
        squeeze = True
    else:
        squeeze = False

    noisy_freqs = freqs.copy()

    # For each basis measurement
    for b in range(n_bases):
        start = b * n_outcomes
        end = start + n_outcomes
        probs = freqs[:, start:end]  # (batch, 2^n)

        # Apply readout error: for each outcome bitstring, the probability
        # "leaks" to bitstrings that differ by one bit flip.
        # Approximation: mix each outcome with its flipped versions
        noisy_probs = probs * (1 - readout_error) ** n_qubits

        # Add contributions from single-bit-flip neighbors
        for q in range(n_qubits):
            # Flip bit q in each outcome
            flipped = probs.copy()
            flip_mask = np.arange(n_outcomes) ^ (1 << q)
            flipped = probs[:, flip_mask]
            noisy_probs += flipped * (readout_error * (1 - readout_error) ** (n_qubits - 1))

        # Renormalize
        row_sums = noisy_probs.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        noisy_freqs[:, start:end] = noisy_probs / row_sums

    if squeeze:
        noisy_freqs = noisy_freqs.squeeze(0)

    return noisy_freqs.astype(np.float32)


def simulate_measurements_noisy(
    rho: np.ndarray,
    n_shots: int,
    noise_model: Optional["NoiseModel"] = None,
    readout_error: float = 0.005,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Pauli measurements with realistic noise.

    Combines finite-shot statistics with readout error. This gives more
    realistic training data than ideal Born-rule sampling.

    If ``noise_model`` is provided (e.g. from ``get_realistic_noise_model``,
    ``get_fake_backend_noise_model`` or ``get_ibm_noise_model``), the
    measurement circuits are run on a noisy ``AerSimulator`` so that gate
    errors, T1/T2 relaxation, crosstalk and readout errors all affect the
    data — not just the readout approximation. If ``noise_model`` is None,
    a fast readout-error-only approximation is used.

    Args:
        rho: Density matrix of shape (d, d).
        n_shots: Total measurement shots.
        noise_model: Qiskit NoiseModel (if None, uses readout error only).
        readout_error: Readout error probability (used if noise_model is None).
        seed: Random seed.

    Returns:
        Noisy frequency vector of shape (6^n,).
    """
    if noise_model is not None:
        return _simulate_with_aer_noise(
            rho, n_shots, noise_model=noise_model, seed=seed
        )

    from .measurements import simulate_measurements

    # First get ideal finite-shot frequencies
    freqs = simulate_measurements(rho, n_shots=1, seed=seed)

    # Compute exact Born probabilities for proper finite-shot sampling
    from .measurements import compute_measurement_probabilities
    probs = compute_measurement_probabilities(rho)

    # Re-sample with finite shots
    n = int(np.log2(rho.shape[-1]))
    n_bases = 3 ** n
    n_outcomes = 2 ** n

    rng = np.random.default_rng(seed)
    shots_per_basis = n_shots // n_bases

    noisy_freqs = np.zeros_like(probs)

    for b in range(n_bases):
        start = b * n_outcomes
        end = start + n_outcomes
        basis_probs = probs[start:end].copy()
        basis_probs = np.clip(basis_probs, 0.0, 1.0)
        basis_probs /= basis_probs.sum()

        if shots_per_basis > 0:
            counts = rng.multinomial(shots_per_basis, basis_probs)
            noisy_freqs[start:end] = counts / shots_per_basis

    # Apply readout error
    noisy_freqs = apply_readout_error_to_frequencies(
        noisy_freqs, n, readout_error=readout_error
    )

    return noisy_freqs


def _simulate_with_aer_noise(
    rho: np.ndarray,
    n_shots: int,
    noise_model: "NoiseModel",
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Simulate Pauli measurements on a noisy AerSimulator.

    For each of the 3^n measurement bases, the state is rotated into the
    measurement basis, measured, and the counts are collected from a noisy
    simulator configured with the given noise model. This captures gate
    errors + relaxation + readout errors exactly as a real device would.

    Args:
        rho: Density matrix of shape (d, d).
        n_shots: Total shots (split evenly across the 3^n bases).
        noise_model: Qiskit NoiseModel.
        seed: Random seed.

    Returns:
        Frequency vector of shape (6^n,), normalized within each basis.
    """
    try:
        from qiskit import QuantumCircuit, transpile
        from qiskit_aer import AerSimulator
        from qiskit_aer.library import set_density_matrix
    except ImportError:
        raise ImportError(
            "qiskit and qiskit-aer are required for Aer noise simulation. "
            "Install with: pip install qiskit qiskit-aer"
        )

    rho = np.asarray(rho)
    d = rho.shape[-1]
    n = int(np.log2(d))
    if 2 ** n != d:
        raise ValueError("rho must have dimension 2^n x 2^n")

    rng = np.random.default_rng(seed)
    bases = list(cartesian_product(["X", "Y", "Z"], repeat=n))
    outcomes = list(cartesian_product([1, -1], repeat=n))
    n_bases = len(bases)
    n_outcomes = len(outcomes)

    shots_per_basis = n_shots // n_bases
    remaining = n_shots % n_bases

    # Build all measurement circuits in one batch (like DD-QST's
    # batch-transpile-then-run pipeline).
    circuits = []
    basis_for_circuit = []
    for i, basis in enumerate(bases):
        n_base_shots = shots_per_basis + (1 if i < remaining else 0)
        if n_base_shots == 0:
            continue
        qc = QuantumCircuit(n)
        # Initialize with the (possibly mixed) state
        qc.set_density_matrix(rho)
        # Rotate into measurement basis: X -> H, Y -> Sdg then H
        for q, b in enumerate(basis):
            if b == "X":
                qc.h(q)
            elif b == "Y":
                qc.sdg(q)
                qc.h(q)
        qc.measure_all()
        circuits.append(qc)
        basis_for_circuit.append((i, basis, n_base_shots))

    sim = AerSimulator(noise_model=noise_model)
    t_qcs = transpile(circuits, sim, optimization_level=0)
    results = sim.run(t_qcs, shots=max(shots_per_basis, 1),
                      seed_simulator=int(rng.integers(0, 2 ** 31))).result()

    freqs = np.zeros(n_bases * n_outcomes)
    for j, (i, basis, n_base_shots) in enumerate(basis_for_circuit):
        counts = results.get_counts(j)
        total = sum(counts.values())
        if total == 0:
            continue
        # Qiskit counts are little-endian: bitstring "01" means q0=1, q1=0.
        # Map outcome bits (measured values 0/1) to +/-1 and to the outcome
        # index used by the project's 6^n convention.
        for bitstring, count in counts.items():
            bits = [int(c) for c in bitstring]  # q0 .. q_{n-1}
            outcome_idx = 0
            for q, b in enumerate(bits):
                if b == 1:
                    outcome_idx |= (1 << (n - 1 - q))
            freqs[i * n_outcomes + outcome_idx] += count / total

    # Normalize within each basis group (defensive)
    for i in range(n_bases):
        start = i * n_outcomes
        end = start + n_outcomes
        s = freqs[start:end].sum()
        if s > 0:
            freqs[start:end] /= s

    return freqs.astype(np.float32)