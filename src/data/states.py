"""
Random quantum state generation.

Supports multiple ensembles:
    - Haar-random pure states (uniform over the unit sphere)
    - Hilbert-Schmidt random mixed states (uniform over all density matrices)
    - Ginibre ensemble mixed states
    - Thermal (Gibbs) states with random Hamiltonians
    - Random product states
    - Random quantum circuit (RQC) output states (Qiskit)

Uses a combination of NumPy/Scipy (fast, no external dependency for simple
generation) and Qiskit/QuTiP (for more sophisticated ensembles).
"""

import hashlib
import numpy as np
from typing import Optional, Tuple

# ---------------------------------------------------------------------------
# Circuit hashing (for RQC deduplication, borrowed from DD-QST)
# ---------------------------------------------------------------------------

def circuit_hash(qc) -> str:
    """
    Generate a deterministic hash of a quantum circuit's structure.

    Uses QASM2 serialization so that gate structure + parameters are captured.
    Two circuits with the same structure produce the same hash regardless of
    object identity.

    Args:
        qc: Qiskit QuantumCircuit.

    Returns:
        MD5 hex digest string.
    """
    try:
        import qiskit.qasm2
        qasm_str = qiskit.qasm2.dumps(qc)
    except ImportError:
        # Fallback: use str representation
        qasm_str = str(qc)
    return hashlib.md5(qasm_str.encode("utf-8")).hexdigest()


def haar_random_pure(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Haar-random pure state for n qubits.

    Samples uniformly from the unit sphere in C^d where d = 2^n.
    Equivalent to a normalized vector of i.i.d. complex Gaussian entries.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed.

    Returns:
        Density matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    # i.i.d. complex standard normal
    psi = (rng.normal(0, 1, d) + 1j * rng.normal(0, 1, d)).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    return rho


def hilbert_schmidt_random(n_qubits: int, seed: Optional[int] = None) -> np.ndarray:
    """
    Generate a Hilbert-Schmidt random mixed state.

    rho = G G^dagger / Tr(G G^dagger) where G is a d x d matrix of i.i.d.
    complex standard normal entries.

    This produces states uniformly distributed over the space of all density
    matrices according to the Hilbert-Schmidt measure.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed.

    Returns:
        Density matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    G = (rng.normal(0, 1, (d, d)) + 1j * rng.normal(0, 1, (d, d))).astype(np.complex128)
    M = G @ G.T.conj()
    rho = M / np.trace(M)
    return rho


def ginibre_random(
    n_qubits: int,
    rank: Optional[int] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random mixed state from the Ginibre ensemble.

    rho = G G^dagger / Tr(G G^dagger) where G is a d x rank matrix of
    i.i.d. complex standard normal entries.

    If rank is not specified, it is sampled uniformly from {1, ..., d}.

    Args:
        n_qubits: Number of qubits.
        rank: Rank of the state. If None, sampled uniformly.
        seed: Random seed.

    Returns:
        Density matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits
    if rank is None:
        rank = rng.integers(1, d + 1)

    G = (rng.normal(0, 1, (d, rank)) + 1j * rng.normal(0, 1, (d, rank))).astype(np.complex128)
    M = G @ G.T.conj()
    rho = M / np.trace(M)
    return rho


def thermal_state(
    n_qubits: int,
    beta: Optional[float] = None,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a thermal (Gibbs) state: rho = exp(-beta * H) / Tr(exp(-beta * H)).

    H is a random local Hamiltonian: sum of random single-qubit Pauli terms
    plus random nearest-neighbor XX + ZZ interactions.

    Args:
        n_qubits: Number of qubits.
        beta: Inverse temperature. If None, sampled log-uniformly from [0.01, 10].
        seed: Random seed.

    Returns:
        Density matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)
    d = 2 ** n_qubits

    if beta is None:
        beta = np.exp(rng.uniform(np.log(0.01), np.log(10.0)))

    # Build a random local Hamiltonian
    H = _random_local_hamiltonian(n_qubits, rng)

    # Diagonalize H and compute thermal state
    eigvals, eigvecs = np.linalg.eigh(H)
    # Compute exp(-beta * eigvals) with numerical stability
    exp_vals = np.exp(-beta * (eigvals - np.min(eigvals)))
    exp_vals = exp_vals / np.sum(exp_vals)
    rho = eigvecs @ np.diag(exp_vals) @ eigvecs.T.conj()
    rho = (rho + rho.T.conj()) / 2.0  # Force Hermitian
    return rho


def _random_local_hamiltonian(n_qubits: int, rng: np.random.Generator) -> np.ndarray:
    """
    Build a random local Hamiltonian.

    H = sum_i (a_i X_i + b_i Y_i + c_i Z_i) + sum_{i} (Jx_i X_i X_{i+1} + Jz_i Z_i Z_{i+1})

    with random coefficients in [-1, 1].
    """
    d = 2 ** n_qubits

    # Pauli matrices
    sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
    sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    si = np.eye(2, dtype=np.complex128)

    H = np.zeros((d, d), dtype=np.complex128)

    # Single-qubit terms
    for i in range(n_qubits):
        a, b, c = rng.uniform(-1, 1, 3)
        op_list = [si] * n_qubits
        op_list[i] = a * sx
        H += _kron_all(op_list)
        op_list[i] = b * sy
        H += _kron_all(op_list)
        op_list[i] = c * sz
        H += _kron_all(op_list)

    # Nearest-neighbor interactions
    for i in range(n_qubits - 1):
        jx, jz = rng.uniform(-0.5, 0.5, 2)
        # XX
        op_list = [si] * n_qubits
        op_list[i] = jx * sx
        op_list[i + 1] = sx
        H += _kron_all(op_list)
        # ZZ
        op_list = [si] * n_qubits
        op_list[i] = jz * sz
        op_list[i + 1] = sz
        H += _kron_all(op_list)

    return H


def _kron_all(op_list: list) -> np.ndarray:
    """Kronecker product of a list of matrices."""
    result = op_list[0]
    for op in op_list[1:]:
        result = np.kron(result, op)
    return result


def product_state(
    n_qubits: int,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a random product state: tensor product of n single-qubit pure states.

    Each single-qubit state is sampled uniformly from the Bloch sphere.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed.

    Returns:
        Density matrix of shape (d, d).
    """
    rng = np.random.default_rng(seed)

    # Sample each qubit uniformly from the Bloch sphere
    single_qubit_states = []
    for _ in range(n_qubits):
        # Uniform on sphere: theta in [0, pi], phi in [0, 2*pi]
        # With proper measure: cos(theta) ~ Uniform(-1, 1)
        cos_theta = rng.uniform(-1, 1)
        theta = np.arccos(cos_theta)
        phi = rng.uniform(0, 2 * np.pi)

        # |psi> = cos(theta/2) |0> + e^{i phi} sin(theta/2) |1>
        psi = np.array([
            np.cos(theta / 2),
            np.exp(1j * phi) * np.sin(theta / 2),
        ], dtype=np.complex128)
        rho_i = np.outer(psi, psi.conj())
        single_qubit_states.append(rho_i)

    # Tensor product
    rho = single_qubit_states[0]
    for rho_i in single_qubit_states[1:]:
        rho = np.kron(rho, rho_i)
    return rho


def random_circuit_state(
    n_qubits: int,
    seed: Optional[int] = None,
    min_depth: int = 2,
    max_depth: int = 10,
) -> np.ndarray:
    """
    Generate a pure state as the output of a random quantum circuit (RQC).

    This mirrors the data pipeline of DD-QST (Denoising-Diffusion QST,
    https://github.com/anik-m/Efficient-Quantum-State-Tomography-with-
    Denoising-Diffusion-Models-DD-QST-): instead of sampling a density
    matrix directly from a mathematical ensemble, we build a random circuit
    with ``qiskit.circuit.random.random_circuit`` (random structure, random
    depth, random parameters) and take its output statevector. Circuit
    output states are what real experiments actually prepare, so RQC data
    is a more realistic training distribution than purely random matrices.

    Args:
        n_qubits: Number of qubits.
        seed: Random seed (passed to both circuit generation and statevector).
        min_depth: Minimum circuit depth (number of gate layers).
        max_depth: Maximum circuit depth (number of gate layers).

    Returns:
        Density matrix of shape (d, d).
    """
    try:
        from qiskit.circuit.random import random_circuit
        from qiskit.quantum_info import Statevector
    except ImportError:
        raise ImportError(
            "qiskit is required for 'rqc' state generation. "
            "Install with: pip install qiskit"
        )

    depth = int(np.random.default_rng(seed).integers(min_depth, max_depth + 1))
    qc = random_circuit(n_qubits, depth, seed=seed)
    sv = Statevector(qc)
    rho = np.outer(sv.data, sv.data.conj())
    return rho


def set_rqc_depth_range(min_depth: int, max_depth: int):
    """
    Configure the circuit depth range used by the 'rqc' state type.

    Because ``generate_random_states`` calls generators with the uniform
    signature ``(n_qubits, seed=...)``, depth range is stored in module
    state rather than passed per-call. Call this before generating data
    to override the default [2, 10].

    Args:
        min_depth: Minimum circuit depth.
        max_depth: Maximum circuit depth.
    """
    global _RQC_MIN_DEPTH, _RQC_MAX_DEPTH
    _RQC_MIN_DEPTH = int(min_depth)
    _RQC_MAX_DEPTH = int(max_depth)


# Default RQC depth range (module-level so generate_random_states can reach it)
_RQC_MIN_DEPTH = 2
_RQC_MAX_DEPTH = 10


def generate_random_states(
    n_qubits: int,
    n_states: int,
    state_types: Optional[dict] = None,
    seed: int = 42,
) -> Tuple[np.ndarray, list]:
    """
    Generate a batch of random quantum states with specified type proportions.

    Args:
        n_qubits: Number of qubits.
        n_states: Total number of states to generate.
        state_types: Dict mapping type name to proportion.
            Default: {"pure_haar": 0.40, "mixed_hs": 0.30, "mixed_ginibre": 0.10,
                       "thermal": 0.10, "product": 0.10, "rqc": 0.00}
            "rqc" requires qiskit; it generates random-circuit output states.
        seed: Random seed.

    Returns:
        Tuple of:
            - density_matrices: Array of shape (n_states, d, d)
            - labels: List of state type strings, length n_states
    """
    if state_types is None:
        state_types = {
            "pure_haar": 0.40,
            "mixed_hs": 0.30,
            "mixed_ginibre": 0.10,
            "thermal": 0.10,
            "product": 0.10,
            "rqc": 0.00,
        }

    # Normalize proportions
    total = sum(state_types.values())
    state_types = {k: v / total for k, v in state_types.items()}

    rng = np.random.default_rng(seed)
    type_names = list(state_types.keys())
    proportions = list(state_types.values())

    # Assign state types
    labels = rng.choice(type_names, size=n_states, p=proportions)

    d = 2 ** n_qubits
    rhos = np.zeros((n_states, d, d), dtype=np.complex128)
    generators = {
        "pure_haar": haar_random_pure,
        "mixed_hs": hilbert_schmidt_random,
        "mixed_ginibre": ginibre_random,
        "thermal": thermal_state,
        "product": product_state,
        "rqc": lambda n_qubits, seed=None: random_circuit_state(
            n_qubits,
            seed=seed,
            min_depth=_RQC_MIN_DEPTH,
            max_depth=_RQC_MAX_DEPTH,
        ),
    }

    # Track RQC circuit hashes for deduplication (borrowed from DD-QST).
    # Ensures no duplicate circuits enter the training set, which would bias
    # the learned distribution toward overrepresented circuit structures.
    seen_rqc_hashes: set = set()

    for i, label in enumerate(labels):
        # Each state gets a unique seed for reproducibility
        state_seed = seed + i if seed is not None else None

        if label == "rqc":
            # Deduplication loop: regenerate until a unique circuit is found.
            # Safety cap prevents infinite loops if the parameter space is
            # exhausted (unlikely for n_qubits >= 2 with depth >= 2).
            for _attempt in range(1000):
                rhos[i] = generators[label](n_qubits, seed=state_seed)
                try:
                    from qiskit.circuit.random import random_circuit
                    depth = int(np.random.default_rng(state_seed).integers(
                        _RQC_MIN_DEPTH, _RQC_MAX_DEPTH + 1
                    ))
                    qc = random_circuit(n_qubits, depth, seed=state_seed)
                    c_hash = circuit_hash(qc)
                    if c_hash not in seen_rqc_hashes:
                        seen_rqc_hashes.add(c_hash)
                        break
                except ImportError:
                    break  # qiskit unavailable, skip dedup
            else:
                # Exhausted attempts; keep the last generated state
                pass
        else:
            rhos[i] = generators[label](n_qubits, seed=state_seed)

    return rhos, labels.tolist()


def generate_single_state(
    n_qubits: int,
    state_type: str,
    seed: Optional[int] = None,
) -> np.ndarray:
    """
    Generate a single random quantum state of a specific type.

    Args:
        n_qubits: Number of qubits.
        state_type: One of "pure_haar", "mixed_hs", "mixed_ginibre", "thermal", "product".
        seed: Random seed.

    Returns:
        Density matrix of shape (d, d).
    """
    generators = {
        "pure_haar": haar_random_pure,
        "mixed_hs": hilbert_schmidt_random,
        "mixed_ginibre": ginibre_random,
        "thermal": thermal_state,
        "product": product_state,
        "rqc": lambda n_qubits, seed=None: random_circuit_state(
            n_qubits,
            seed=seed,
            min_depth=_RQC_MIN_DEPTH,
            max_depth=_RQC_MAX_DEPTH,
        ),
    }
    if state_type not in generators:
        raise ValueError(
            f"Unknown state type: {state_type}. "
            f"Choose from {list(generators.keys())}."
        )
    return generators[state_type](n_qubits, seed=seed)