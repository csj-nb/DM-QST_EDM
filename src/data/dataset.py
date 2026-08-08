"""
PyTorch Dataset for Quantum State Tomography.

Combines random state generation, Pauli measurement simulation, and
Cholesky encoding into a trainable PyTorch Dataset.

Training with variable shot counts:
    At each __getitem__ call, the original cached measurement frequencies
    (generated at n_shots_full) are re-noised to simulate a random shot count
    sampled from [n_shots_min, n_shots_max]. This exposes the model to
    varying noise levels during training, improving low-shot generalization.

    Validation and test sets use the full-shot data (n_shots_full) for
    consistent evaluation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .states import generate_random_states
from .measurements import (
    simulate_measurements,
    compute_measurement_probabilities,
    fisher_z_transform,
    inverse_fisher_z_transform,
)
from ..representation.cholesky import dm_to_cholesky


# Module-level helper for multiprocessing (must be picklable)
def _sim_one(args):
    rho, n_shots, seed = args
    return simulate_measurements(rho, n_shots=n_shots, seed=seed)


# Module-level helper returning both frequencies and raw counts (picklable).
# Counts are the sufficient statistic of the multinomial measurement model;
# keeping them lets the model see the shot budget behind each frequency.
def _sim_one_with_counts(args):
    rho, n_shots, seed = args
    return simulate_measurements(
        rho, n_shots=n_shots, seed=seed, return_counts=True
    )


def counts_to_condition_channel(
    counts: np.ndarray,
    n_qubits: int,
    n_shots: int,
) -> np.ndarray:
    """
    Convert raw counts to a model-friendly condition channel.

    c = log1p(counts) / log1p(shots_per_basis)   in [0, 1].

    The log1p keeps the scale bounded and stable for counts from 0 up to
    shots_per_basis while preserving the *absolute* count magnitude (and
    therefore the shot budget / confidence) that pure frequencies discard.
    Frequencies encode p̂ = k/N; counts encode k itself, which is the
    sufficient statistic of the multinomial measurement model.
    """
    counts = np.asarray(counts, dtype=np.float32)
    n_bases = 3 ** n_qubits
    shots_per_basis = max(int(n_shots // n_bases), 1)
    denom = float(np.log1p(shots_per_basis))
    return np.log1p(np.maximum(counts, 0.0)) / denom


class QSTDataset(Dataset):
    """
    PyTorch Dataset for diffusion-model-based quantum state tomography.

    Each item is a tuple:
        (cholesky_vector, measurement_frequencies, density_matrix)
    where:
        - cholesky_vector: torch.Tensor of shape (d*d,), the target in Cholesky space
        - measurement_frequencies: torch.Tensor of shape (6^n,), the condition input
        - density_matrix: torch.Tensor of shape (d, d), the ground truth (for evaluation)

    The dataset can either generate data on-the-fly or load from a cache file.
    """

    def __init__(
        self,
        n_qubits: int,
        n_states: int = 10000,
        n_shots: int = 10000,
        state_types: Optional[Dict[str, float]] = None,
        regularization_eps: float = 1e-6,
        use_fisher_z: bool = True,
        seed: int = 42,
        cache_path: Optional[str] = None,
        # Variable shot training
        is_train: bool = False,
        n_shots_min: int = 100,
        n_shots_max: int = 50000,
        # Realistic noise model
        use_noise: bool = False,
        readout_error: float = 0.005,
        noise_model: Optional[Any] = None,
        # Low-shot representation (scheme 1: arcsin-sqrt + counts channel)
        use_arcsin_sqrt: bool = False,
        use_counts_channel: bool = False,
        use_shot_channel: bool = False,
        # Fix A: remove the redundant global-scaling direction (unit-norm vectors)
        cholesky_normalize: bool = False,
        # Representation: "cholesky" (default) or "hermitian" (direct rho params)
        representation: str = "cholesky",
    ):
        """
        Args:
            n_qubits: Number of qubits.
            n_states: Number of states in the dataset.
            n_shots: Total measurement shots per state.
            state_types: Dict of state type proportions.
            regularization_eps: Epsilon for Cholesky regularization.
            use_fisher_z: Whether to apply Fisher z-transform to measurements.
            seed: Random seed for reproducibility.
            cache_path: Path to cache file. If provided and exists, load from cache.
            is_train: If True, apply variable-shot re-noising at each __getitem__.
            n_shots_min: Minimum shot count for variable-shot training.
            n_shots_max: Maximum shot count for variable-shot training.
        """
        self.n_qubits = n_qubits
        self.n_states = n_states
        self.n_shots = n_shots
        self.regularization_eps = regularization_eps
        self.use_fisher_z = use_fisher_z
        self.seed = seed
        self.state_types = dict(state_types) if state_types is not None else None
        self.d = 2 ** n_qubits
        self.cholesky_dim = self.d * self.d
        self.is_train = is_train
        self.n_shots_min = n_shots_min
        self.n_shots_max = n_shots_max
        self.use_noise = use_noise
        self.readout_error = readout_error
        self.noise_model = noise_model
        self.use_arcsin_sqrt = use_arcsin_sqrt
        self.use_counts_channel = use_counts_channel
        self.use_shot_channel = use_shot_channel
        self.cholesky_normalize = cholesky_normalize
        self.representation = representation
        # Shot-feature normalization reference (log1p scale, [0,1]-ish).
        self.n_shots_max_norm = max(n_shots_max, n_shots, 1)

        # Condition dimension: 6^n per transform channel, doubled when the
        # raw counts channel is appended, +1 when the shot-budget feature
        # (log1p(n_shots)/log1p(n_shots_max)) is appended.
        self.cond_input_dim = (6 ** n_qubits) * (2 if use_counts_channel else 1)
        if use_shot_channel:
            self.cond_input_dim += 1

        # Try to load from cache (validated against current parameters)
        if cache_path and os.path.exists(cache_path):
            loaded = self._try_load_cache(cache_path, state_types)
            if not loaded:
                print(f"    Cache {cache_path} stale or incompatible, regenerating...")
                self._generate_data(state_types)
                self._save_cache(cache_path)
        else:
            self._generate_data(state_types)
            if cache_path:
                self._save_cache(cache_path)

    def _generate_data(self, state_types: Optional[Dict[str, float]]):
        """Generate the full dataset with parallel measurement simulation."""
        rng = np.random.default_rng(self.seed + 1)
        meas_seeds = rng.integers(0, 2 ** 31, size=self.n_states)

        # Generate random density matrices (fast, vectorized internally)
        self.density_matrices, self.state_labels = generate_random_states(
            n_qubits=self.n_qubits,
            n_states=self.n_states,
            state_types=state_types,
            seed=self.seed,
        )

        # Parallel measurement simulation (the bottleneck)
        n_workers = min(mp.cpu_count(), 16)
        n_out = 6 ** self.n_qubits
        self.measurement_freqs = np.zeros(
            (self.n_states, n_out), dtype=np.float32
        )
        if self.use_counts_channel:
            self.measurement_counts = np.zeros(
                (self.n_states, n_out), dtype=np.float32
            )

        # --- Noisy-Aer path: full device noise model (gate + relaxation +
        # readout). Runs serially because each call already batch-simulates
        # all 3^n bases on a single AerSimulator; per-state parallelism would
        # just multiply memory usage of duplicated simulators.
        if self.use_noise and self.noise_model is not None:
            from .noise_model import simulate_measurements_noisy

            print(
                f"    Aer noise simulation with "
                f"{type(self.noise_model).__name__}..."
            )
            for i in range(self.n_states):
                self.measurement_freqs[i] = simulate_measurements_noisy(
                    self.density_matrices[i],
                    self.n_shots,
                    noise_model=self.noise_model,
                    seed=int(meas_seeds[i]),
                )
                if self.use_counts_channel:
                    # Aer backend only returns frequencies; recover counts by
                    # rounding (exact up to the shot-distribution remainder).
                    n_bases = 3 ** self.n_qubits
                    spb = max(self.n_shots // n_bases, 1)
                    self.measurement_counts[i] = np.round(
                        self.measurement_freqs[i] * spb
                    )
                if (i + 1) % 5000 == 0 or i + 1 == self.n_states:
                    print(f"      {i + 1}/{self.n_states}")

        work_items = [
            (self.density_matrices[i], self.n_shots, int(meas_seeds[i]))
            for i in range(self.n_states)
        ]

        if not (self.use_noise and self.noise_model is not None):
            print(
                f"    Parallel measurement sim: {n_workers} workers "
                f"x {self.n_states} states"
            )
            worker = _sim_one_with_counts if self.use_counts_channel else _sim_one
            with ProcessPoolExecutor(
                max_workers=n_workers, mp_context=mp.get_context("spawn")
            ) as executor:
                futures = {
                    executor.submit(worker, w): i
                    for i, w in enumerate(work_items)
                }
                done = 0
                for future in as_completed(futures):
                    i = futures[future]
                    res = future.result()
                    if self.use_counts_channel:
                        self.measurement_freqs[i], self.measurement_counts[i] = res
                    else:
                        self.measurement_freqs[i] = res
                    done += 1
                    if done % 5000 == 0 or done == self.n_states:
                        print(f"      {done}/{self.n_states}")

        # Exact Born-rule probabilities (no shot noise). Used as the ground
        # truth for variable-shot re-sampling during training; using the
        # finite-shot frequencies instead would double-count the shot noise
        # of the cached n_shots measurement.
        self.measurement_probs = compute_measurement_probabilities(
            self.density_matrices
        ).astype(np.float32)

        # Apply realistic readout error if enabled. Skip when a full noise
        # model was used: the Aer simulation already includes readout error.
        if self.use_noise and self.noise_model is None:
            from .noise_model import apply_readout_error_to_frequencies
            print(f"    Applying readout error (p={self.readout_error})...")
            self.measurement_freqs = apply_readout_error_to_frequencies(
                self.measurement_freqs,
                self.n_qubits,
                readout_error=self.readout_error,
            )
            # Recompute exact probabilities from noisy frequencies
            # (for variable-shot re-sampling)
            self.measurement_probs = self.measurement_freqs.copy()

        # Apply variance-stabilizing transform. arcsin-sqrt is preferred for
        # low-shot data (constant variance, well-defined at 0/1, no clipping);
        # Fisher z (logit) is kept for backward compatibility.
        if self.use_arcsin_sqrt:
            from .measurements import arcsin_sqrt_transform

            self.measurement_condition = arcsin_sqrt_transform(
                self.measurement_freqs
            ).astype(np.float32)
        elif self.use_fisher_z:
            self.measurement_condition = fisher_z_transform(
                self.measurement_freqs
            ).astype(np.float32)
        else:
            self.measurement_condition = self.measurement_freqs.astype(np.float32)

        # Append raw counts channel (low-shot scheme 1): the shot budget
        # behind each frequency is a sufficient statistic the pure frequency
        # vector discards. log1p-normalized to a stable [0, 1] scale.
        if self.use_counts_channel:
            counts_cond = counts_to_condition_channel(
                self.measurement_counts, self.n_qubits, self.n_shots
            )
            self.measurement_condition = np.concatenate(
                [self.measurement_condition, counts_cond], axis=1
            ).astype(np.float32)

        # Convert density matrices to Cholesky vectors (vectorized, fast)
        self.cholesky_vectors = dm_to_cholesky(
            self.density_matrices,
            eps=self.regularization_eps,
            normalize=self.cholesky_normalize,
        ).astype(np.float32)

    # Cache format version. Bump whenever the on-disk layout changes
    # (e.g. the Cholesky vector ordering), so stale caches are regenerated
    # instead of silently producing mismatched data.
    CACHE_VERSION = 3

    def _save_cache(self, path: str):
        """Save dataset to cache file (including provenance metadata)."""
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        data = {
            "cache_version": self.CACHE_VERSION,
            "n_qubits": self.n_qubits,
            "n_states": self.n_states,
            "n_shots": self.n_shots,
            "regularization_eps": self.regularization_eps,
            "use_fisher_z": self.use_fisher_z,
            "seed": self.seed,
            "state_types": self.state_types,
            "noise_model": None if self.noise_model is None else str(self.noise_model),
            "use_arcsin_sqrt": self.use_arcsin_sqrt,
            "use_counts_channel": self.use_counts_channel,
            "use_shot_channel": self.use_shot_channel,
            "cholesky_normalize": self.cholesky_normalize,
            "density_matrices": self.density_matrices,
            "state_labels": self.state_labels,
            "measurement_freqs": self.measurement_freqs,
            "measurement_probs": self.measurement_probs,
            "measurement_condition": self.measurement_condition,
            "cholesky_vectors": self.cholesky_vectors,
            "measurement_counts": getattr(self, "measurement_counts", None),
        }
        with open(path, "wb") as f:
            pickle.dump(data, f)

    def _try_load_cache(self, path: str, state_types) -> bool:
        """
        Load dataset from cache if it is compatible with the current
        configuration. Returns True on success, False if the cache is stale,
        has a different version, or was generated with different parameters.
        """
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception:
            return False

        # Validate the cache version and the parameters that affect the data.
        # If any differ from the requested configuration, the cache is stale
        # and must be regenerated (silent reuse was silently corrupting
        # experiments before this check existed).
        checks = {
            "cache_version": data.get("cache_version") == self.CACHE_VERSION,
            "n_qubits": data.get("n_qubits") == self.n_qubits,
            "n_shots": data.get("n_shots") == self.n_shots,
            "regularization_eps": (
                abs(data.get("regularization_eps", -1) - self.regularization_eps) < 1e-12
            ),
            "use_fisher_z": data.get("use_fisher_z") == self.use_fisher_z,
            "seed": data.get("seed") == self.seed,
            "state_types": data.get("state_types") == self.state_types,
            "noise_model": data.get("noise_model") == (
                None if self.noise_model is None else str(self.noise_model)
            ),
            "use_arcsin_sqrt": data.get("use_arcsin_sqrt") == self.use_arcsin_sqrt,
            "use_counts_channel": data.get("use_counts_channel") == self.use_counts_channel,
            "use_shot_channel": data.get("use_shot_channel") == self.use_shot_channel,
            "cholesky_normalize": data.get("cholesky_normalize") == self.cholesky_normalize,
        }
        if not all(checks.values()):
            return False

        self.n_qubits = data["n_qubits"]
        self.n_shots = data["n_shots"]
        self.regularization_eps = data["regularization_eps"]
        self.use_fisher_z = data["use_fisher_z"]
        self.seed = data["seed"]
        self.density_matrices = data["density_matrices"]
        self.state_labels = data["state_labels"]
        self.measurement_freqs = data["measurement_freqs"]
        self.measurement_probs = data.get("measurement_probs")
        self.measurement_condition = data["measurement_condition"]
        self.cholesky_vectors = data["cholesky_vectors"]
        self.measurement_counts = data.get("measurement_counts")
        self.n_states = len(self.density_matrices)

        # If the cache predates the exact Born-probability field, regenerate.
        if self.measurement_probs is None:
            return False
        # If counts channel is requested, the cache must contain counts.
        if self.use_counts_channel and self.measurement_counts is None:
            return False
        return True

    def __len__(self) -> int:
        return self.n_states

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # Get base data from cache
        if self.representation == "hermitian":
            from src.representation.hermitian import dm_to_vec_np
            x_0 = torch.from_numpy(dm_to_vec_np(self.density_matrices[idx]).astype(np.float32))
        elif self.representation == "bloch":
            from src.representation.bloch import dm_to_vec_np as _bloch_dm_to_vec_np
            x_0 = torch.from_numpy(_bloch_dm_to_vec_np(self.density_matrices[idx]).astype(np.float32))
        else:
            x_0 = torch.from_numpy(self.cholesky_vectors[idx])
        dm = self.density_matrices[idx]

        # --- Variable-shot re-noising for training ---
        if self.is_train:
            # Sample random shot count.
            # Log-uniform sampling over [n_shots_min, n_shots_max]: each order
            # of magnitude receives equal training mass, so low-shot regimes
            # (the evaluation focus: 50-1000 shots) are not starved by a
            # uniform draw whose mass sits almost entirely above 5000 shots.
            # Uniform sampling would give shots in [5000, 50000] ~90% of the
            # data and only ~0.5% to shots <= 300.
            log_min = float(np.log(self.n_shots_min))
            log_max = float(np.log(self.n_shots_max))
            n_shots = int(np.exp(np.random.uniform(log_min, log_max)))
            n_shots = min(max(n_shots, self.n_shots_min), self.n_shots_max)

            # Re-sample the measurement at this shot count from the EXACT
            # Born-rule probabilities (cached in measurement_probs). Using
            # the finite-shot cached frequencies as "true probabilities"
            # would double-count the shot noise of the full measurement
            # (variance ~ p(1-p)/n_shots + p(1-p)/N_full), biasing training
            # toward the cached shot budget.
            raw_probs = self.measurement_probs[idx].copy()
            # Renormalize within each basis before re-sampling
            n_bases = 3 ** self.n_qubits
            n_outcomes = 2 ** self.n_qubits
            new_freqs = np.zeros_like(raw_probs)
            new_counts = np.zeros_like(raw_probs, dtype=np.float32)
            for b in range(n_bases):
                start = b * n_outcomes
                end = start + n_outcomes
                probs = raw_probs[start:end]
                probs = np.clip(probs, 0.0, 1.0)
                total = probs.sum()
                if total <= 0:
                    # Degenerate: fall back to uniform within the basis.
                    probs = np.full(n_outcomes, 1.0 / n_outcomes)
                else:
                    probs = probs / total
                    # numpy's RandomState.multinomial validates the pvals with
                    # PYTHON's builtin sum(pvals[:-1]) > 1.0 (sequential, not
                    # pairwise), so even np.sum(...) <= 1.0 can still raise.
                    # Shrink the head by a generous 1e-7 relative margin --
                    # ~9 orders of magnitude above float64 rounding (~1e-16)
                    # but negligible for the sampled distribution -- then
                    # absorb the residual into the last outcome.
                    probs = np.asarray(probs, dtype=np.float64)
                    probs[:-1] *= 0.9999999
                    probs[-1] = max(0.0, 1.0 - probs[:-1].sum())
                shots_per_basis = max(1, n_shots // n_bases)
                counts = np.random.multinomial(shots_per_basis, probs)
                new_freqs[start:end] = counts / shots_per_basis
                new_counts[start:end] = counts

            condition_np = new_freqs.astype(np.float32)
            if self.use_arcsin_sqrt:
                from .measurements import arcsin_sqrt_transform

                condition_np = arcsin_sqrt_transform(condition_np).astype(np.float32)
            elif self.use_fisher_z:
                condition_np = fisher_z_transform(condition_np).astype(np.float32)
            if self.use_counts_channel:
                counts_cond = counts_to_condition_channel(
                    new_counts, self.n_qubits, n_shots
                )
                condition_np = np.concatenate([condition_np, counts_cond], axis=0)
            if self.use_shot_channel:
                # Shot-budget feature: log1p-normalized shot count so the model
                # can tell a reliable 10000-shot measurement from a noisy 50-shot
                # one even though the counts channel is scale-normalized.
                shot_feat = np.log1p(n_shots) / np.log1p(self.n_shots_max_norm)
                condition_np = np.concatenate(
                    [condition_np, np.asarray([shot_feat], dtype=np.float32)], axis=0
                )
        else:
            # Use cached condition (fixed shot count)
            condition_np = self.measurement_condition[idx]
            if self.use_shot_channel:
                shot_feat = np.log1p(self.n_shots) / np.log1p(self.n_shots_max_norm)
                condition_np = np.concatenate(
                    [condition_np, np.asarray([shot_feat], dtype=np.float32)], axis=0
                )

        if self.is_train:
            measurement_target = new_freqs.astype(np.float32)
        else:
            measurement_target = self.measurement_freqs[idx].astype(np.float32)

        return {
            "x_0": x_0,
            "condition": torch.from_numpy(condition_np),
            "measurement_target": torch.from_numpy(measurement_target),
            "density_matrix": torch.from_numpy(
                np.stack([
                    np.real(dm),
                    np.imag(dm),
                ])
            ),
            "label": self.state_labels[idx],
        }

    def get_density_matrix(self, idx: int) -> np.ndarray:
        """Return the density matrix at the given index."""
        return self.density_matrices[idx]


def create_dataloaders(
    n_qubits: int,
    n_train: int = 50000,
    n_val: int = 5000,
    n_test: int = 5000,
    n_shots: int = 10000,
    batch_size: int = 128,
    state_types: Optional[Dict[str, float]] = None,
    regularization_eps: float = 1e-6,
    use_fisher_z: bool = True,
    seed: int = 42,
    data_dir: str = "./data",
    num_workers: int = 0,
    # Variable shot training
    use_variable_shots: bool = False,
    n_shots_min: int = 100,
    n_shots_max: int = 50000,
    # Realistic noise model
    use_noise: bool = False,
    readout_error: float = 0.005,
    noise_model: Optional[Any] = None,
    # Low-shot representation (scheme 1: arcsin-sqrt + counts channel)
    use_arcsin_sqrt: bool = False,
    use_counts_channel: bool = False,
    use_shot_channel: bool = False,
    # Fix A: remove redundant global-scaling direction (unit-norm vectors)
    cholesky_normalize: bool = False,
    # Representation: "cholesky" (default) or "hermitian" (direct rho params)
    representation: str = "cholesky",
) -> Dict[str, Any]:
    """
    Create train, validation, and test dataloaders.

    Args:
        n_qubits: Number of qubits.
        n_train: Number of training states.
        n_val: Number of validation states.
        n_test: Number of test states.
        n_shots: Measurement shots per state (full-shot reference).
        batch_size: Batch size for training.
        state_types: Dict of state type proportions.
        regularization_eps: Cholesky regularization epsilon.
        use_fisher_z: Whether to use Fisher z-transform.
        seed: Base random seed.
        data_dir: Directory for caching datasets.
        num_workers: Number of dataloader workers.
        use_variable_shots: If True, train dataset re-noises each sample
            with a random shot count (from n_shots_min to n_shots_max).
        n_shots_min: Minimum shot count for variable-shot training.
        n_shots_max: Maximum shot count for variable-shot training.

    Returns:
        Dict with "train_loader", "val_loader", "test_loader", and the
        dataset objects.
    """
    os.makedirs(data_dir, exist_ok=True)

    def make_dataset(n_states, split_seed, cache_name, is_train=False):
        cache_path = os.path.join(data_dir, cache_name)
        # Use different cache name when noise is enabled to avoid
        # accidentally reusing ideal-data caches for noisy training.
        if use_noise:
            cache_path = cache_path.replace(".pkl", "_noisy.pkl")
        return QSTDataset(
            n_qubits=n_qubits,
            n_states=n_states,
            n_shots=n_shots,
            state_types=state_types,
            regularization_eps=regularization_eps,
            use_fisher_z=use_fisher_z,
            seed=split_seed,
            cache_path=cache_path,
            is_train=is_train,
            n_shots_min=n_shots_min if is_train else n_shots,
            n_shots_max=n_shots_max,  # 统一 shot 特征归一化基准（val/test 不重采样，此值仅用于 log1p 归一化）
            # Noise model
            use_noise=use_noise,
            readout_error=readout_error,
            noise_model=noise_model,
            # Low-shot representation
            use_arcsin_sqrt=use_arcsin_sqrt,
            use_counts_channel=use_counts_channel,
            use_shot_channel=use_shot_channel,
            cholesky_normalize=cholesky_normalize,
            representation=representation,
        )

    train_dataset = make_dataset(
        n_train, seed, f"qst_n{n_qubits}_train.pkl",
        is_train=use_variable_shots,
    )
    val_dataset = make_dataset(n_val, seed + 1000, f"qst_n{n_qubits}_val.pkl")
    test_dataset = make_dataset(n_test, seed + 2000, f"qst_n{n_qubits}_test.pkl")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )

    return {
        "train_loader": train_loader,
        "val_loader": val_loader,
        "test_loader": test_loader,
        "train_dataset": train_dataset,
        "val_dataset": val_dataset,
        "test_dataset": test_dataset,
    }