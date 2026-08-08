"""
Compute sigma_data statistics from generated training data.

Crucial for EDM preconditioning: we need sigma_data for the Cholesky
coordinate layout. Two granularities supported:

  - grouped (default): separate sigma_data for diagonal vs off-diagonal
    elements (they have fundamentally different distributions);
  - per-dim (--per_dim, Fix B): one sigma_data per Cholesky coordinate
    (16 values for 2 qubits). Matches the preconditioner to the rho-loss
    geometry anisotropy found by the Jacobian analysis (J^T J diagonal
    ranges ~9.4x across coordinates).

Vectors are taken from the training cache directly when available
(preferred: the cache holds exactly the normalized vectors the model
sees), otherwise converted on the fly from density matrices.
"""

import numpy as np
import sys
import os
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.representation.cholesky import dm_to_cholesky, cholesky_dim


def load_vectors(data_dir: str, n_qubits: int, max_states: int, normalize: bool):
    """Return (cholesky_vectors, n_states) from cache or on-the-fly gen."""
    d = 2 ** n_qubits
    cache_path = Path(data_dir) / f"qst_n{n_qubits}_train.pkl"
    if cache_path.exists():
        try:
            import pickle
            with open(cache_path, "rb") as f:
                data = pickle.load(f)
            # Prefer the cached Cholesky vectors (exact training representation).
            if "cholesky_vectors" in data:
                vecs = np.asarray(data["cholesky_vectors"][:max_states], dtype=np.float64)
                print(f"Loaded {len(vecs)} cached cholesky_vectors from {cache_path}")
                return vecs, len(vecs)
            states = np.asarray(data["density_matrices"][:max_states])
            print(f"Loaded {len(states)} density matrices from {cache_path}; converting...")
        except Exception as e:
            print(f"  Warning: could not load cache ({e}), generating on the fly...")
            states = None
    else:
        states = None

    if states is None:
        from src.data.states import generate_random_states
        states, _ = generate_random_states(
            n_qubits=n_qubits,
            n_states=max_states,
            state_types={
                "pure_haar": 0.40,
                "mixed_hs": 0.30,
                "mixed_ginibre": 0.10,
                "thermal": 0.10,
                "product": 0.10,
            },
            seed=42,
        )
        print(f"Generated {max_states} random states for n_qubits={n_qubits}...")

    vecs = []
    for i in range(len(states)):
        try:
            vecs.append(dm_to_cholesky(states[i], eps=1e-6, normalize=normalize))
        except Exception as e:
            print(f"  Warning: state {i} failed Cholesky: {e}")
    return np.array(vecs), len(vecs)


def compute_sigma_data(data_dir: str, n_qubits: int = 2, max_states: int = 10000,
                       per_dim: bool = False, normalize: bool = False):
    d = 2 ** n_qubits
    D = d * d

    vecs, n = load_vectors(data_dir, n_qubits, max_states, normalize)
    print(f"Processing {n} states -> {vecs.shape}")

    # Grouped stats
    diag = vecs[:, :d]
    off = vecs[:, d:]
    sigma_diag = float(np.std(diag))
    sigma_off = float(np.std(off))
    sigma_global = float(np.std(vecs))

    print(f"\n  ===== Sigma Data Statistics (n_qubits={n_qubits}, n={n}) =====")
    print(f"  Diagonal ({d}):   std={sigma_diag:.6f}")
    print(f"  Off-diag ({D-d}): std={sigma_off:.6f}")
    print(f"  Global   ({D}):   std={sigma_global:.6f}")

    result = {
        "sigma_data_diag": sigma_diag,
        "sigma_data_off": sigma_off,
        "sigma_data_global": sigma_global,
    }

    if per_dim:
        # Per-coordinate empirical std (Fix B).
        per_dim_std = np.std(vecs, axis=0)  # (D,)
        # Report anisotropy relative to the global std.
        print(f"\n  ===== Per-dim sigma_data (Fix B, {D} coords) =====")
        print(f"  min={per_dim_std.min():.6f}  max={per_dim_std.max():.6f}  "
              f"max/min ratio={per_dim_std.max() / per_dim_std.min():.3f}")
        print("  Values (YAML list):")
        print("  sigma_data_per_dim: [" + ", ".join(f"{v:.6f}" for v in per_dim_std) + "]")
        result["sigma_data_per_dim"] = per_dim_std.tolist()

    print(f"\n  Grouped EDM config:")
    print(f"    sigma_data_diag: {sigma_diag:.4f}")
    print(f"    sigma_data_off:  {sigma_off:.4f}")
    print(f"    sigma_max:       {sigma_global * 5:.4f}  (5x global std)")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_qubits", type=int, default=2)
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--max_states", type=int, default=10000)
    parser.add_argument("--per_dim", action="store_true",
                        help="Also emit per-coordinate sigma_data (Fix B)")
    parser.add_argument("--normalize", action="store_true",
                        help="Normalize Cholesky vectors when converting from density matrices")
    args = parser.parse_args()

    compute_sigma_data(args.data_dir, args.n_qubits, args.max_states,
                       per_dim=args.per_dim, normalize=args.normalize)
