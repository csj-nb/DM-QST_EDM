"""
Pre-generate and cache QST datasets.

Usage:
    python experiments/generate_data.py --n_qubits 2 --n_train 50000
    python experiments/generate_data.py --config configs/default.yaml
"""

import argparse
import os
import sys
import time

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.data.dataset import QSTDataset


def main():
    parser = argparse.ArgumentParser(description="Generate QST datasets")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("--n_qubits", type=int, default=None, help="Number of qubits")
    parser.add_argument("--n_train", type=int, default=None, help="Training set size")
    parser.add_argument("--n_val", type=int, default=None, help="Validation set size")
    parser.add_argument("--n_test", type=int, default=None, help="Test set size")
    parser.add_argument("--n_shots", type=int, default=None, help="Measurement shots")
    parser.add_argument("--data_dir", type=str, default="./data", help="Output directory")
    parser.add_argument("--no_cache", action="store_true", help="Don't save cache")
    parser.add_argument(
        "--use-noise", action="store_true",
        help="Apply realistic noise to measurement data",
    )
    parser.add_argument(
        "--noise-model", type=str, default=None, metavar="NAME",
        help="Fake-backend noise model name (e.g. FakeTorino, FakeBrisbane). "
             "Replays a real IBM device's calibration noise without an IBM "
             "account. Implies --use-noise.",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command-line args
    n_qubits = args.n_qubits or config["n_qubits"]
    n_train = args.n_train or config["data"]["n_train_states"]
    n_val = args.n_val or config["data"]["n_val_states"]
    n_test = args.n_test or config["data"]["n_test_states"]
    n_shots = args.n_shots or config["data"]["n_measurement_shots"]
    seed = config["data"]["seed"]
    eps = config["data"]["regularization_eps"]
    state_types = config["data"]["state_types"]
    use_fisher_z = config.get("use_fisher_z", True)
    use_arcsin_sqrt = config.get("use_arcsin_sqrt", False)
    use_counts_channel = config.get("use_counts_channel", False)
    use_shot_channel = config.get("use_shot_channel", False)
    cholesky_normalize = config.get("cholesky_normalize", False)

    # --- Noise model setup (DD-QST style realistic device noise) ---
    use_noise = args.use_noise or args.noise_model is not None
    noise_model = None
    if args.noise_model is not None:
        from src.data.noise_model import get_fake_backend_noise_model

        print(f"  Using fake-backend noise model: {args.noise_model}")
        noise_model = get_fake_backend_noise_model(args.noise_model)
        use_noise = True

    os.makedirs(args.data_dir, exist_ok=True)

    d = 2 ** n_qubits
    cond_dim = 6 ** n_qubits

    print(f"Generating datasets for n_qubits={n_qubits} (d={d}, cond_dim={cond_dim})")
    print(f"  Train: {n_train}, Val: {n_val}, Test: {n_test}")
    print(f"  Shots: {n_shots}")
    print(f"  State types: {state_types}")
    print(f"  Noise: {'none' if noise_model is None else type(noise_model).__name__}")

    sets = {
        "train": (n_train, seed),
        "val": (n_val, seed + 1000),
        "test": (n_test, seed + 2000),
    }

    for name, (size, s) in sets.items():
        cache_path = os.path.join(args.data_dir, f"qst_n{n_qubits}_{name}.pkl") if not args.no_cache else None

        t0 = time.time()
        print(f"\nGenerating {name} set ({size} states)...")

        dataset = QSTDataset(
            n_qubits=n_qubits,
            n_states=size,
            n_shots=n_shots,
            state_types=state_types,
            regularization_eps=eps,
            use_fisher_z=use_fisher_z,
            seed=s,
            cache_path=cache_path,
            use_noise=use_noise,
            noise_model=noise_model,
            use_arcsin_sqrt=use_arcsin_sqrt,
            use_counts_channel=use_counts_channel,
            use_shot_channel=use_shot_channel,
            cholesky_normalize=cholesky_normalize,
        )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s ({elapsed / 60:.1f} min)")

        # Print summary statistics
        import numpy as np
        cholesky_mean = np.mean(dataset.cholesky_vectors)
        cholesky_std = np.std(dataset.cholesky_vectors)
        meas_mean = np.mean(dataset.measurement_condition)
        meas_std = np.std(dataset.measurement_condition)

        print(f"  Cholesky vectors: mean={cholesky_mean:.4f}, std={cholesky_std:.4f}")
        print(f"  Measurement conditions: mean={meas_mean:.4f}, std={meas_std:.4f}")

        # Count state types
        from collections import Counter
        type_counts = Counter(dataset.state_labels)
        for stype, count in sorted(type_counts.items()):
            print(f"    {stype}: {count} ({100 * count / size:.1f}%)")

    print("\nAll datasets generated!")


if __name__ == "__main__":
    main()
