"""
Train QST-EDM with realistic noise-enhanced data.

This entry point extends the standard training pipeline with:
  - Realistic readout error model (no IBM account required)
  - Optional IBM Quantum device noise import
  - ERDM-style loss reweighting (focuses on intermediate noise levels)

The noise model makes synthetic training data more representative of real
quantum hardware, improving generalization when deploying on actual devices.

Usage:
    # Train with default realistic noise (no IBM account needed)
    python experiments/train_noisy.py --config configs/n2_edm_noisy.yaml

    # Train with real IBM device noise (requires IBM Quantum account)
    python experiments/train_noisy.py --config configs/n2_edm_noisy.yaml \
        --ibm-backend ibm_brisbane

    # Disable noise model (equivalent to standard training)
    python experiments/train_noisy.py --config configs/n2_edm_noisy.yaml \
        --no-noise
"""

import argparse
import os
import sys
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, save_config
from src.data.dataset import create_dataloaders
from src.data.noise_model import (
    get_realistic_noise_model,
    get_ibm_noise_model,
    apply_readout_error_to_frequencies,
)
from src.models.edm import EDM
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(
        description="Train QST-EDM with realistic noise-enhanced data"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/n2_edm_noisy.yaml",
        help="Path to config YAML",
    )
    parser.add_argument("--n_qubits", type=int, default=None, help="Override n_qubits")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs", help="Output directory")
    parser.add_argument("--data_dir", type=str, default="./data", help="Data cache directory")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu, cuda)")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument(
        "--ibm-backend",
        type=str,
        default=None,
        help="IBM backend name for real device noise (e.g., ibm_brisbane)",
    )
    parser.add_argument(
        "--no-noise",
        action="store_true",
        help="Disable noise model (equivalent to standard training)",
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with command-line args
    if args.n_qubits is not None:
        config["n_qubits"] = args.n_qubits
    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size
    if args.lr is not None:
        config["training"]["learning_rate"] = args.lr

    n_qubits = config["n_qubits"]
    d = 2 ** n_qubits
    cond_input_dim = 6 ** n_qubits
    cholesky_dim = d * d

    print("=" * 60)
    print("QST-EDM with Realistic Noise Model")
    print("=" * 60)
    print(f"  Qubits: {n_qubits} (Hilbert dim: {d})")
    print(f"  Cholesky dim: {cholesky_dim}")
    print(f"  Condition dim: {cond_input_dim}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Setup noise model
    noise_config = config.get("noise_model", {})
    use_noise = noise_config.get("enabled", True) and not args.no_noise
    readout_error = noise_config.get("readout_error", 0.005)

    if use_noise:
        print()
        print("  Noise Model:")
        if args.ibm_backend:
            print(f"    Source: IBM Quantum ({args.ibm_backend})")
            try:
                noise_model = get_ibm_noise_model(args.ibm_backend)
                print("    Status: loaded from real device")
            except Exception as e:
                print(f"    Warning: failed to load IBM backend: {e}")
                print("    Falling back to default realistic noise model")
                noise_model = get_realistic_noise_model(
                    readout_error=readout_error,
                    single_qubit_gate_error=noise_config.get("single_qubit_gate_error", 0.001),
                    two_qubit_gate_error=noise_config.get("two_qubit_gate_error", 0.01),
                )
        else:
            print(f"    Source: synthetic (readout_error={readout_error})")
            noise_model = None  # Will use lightweight readout error in dataset
            print("    Status: using lightweight readout error model")
    else:
        noise_model = None
        print("  Noise Model: DISABLED")

    # Create dataloaders
    print("\nSetting up data loaders...")

    use_variable_shots = config.get("use_variable_shots", True)
    n_shots_min = config.get("n_shots_min", 100)
    n_shots_max = config.get("n_shots_max", 50000)

    if use_variable_shots:
        print(f"  Variable-shot training: {n_shots_min} ~ {n_shots_max} shots")

    dataloaders = create_dataloaders(
        n_qubits=n_qubits,
        n_train=config["data"]["n_train_states"],
        n_val=config["data"]["n_val_states"],
        n_test=config["data"]["n_test_states"],
        n_shots=config["data"]["n_measurement_shots"],
        batch_size=config["training"]["batch_size"],
        state_types=config["data"]["state_types"],
        regularization_eps=config["data"]["regularization_eps"],
        seed=config["data"]["seed"],
        data_dir=args.data_dir,
        use_variable_shots=use_variable_shots,
        n_shots_min=n_shots_min,
        n_shots_max=n_shots_max,
        # Noise model parameters
        use_noise=use_noise,
        readout_error=readout_error,
    )

    train_loader = dataloaders["train_loader"]
    val_loader = dataloaders["val_loader"]

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Build model
    print("\nBuilding model...")
    model = EDM(
        d=d,
        cond_input_dim=cond_input_dim,
        base_channels=config["model"].get("base_channels", 64),
        dim_mults=tuple(config["model"]["dim_mults"]),
        num_res_blocks=config["model"].get("num_res_blocks", 2),
        cond_dim=config["model"]["cond_dim"],
        cond_dropout_prob=config["model"]["cond_dropout_prob"],
        loss_type=config.get("loss_type", "l2"),
        sigma_min=config.get("sigma_min", 0.001),
        sigma_max=config.get("sigma_max", 0.8),
        sigma_data_diag=config.get("sigma_data_diag", 0.3),
        sigma_data_off=config.get("sigma_data_off", 0.2),
        P_mean=config.get("P_mean", -1.2),
        P_std=config.get("P_std", 1.2),
        rho=config.get("rho", 7.0),
        lambda_rank=config.get("lambda_rank", 0.0),
        lambda_rank_warmup=config.get("lambda_rank_warmup", 1000),
        lambda_meas=config.get("lambda_meas", 0.0),
        lambda_meas_warmup=config.get("lambda_meas_warmup", 1000),
        use_preconditioning=config.get("use_preconditioning", True),
        use_loss_reweighting=config.get("use_loss_reweighting", True),
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")
    print(f"  ERDM loss reweighting: {'ON' if model.use_loss_reweighting else 'OFF'}")

    # Create trainer
    output_dir = os.path.join(args.output_dir, f"qst_n{n_qubits}_noisy")
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    # Save config
    os.makedirs(output_dir, exist_ok=True)
    save_config(config, os.path.join(output_dir, "config.yaml"))

    # Resume from checkpoint
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...")
    print("=" * 60)
    trainer.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=config["training"]["epochs"],
    )

    print("\nTraining complete!")
    print(f"Best validation loss: {trainer.best_val_loss:.6f}")
    print(f"Checkpoints saved to: {output_dir}/checkpoints/")


if __name__ == "__main__":
    main()
