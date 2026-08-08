"""
Ablation experiments for DM-QST.

Systematically test the impact of key design choices:
    1. Conditioning dropout probability (classifier-free guidance strength)
    2. Diffusion timesteps vs DDIM steps trade-off
    3. Noise schedule type (linear vs cosine)
    4. Model size (channel multipliers)
    5. Conditioning architecture (FiLM vs simple concatenation)

Usage:
    python experiments/ablation.py --ablation cond_dropout --n_qubits 1
    python experiments/ablation.py --ablation ddim_steps --n_qubits 2
    python experiments/ablation.py --ablation all --n_qubits 1
"""

import argparse
import os
import sys
import json
import time
from itertools import product

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.data.dataset import QSTDataset
from src.models.diffusion import DDPM
from src.representation.cholesky import cholesky_to_dm
from src.evaluation.metrics import fidelity, trace_distance


def run_single_experiment(
    n_qubits: int,
    n_train: int,
    n_val: int,
    epochs: int,
    batch_size: int,
    lr: float,
    d: int,
    cond_dim: int,
    timesteps: int,
    cond_dropout_prob: float,
    beta_schedule: str,
    dim_mults: tuple,
    base_channels: int,
    device: torch.device,
    seed: int = 42,
) -> dict:
    """Run a single ablation experiment and return results."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Data
    train_dataset = QSTDataset(
        n_qubits=n_qubits, n_states=n_train, n_shots=10000,
        regularization_eps=1e-6, seed=seed,
    )
    val_dataset = QSTDataset(
        n_qubits=n_qubits, n_states=n_val, n_shots=10000,
        regularization_eps=1e-6, seed=seed + 1000,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
    )

    # Model
    model = DDPM(
        d=d, timesteps=timesteps, cond_input_dim=cond_dim,
        base_channels=base_channels, dim_mults=dim_mults,
        cond_dim=64, cond_dropout_prob=cond_dropout_prob,
        beta_schedule=beta_schedule,
    ).to(device)

    # Train
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in train_loader:
            x_0 = batch["x_0"].to(device)
            condition = batch["condition"].to(device)
            loss = model.training_loss(x_0, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        train_losses.append(epoch_loss / len(train_loader))

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch["x_0"].to(device)
                condition = batch["condition"].to(device)
                val_loss += model.training_loss(x_0, condition).item()
        val_losses.append(val_loss / len(val_loader))
        model.train()

    # Evaluate on 20 test states
    model.eval()
    fids = []
    with torch.no_grad():
        for i in range(min(20, n_val)):
            condition = val_dataset[i]["condition"].unsqueeze(0).to(device)
            x_pred = model.ddim_sample(condition, ddim_steps=50, progress=False)
            x_pred_np = x_pred.cpu().numpy()[0]
            rho_pred = cholesky_to_dm(x_pred_np)
            rho_true = val_dataset.density_matrices[i]
            fids.append(fidelity(rho_true, rho_pred))

    return {
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "mean_fidelity": float(np.mean(fids)),
        "std_fidelity": float(np.std(fids)),
        "train_losses": train_losses,
        "val_losses": val_losses,
        "n_params": sum(p.numel() for p in model.parameters()),
    }


def ablation_cond_dropout(args, device):
    """Ablation: conditioning dropout probability."""
    print("\n" + "=" * 60)
    print("ABLATION: Conditioning Dropout Probability")
    print("=" * 60)

    config = load_config(args.config)
    n_qubits = args.n_qubits or config["n_qubits"]
    d = 2 ** n_qubits
    cond_dim = 6 ** n_qubits

    dropout_probs = [0.0, 0.05, 0.1, 0.2, 0.3]
    results = {}

    for p in dropout_probs:
        print(f"\n  cond_dropout_prob = {p}")
        t0 = time.time()
        res = run_single_experiment(
            n_qubits=n_qubits, n_train=args.n_train, n_val=args.n_val,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            d=d, cond_dim=cond_dim, timesteps=args.timesteps,
            cond_dropout_prob=p, beta_schedule="cosine",
            dim_mults=(1, 2), base_channels=32, device=device,
        )
        res["time"] = time.time() - t0
        results[p] = res
        print(f"    Fidelity: {res['mean_fidelity']:.4f} +/- {res['std_fidelity']:.4f}")
        print(f"    Val loss: {res['final_val_loss']:.4f}, Time: {res['time']:.0f}s")

    return {"cond_dropout": results}


def ablation_ddim_steps(args, device):
    """Ablation: DDIM sampling steps vs fidelity trade-off."""
    print("\n" + "=" * 60)
    print("ABLATION: DDIM Sampling Steps")
    print("=" * 60)

    config = load_config(args.config)
    n_qubits = args.n_qubits or config["n_qubits"]
    d = 2 ** n_qubits
    cond_dim = 6 ** n_qubits

    # Train one model
    print("\n  Training base model...")
    model = DDPM(
        d=d, timesteps=args.timesteps, cond_input_dim=cond_dim,
        base_channels=32, dim_mults=(1, 2), cond_dim=64,
        cond_dropout_prob=0.1, beta_schedule="cosine",
    ).to(device)

    train_dataset = QSTDataset(
        n_qubits=n_qubits, n_states=args.n_train, n_shots=10000,
        regularization_eps=1e-6, seed=42,
    )
    val_dataset = QSTDataset(
        n_qubits=n_qubits, n_states=args.n_val, n_shots=10000,
        regularization_eps=1e-6, seed=1042,
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    model.train()
    for epoch in range(args.epochs):
        for batch in train_loader:
            x_0 = batch["x_0"].to(device)
            condition = batch["condition"].to(device)
            loss = model.training_loss(x_0, condition)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    # Test different DDIM steps
    model.eval()
    ddim_steps_list = [5, 10, 20, 50, 100, 200, 500, 1000]
    results = {}

    print("\n  Evaluating DDIM steps...")
    with torch.no_grad():
        for steps in ddim_steps_list:
            t0 = time.time()
            fids = []
            for i in range(min(20, args.n_val)):
                condition = val_dataset[i]["condition"].unsqueeze(0).to(device)
                use_ddim = steps < args.timesteps
                if use_ddim:
                    x_pred = model.ddim_sample(condition, ddim_steps=steps, progress=False)
                else:
                    x_pred = model.sample(condition, progress=False)
                x_pred_np = x_pred.cpu().numpy()[0]
                rho_pred = cholesky_to_dm(x_pred_np)
                rho_true = val_dataset.density_matrices[i]
                fids.append(fidelity(rho_true, rho_pred))

            elapsed = time.time() - t0
            results[steps] = {
                "mean_fidelity": float(np.mean(fids)),
                "std_fidelity": float(np.std(fids)),
                "time_per_sample": elapsed / min(20, args.n_val),
            }
            print(f"    {steps:4d} steps: Fidelity={results[steps]['mean_fidelity']:.4f}, "
                  f"Time/sample={results[steps]['time_per_sample']:.3f}s")

    return {"ddim_steps": results}


def ablation_noise_schedule(args, device):
    """Ablation: linear vs cosine noise schedule."""
    print("\n" + "=" * 60)
    print("ABLATION: Noise Schedule")
    print("=" * 60)

    config = load_config(args.config)
    n_qubits = args.n_qubits or config["n_qubits"]
    d = 2 ** n_qubits
    cond_dim = 6 ** n_qubits

    schedules = ["linear", "cosine"]
    results = {}

    for sched in schedules:
        print(f"\n  schedule = {sched}")
        t0 = time.time()
        res = run_single_experiment(
            n_qubits=n_qubits, n_train=args.n_train, n_val=args.n_val,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            d=d, cond_dim=cond_dim, timesteps=args.timesteps,
            cond_dropout_prob=0.1, beta_schedule=sched,
            dim_mults=(1, 2), base_channels=32, device=device,
        )
        res["time"] = time.time() - t0
        results[sched] = res
        print(f"    Fidelity: {res['mean_fidelity']:.4f} +/- {res['std_fidelity']:.4f}")
        print(f"    Val loss: {res['final_val_loss']:.4f}")

    return {"noise_schedule": results}


def ablation_model_size(args, device):
    """Ablation: model size (channel multipliers)."""
    print("\n" + "=" * 60)
    print("ABLATION: Model Size")
    print("=" * 60)

    config = load_config(args.config)
    n_qubits = args.n_qubits or config["n_qubits"]
    d = 2 ** n_qubits
    cond_dim = 6 ** n_qubits

    model_configs = [
        ("small", (1,), 16),
        ("medium", (1, 2), 32),
        ("large", (1, 2, 4), 64),
    ]
    results = {}

    for name, mults, base_ch in model_configs:
        # Skip configurations that don't make sense for small d
        if d <= 2 and len(mults) > 2:
            continue

        print(f"\n  model = {name} (mults={mults}, base_ch={base_ch})")
        t0 = time.time()
        res = run_single_experiment(
            n_qubits=n_qubits, n_train=args.n_train, n_val=args.n_val,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
            d=d, cond_dim=cond_dim, timesteps=args.timesteps,
            cond_dropout_prob=0.1, beta_schedule="cosine",
            dim_mults=mults, base_channels=base_ch, device=device,
        )
        res["time"] = time.time() - t0
        results[name] = res
        print(f"    Params: {res['n_params']:,}")
        print(f"    Fidelity: {res['mean_fidelity']:.4f} +/- {res['std_fidelity']:.4f}")

    return {"model_size": results}


def main():
    parser = argparse.ArgumentParser(description="Ablation experiments for DM-QST")
    parser.add_argument("--ablation", type=str, default="all",
                        choices=["cond_dropout", "ddim_steps", "noise_schedule",
                                 "model_size", "all"],
                        help="Which ablation to run")
    parser.add_argument("--config", type=str, default=None, help="Config YAML")
    parser.add_argument("--n_qubits", type=int, default=1, help="Number of qubits")
    parser.add_argument("--n_train", type=int, default=2000, help="Training set size")
    parser.add_argument("--n_val", type=int, default=500, help="Validation set size")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--timesteps", type=int, default=200, help="Diffusion steps")
    parser.add_argument("--output_dir", type=str, default="./outputs/ablations",
                        help="Output directory")
    parser.add_argument("--device", type=str, default=None, help="Device")
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    all_results = {
        "config": {
            "n_qubits": args.n_qubits,
            "n_train": args.n_train,
            "n_val": args.n_val,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "timesteps": args.timesteps,
        }
    }

    ablation_fns = {
        "cond_dropout": ablation_cond_dropout,
        "ddim_steps": ablation_ddim_steps,
        "noise_schedule": ablation_noise_schedule,
        "model_size": ablation_model_size,
    }

    if args.ablation == "all":
        for name, fn in ablation_fns.items():
            all_results.update(fn(args, device))
    else:
        all_results.update(ablation_fns[args.ablation](args, device))

    # Save results
    output_path = os.path.join(args.output_dir, f"ablation_{args.ablation}.json")

    # Convert to serializable format
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {str(k): make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(output_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("ABLATION SUMMARY")
    print("=" * 60)
    for ablation_name, ablation_results in all_results.items():
        if ablation_name == "config":
            continue
        print(f"\n{ablation_name}:")
        for variant, metrics in ablation_results.items():
            if isinstance(metrics, dict) and "mean_fidelity" in metrics:
                print(f"  {variant}: Fidelity = {metrics['mean_fidelity']:.4f}")


if __name__ == "__main__":
    main()
