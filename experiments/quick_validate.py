"""
Quick end-to-end validation of the full DM-QST pipeline.

Runs a minimal training loop (1 qubit, small dataset, few epochs)
to verify that all components integrate correctly.
"""

import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch

from src.data.dataset import QSTDataset
from src.models.diffusion import DDPM
from src.representation.cholesky import cholesky_to_dm
from src.evaluation.metrics import fidelity


def main():
    print("=" * 60)
    print("DM-QST: End-to-End Validation")
    print("=" * 60)

    n_qubits = 1
    d = 2 ** n_qubits
    cond_dim = 6 ** n_qubits
    cholesky_dim = d * d
    n_train = 500
    n_val = 100
    n_shots = 10000
    batch_size = 64
    epochs = 10
    timesteps = 100  # Fewer steps for quick validation

    device = torch.device("cpu")

    print(f"\nConfig:")
    print(f"  Qubits: {n_qubits} (d={d})")
    print(f"  Cholesky dim: {cholesky_dim}")
    print(f"  Condition dim: {cond_dim}")
    print(f"  Train/Val sizes: {n_train}/{n_val}")
    print(f"  Epochs: {epochs}")
    print(f"  Diffusion steps: {timesteps}")
    print(f"  Device: {device}")

    # Generate datasets
    print("\n[1/4] Generating datasets...")
    t0 = time.time()

    train_dataset = QSTDataset(
        n_qubits=n_qubits,
        n_states=n_train,
        n_shots=n_shots,
        regularization_eps=1e-6,
        seed=42,
    )
    val_dataset = QSTDataset(
        n_qubits=n_qubits,
        n_states=n_val,
        n_shots=n_shots,
        regularization_eps=1e-6,
        seed=1042,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    print(f"  Generated in {time.time() - t0:.1f}s")
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Cholesky stats: mean={np.mean(train_dataset.cholesky_vectors):.4f}, "
          f"std={np.std(train_dataset.cholesky_vectors):.4f}")

    # Build model
    print("\n[2/4] Building model...")
    model = DDPM(
        d=d,
        timesteps=timesteps,
        cond_input_dim=cond_dim,
        base_channels=32,
        dim_mults=(1, 2),
        cond_dim=32,
        cond_dropout_prob=0.1,
        beta_schedule="cosine",
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Train
    print("\n[3/4] Training...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)

    train_losses = []
    val_losses = []

    model.train()
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

        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x_0 = batch["x_0"].to(device)
                condition = batch["condition"].to(device)
                val_loss += model.training_loss(x_0, condition).item()
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()

        if (epoch + 1) % 2 == 0 or epoch == 0:
            print(f"  Epoch {epoch + 1}/{epochs}: "
                  f"Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

    print(f"  Final train loss: {train_losses[-1]:.4f}")
    print(f"  Final val loss: {val_losses[-1]:.4f}")

    # Evaluate: reconstruct a few test states
    print("\n[4/4] Evaluating...")
    model.eval()

    test_states = val_dataset.density_matrices[:5]
    fidelities = []

    with torch.no_grad():
        for i in range(5):
            condition = val_dataset[i]["condition"].unsqueeze(0).to(device)

            # DDIM sampling (fast)
            x_pred = model.ddim_sample(
                condition, ddim_steps=20, progress=False
            )
            x_pred_np = x_pred.cpu().numpy()[0]

            rho_pred = cholesky_to_dm(x_pred_np)
            rho_true = test_states[i]

            fid = fidelity(rho_true, rho_pred)
            fidelities.append(fid)
            print(f"  State {i}: Fidelity = {fid:.4f}")

    mean_fid = np.mean(fidelities)
    print(f"\n  Mean fidelity: {mean_fid:.4f}")

    # Check: loss should decrease
    loss_improvement = train_losses[0] - train_losses[-1]
    print(f"\n  Loss improvement: {loss_improvement:.4f}")

    print("\n" + "=" * 60)
    if loss_improvement > 0:
        print("VALIDATION PASSED!")
        print("The pipeline works end-to-end.")
        print("Next step: train with more epochs and data for better fidelity.")
    else:
        print("WARNING: Loss did not decrease. Check model/hyperparameters.")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    main()
