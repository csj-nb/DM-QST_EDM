"""
Train the DDPM-QST model.

Usage:
    python experiments/train.py --config configs/default.yaml
    python experiments/train.py --n_qubits 2 --epochs 100 --batch_size 64
"""

import argparse
import os
import sys
import torch
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config, save_config
from src.data.dataset import create_dataloaders
from src.models.diffusion import DDPM
from src.training.trainer import Trainer


def build_val_fidelity_fn(
    val_dataset,
    config,
    n_states: int = 10,
    shots: int = 300,
    K: int = 3,
):
    """
    Lightweight in-training QST-quality monitor.

    Every val_every epochs, samples K posterior states per val state at a
    FIXED shot count (default 300) and reports mean fidelity to the true
    state. This is the quantity that matters for QST (loss alone is not
    informative: Cholesky L2 can fall while fidelity stalls). Uses EMA
    weights via ema.apply_shadow()/restore().

    Returns:
        fn(model, ema, device) -> float mean fidelity.
    """
    import numpy as np
    import torch

    from src.data.measurements import (
        simulate_measurements,
        arcsin_sqrt_transform,
        fisher_z_transform,
    )
    from src.data.dataset import counts_to_condition_channel
    from src.representation.cholesky import cholesky_to_dm
    from src.representation import hermitian as _herm_rep
    from src.evaluation.metrics import fidelity
    representation = config.get("representation", "cholesky")

    n_qubits = config["n_qubits"]
    use_counts = config.get("use_counts_channel", False)
    use_shot = config.get("use_shot_channel", False)
    n_shots_max = config.get("n_shots_max", 50000)
    model_type = config.get("model_type", "ddpm")

    n_total = len(val_dataset)
    idxs = np.random.RandomState(0).choice(
        n_total, min(n_states, n_total), replace=False
    )

    def _sample(model, cond, device):
        cond_b = cond.repeat(K, 1)
        if model_type == "edm":
            n_steps = config.get("n_steps_eval", 35)
            return model.sample(cond_b, n_steps=n_steps, progress=False)
        if model_type == "flow":
            return model.sample(cond_b, n_steps=4, progress=False)
        if model_type == "vdm":
            return model.sample(cond_b, n_steps=100, progress=False)
        return model.sample(cond_b, progress=False)

    def fn(model, ema, device):
        model.eval()
        ema.apply_shadow()
        fids = []
        try:
            with torch.no_grad():
                for i in idxs:
                    rho_true = val_dataset.density_matrices[i]
                    freqs = simulate_measurements(
                        rho_true, n_shots=shots, seed=1000 + i,
                        return_counts=use_counts,
                    )
                    if use_counts:
                        freqs, counts = freqs
                    if config.get("use_arcsin_sqrt", False):
                        cond_arr = arcsin_sqrt_transform(freqs).astype(np.float32)
                    elif config.get("use_fisher_z", True):
                        cond_arr = fisher_z_transform(freqs, eps=1e-6).astype(np.float32)
                    else:
                        cond_arr = freqs.astype(np.float32)
                    if use_counts:
                        cc = counts_to_condition_channel(counts, n_qubits, shots)
                        cond_arr = np.concatenate([cond_arr, cc], axis=0)
                    if use_shot:
                        shot_feat = np.log1p(shots) / np.log1p(max(n_shots_max, 1))
                        cond_arr = np.concatenate(
                            [cond_arr, np.asarray([shot_feat], dtype=np.float32)],
                            axis=0,
                        )
                    cond = torch.from_numpy(cond_arr).unsqueeze(0).to(device)
                    x_pred = _sample(model, cond, device).cpu().numpy()
                    if representation == "hermitian":
                        rhos = [_herm_rep.vec_to_dm_np(x_pred[j]) for j in range(K)]
                        # Hermitian 表示输出未保证合法（trace/PSD），必须先投影，
                        # 否则 fidelity 对 trace>1 的输入虚高（实测 0.98 假象 -> 投影后 0.54）
                        rhos = [_herm_rep.project_to_valid_np(r) for r in rhos]
                    elif representation == "bloch":
                        from src.representation import bloch as _bloch_rep
                        rhos = [_bloch_rep.vec_to_dm_np(x_pred[j]) for j in range(K)]
                        rhos = [_bloch_rep.project_to_valid_np(r) for r in rhos]
                    else:
                        rhos = [cholesky_to_dm(x_pred[j]) for j in range(K)]
                    rho_est = np.mean(rhos, axis=0)
                    fids.append(fidelity(rho_true, rho_est))
            return float(np.mean(fids))
        finally:
            ema.restore()
            model.train()

    return fn


def main():
    parser = argparse.ArgumentParser(description="Train DDPM-QST")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                        help="Path to config YAML")
    parser.add_argument("--n_qubits", type=int, default=None, help="Override n_qubits")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Output directory")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data cache directory")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu, cuda, cuda:0)")
    parser.add_argument("--resume", type=str, default=None,
                        help="Resume from checkpoint")
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

    # Set global random seeds for reproducibility. The data generation is
    # deterministic (seeded by config), but q_sample noise, dropout, and the
    # DataLoader shuffle are not unless we seed torch/numpy/random here.
    seed = config["data"]["seed"]
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    n_qubits = config["n_qubits"]
    d = 2 ** n_qubits
    # Low-shot representation (scheme 1): arcsin-sqrt transform + counts
    # channel doubles the condition dimension (6^n -> 2*6^n), and the
    # shot-budget feature appends one more scalar.
    use_arcsin_sqrt = config.get("use_arcsin_sqrt", False)
    use_counts_channel = config.get("use_counts_channel", False)
    use_shot_channel = config.get("use_shot_channel", False)
    cholesky_normalize = config.get("cholesky_normalize", False)
    representation = config.get("representation", "cholesky")
    cond_input_dim = 6 ** n_qubits
    if use_counts_channel:
        cond_input_dim *= 2
    if use_shot_channel:
        cond_input_dim += 1
    # Isometric representations (bloch / hermitian_fix_trace) have d*d-1
    # coordinates (trace is fixed); everything else uses d*d.
    if representation in ("bloch", "hermitian_fix_trace"):
        cholesky_dim = d * d - 1
    else:
        cholesky_dim = d * d

    print("=" * 60)
    print("DM-QST: Diffusion Models for Quantum State Tomography")
    print("=" * 60)
    print(f"  Qubits: {n_qubits} (Hilbert dim: {d})")
    print(f"  Cholesky dim: {cholesky_dim}")
    print(f"  Condition dim: {cond_input_dim}")
    print(f"  Diffusion steps: {config['diffusion_timesteps']}")
    print(f"  Beta schedule: {config['beta_schedule']}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Create dataloaders
    print("\nSetting up data loaders...")

    # Variable-shot training: expose model to different noise levels
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
        use_arcsin_sqrt=use_arcsin_sqrt,
        use_counts_channel=use_counts_channel,
        use_shot_channel=use_shot_channel,
        cholesky_normalize=cholesky_normalize,
        representation=representation,
    )

    train_loader = dataloaders["train_loader"]
    val_loader = dataloaders["val_loader"]

    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches: {len(val_loader)}")

    # Build model
    print("\nBuilding model...")
    model_type = config.get("model_type", "ddpm")

    if model_type == "flow":
        from src.models.flow_matching import FlowMatching
        model = FlowMatching(
            d=d,
            cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            loss_type=config.get("loss_type", "l2"),
        )
        print("  Model type: Flow Matching")
    elif model_type == "vdm":
        from src.models.vdm import VDM
        model = VDM(
            d=d,
            cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            loss_type=config.get("loss_type", "l2"),
            gamma_min=config.get("gamma_min", -15.0),
            gamma_max=config.get("gamma_max", 15.0),
            lambda_rank=config.get("lambda_rank", 0.0),
            lambda_rank_warmup=config.get("lambda_rank_warmup", 1000),
        )
        print("  Model type: VDM (Variational Diffusion Model)")
    elif model_type == "edm":
        from src.models.edm import EDM
        model = EDM(
            d=d,
            cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            cond_inject=config.get("cond_inject", "film"),
            loss_space=config.get("loss_space", config.get("model", {}).get("loss_space", "cholesky")),
            loss_type=config.get("loss_type", "l2"),
            sigma_min=config.get("sigma_min", 0.001),
            sigma_max=config.get("sigma_max", 0.8),
            sigma_data_diag=config.get("sigma_data_diag", 0.3),
            sigma_data_off=config.get("sigma_data_off", 0.2),
            sigma_data_per_dim=config.get("sigma_data_per_dim"),
            P_mean=config.get("P_mean", -1.2),
            P_std=config.get("P_std", 1.2),
            representation=representation,
            out_dim=cholesky_dim,
            rho=config.get("rho", 7.0),
            lambda_rank=config.get("lambda_rank", 0.0),
            lambda_rank_warmup=config.get("lambda_rank_warmup", 1000),
            lambda_meas=config.get("lambda_meas", 0.0),
            lambda_meas_warmup=config.get("lambda_meas_warmup", 1000),
            meas_loss_type=config.get("meas_loss_type", "l2"),
            meas_loss_p_min=config.get("meas_loss_p_min", 1e-3),
            use_preconditioning=config.get("use_preconditioning", True),
            # ERDM-style loss reweighting
            use_loss_reweighting=config.get("use_loss_reweighting", True),
        )
        precond_status = "with" if config.get("use_preconditioning", True) else "without"
        reweight_status = "ON" if config.get("use_loss_reweighting", True) else "OFF"
        print(f"  Model type: QST-EDM ({precond_status} preconditioning)")
        print(f"  ERDM loss reweighting: {reweight_status}")
    else:
        model = DDPM(
            d=d,
            timesteps=config["diffusion_timesteps"],
            cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            beta_schedule=config["beta_schedule"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
            loss_type=config.get("loss_type", "l2"),
        )
        print("  Model type: DDPM")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {n_params:,}")

    # Create trainer
    output_dir = os.path.join(args.output_dir, f"qst_n{n_qubits}")
    trainer = Trainer(
        model=model,
        config=config,
        device=device,
        output_dir=output_dir,
    )

    # Inject the in-training fidelity monitor: every val_every epochs, sample
    # a few val states at 300 shots and report mean fidelity (EMA weights).
    # This shows whether QST quality is actually improving -- loss alone is
    # not informative (Cholesky L2 can fall while fidelity stalls).
    trainer.val_fidelity_fn = build_val_fidelity_fn(
        dataloaders["val_dataset"], config,
        n_states=10, shots=300, K=3,
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