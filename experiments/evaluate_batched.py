"""
Batched evaluation for QST-EDM (drop-in replacement for experiments/evaluate.py).

Why this exists
---------------
The original evaluate.py samples the K density matrices for each state one at
a time (batch=1 forward pass each, 35 NFE serial), so a full 5000-state sweep
takes 30-100+ hours on GPU and produces NO output until the very end.

This version:
  1. Batches the (n_repeats x K) samples of each state into ONE forward pass
     (condition tensor of shape (n_repeats*K, cond_dim)), cutting the
     sampling wall-clock time by roughly 20-200x on GPU.
  2. Checkpoints results.json after EVERY shot level, so a killed or timed-out
     run never loses the shot levels already completed.

Sampling math is identical to evaluate.py (same measurement seeds, same
condition construction, same MMSE averaging in density-matrix space, same
shot-adaptive CFG), so numbers are directly comparable.

Usage (identical CLI to evaluate.py):
    python experiments/evaluate_batched.py \
      --checkpoint outputs/abl_stage0/checkpoints/final.pt \
      --config configs/n2_lowshot.yaml \
      --n_states 100 --output_dir outputs/eval_lowshot_batched
"""

import argparse
import os
import sys
import json
import time

import numpy as np
import torch

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.utils.config import load_config
from src.data.dataset import QSTDataset, counts_to_condition_channel
from src.data.measurements import (
    simulate_measurements,
    fisher_z_transform,
    arcsin_sqrt_transform,
)
from src.models.diffusion import DDPM
from src.representation.cholesky import cholesky_to_dm
from src.evaluation.metrics import fidelity, trace_distance
from src.evaluation.baselines import mle_reconstruct, linear_inversion
from src.evaluation.plotting import plot_fidelity_vs_shots


def build_condition(config, n_qubits, n_shots, meas_freqs, meas_counts):
    """Build the condition vector exactly as evaluate.py / training does.

    Returns a 1-D numpy array of shape (cond_input_dim,).
    """
    if config.get("use_arcsin_sqrt", False):
        cond_arr = arcsin_sqrt_transform(meas_freqs).astype(np.float32)
    elif config.get("use_fisher_z", True):
        cond_arr = fisher_z_transform(meas_freqs, eps=1e-6).astype(np.float32)
    else:
        cond_arr = meas_freqs.astype(np.float32)
    if config.get("use_counts_channel", False):
        counts_cond = counts_to_condition_channel(meas_counts, n_qubits, n_shots)
        cond_arr = np.concatenate([cond_arr, counts_cond], axis=0)
    return cond_arr


def save_results(results, shot_levels, output_dir):
    """Write results.json (and the fidelity plot) for completed shot levels."""
    os.makedirs(output_dir, exist_ok=True)
    results_json = {
        "n_qubits": results["n_qubits"],
        "n_states": results["n_states"],
        "shot_levels": results["shot_levels"],
        "ddpm": {},
        "mle": {},
        "linear": {},
    }
    for method in ["ddpm", "mle", "linear"]:
        for n_shots in shot_levels:
            fids = results[method][n_shots]["fidelity"]
            tds = results[method][n_shots]["trace_distance"]
            if not fids:
                continue
            results_json[method][str(n_shots)] = {
                "fidelity_mean": float(np.mean(fids)),
                "fidelity_std": float(np.std(fids)),
                "trace_distance_mean": float(np.mean(tds)),
                "trace_distance_std": float(np.std(tds)),
                "n_samples": len(fids),
            }
            if results[method][n_shots]["time"]:
                results_json[method][str(n_shots)]["time_mean"] = float(
                    np.mean(results[method][n_shots]["time"])
                )
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)

    # Fidelity vs shots plot for whatever is available so far.
    ddpm_means, ddpm_stds = [], []
    mle_means, mle_stds = [], []
    linear_means, linear_stds = [], []
    done_levels = []
    for n_shots in shot_levels:
        if not results["ddpm"][n_shots]["fidelity"]:
            continue
        done_levels.append(n_shots)
        ddpm_means.append(np.mean(results["ddpm"][n_shots]["fidelity"]))
        ddpm_stds.append(np.std(results["ddpm"][n_shots]["fidelity"]))
        mle_means.append(np.mean(results["mle"][n_shots]["fidelity"]) if results["mle"][n_shots]["fidelity"] else 0.0)
        mle_stds.append(np.std(results["mle"][n_shots]["fidelity"]) if results["mle"][n_shots]["fidelity"] else 0.0)
        linear_means.append(np.mean(results["linear"][n_shots]["fidelity"]) if results["linear"][n_shots]["fidelity"] else 0.0)
        linear_stds.append(np.std(results["linear"][n_shots]["fidelity"]) if results["linear"][n_shots]["fidelity"] else 0.0)
    if done_levels:
        plot_kwargs = {
            "shot_levels": done_levels,
            "ddpm_fidelities": np.array(ddpm_means),
            "ddpm_std": np.array(ddpm_stds),
            "title": f"QST Fidelity Comparison (n={results['n_qubits']}, {results['n_states']} states)",
            "save_path": os.path.join(output_dir, "fidelity_vs_shots.png"),
            "mle_fidelities": np.array(mle_means),
            "mle_std": np.array(mle_stds),
            "linear_fidelities": np.array(linear_means),
            "linear_std": np.array(linear_stds),
        }
        try:
            plot_fidelity_vs_shots(**plot_kwargs)
        except Exception as e:  # plotting must never kill the run
            print(f"  [plot skipped: {e}]")
    return results_path


def main():
    parser = argparse.ArgumentParser(description="Batched evaluate (QST-EDM)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--n_states", type=int, default=100,
                        help="Number of test states to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data cache directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval_batched",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu, cuda)")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="Sampling steps. Default: config (n_steps_eval / ddim_steps)")
    parser.add_argument("--skip_mle", action="store_true",
                        help="Skip MLE baseline")
    parser.add_argument("--skip_linear", action="store_true",
                        help="Skip linear inversion baseline")
    parser.add_argument("--mle_repeats", type=int, default=None,
                        help="MLE repeats (default: same as n_repeats)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling reproducibility")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Load checkpoint / config ----
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint.get("config", load_config(args.config))
    n_qubits = config["n_qubits"]
    d = 2 ** n_qubits

    # IMPORTANT: model-structure fields (use_arcsin_sqrt / use_counts_channel /
    # model / sigma_* / rho ...) MUST come from the checkpoint so the loaded
    # weights match the architecture. But the EVALUATION settings (shot levels,
    # repeats, MMSE samples, CFG mode) should come from --config when given,
    # so a run can re-evaluate an old checkpoint with the latest eval recipe.
    if args.config:
        file_cfg = load_config(args.config)
        if "evaluation" in file_cfg:
            print("  Overriding evaluation settings from --config "
                  f"({args.config})")
            config["evaluation"] = file_cfg["evaluation"]

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Qubits: {n_qubits}, Hilbert dim: {d}, Device: {device}")

    # ---- Build model (same branches as evaluate.py) ----
    cond_input_dim = 6 ** n_qubits
    if config.get("use_counts_channel", False):
        cond_input_dim *= 2
    model_type = config.get("model_type", "ddpm")

    if model_type == "flow":
        from src.models.flow_matching import FlowMatching
        model = FlowMatching(
            d=d, cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            loss_type=config.get("loss_type", "l2"),
        )
    elif model_type == "vdm":
        from src.models.vdm import VDM
        model = VDM(
            d=d, cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            loss_type=config.get("loss_type", "l2"),
        )
    elif model_type == "edm":
        from src.models.edm import EDM
        model = EDM(
            d=d, cond_input_dim=cond_input_dim,
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
            use_preconditioning=config.get("use_preconditioning", True),
            lambda_rank=config.get("lambda_rank", 0.0),
            lambda_rank_warmup=config.get("lambda_rank_warmup", 1000),
            lambda_meas=config.get("lambda_meas", 0.0),
            lambda_meas_warmup=config.get("lambda_meas_warmup", 1000),
            use_loss_reweighting=config.get("use_loss_reweighting", True),
        )
    else:
        model = DDPM(
            d=d, timesteps=config["diffusion_timesteps"],
            cond_input_dim=cond_input_dim,
            base_channels=config["model"].get("base_channels", 64),
            dim_mults=tuple(config["model"]["dim_mults"]),
            num_res_blocks=config["model"].get("num_res_blocks", 2),
            cond_dim=config["model"]["cond_dim"],
            cond_dropout_prob=config["model"]["cond_dropout_prob"],
            beta_schedule=config["beta_schedule"],
            beta_start=config["beta_start"],
            beta_end=config["beta_end"],
        )
    model.load_state_dict(checkpoint["model_state_dict"])
    if "ema_shadow" in checkpoint:
        model.load_state_dict(checkpoint["ema_shadow"], strict=False)
        print("  Using EMA weights from checkpoint")
    model = model.to(device)
    model.eval()
    print(f"  Model type: {model_type.upper()}, params: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

    # ---- Eval settings ----
    eval_cfg = config["evaluation"]
    shot_levels = eval_cfg["shot_levels"]
    n_repeats = eval_cfg.get("n_repeats", 5)
    mle_repeats = args.mle_repeats if args.mle_repeats is not None else n_repeats
    K = eval_cfg.get("n_samples_per_state", 1)

    cfg_mode = eval_cfg.get("cfg_mode", "none")
    cfg_weight_fixed = eval_cfg.get("cfg_weight", 2.0)
    cfg_weight_max = eval_cfg.get("cfg_weight_max", 2.0)
    cfg_n_ref = eval_cfg.get("cfg_n_ref", 1000.0)

    def shot_cfg_weight(n_shots):
        if cfg_mode == "fixed":
            return float(cfg_weight_fixed)
        if cfg_mode == "adaptive":
            return float(cfg_weight_max) * np.sqrt(n_shots / (n_shots + cfg_n_ref))
        return None

    print(f"  CFG mode: {cfg_mode}"
          + (f" (w_max={cfg_weight_max}, n_ref={cfg_n_ref})" if cfg_mode == "adaptive"
             else f" (w={cfg_weight_fixed})" if cfg_mode == "fixed" else ""))
    print(f"  n_states={args.n_states}, shot_levels={shot_levels}, "
          f"n_repeats={n_repeats}, K={K} samples/state (batched)")

    # ---- Test states ----
    print(f"\nPreparing {args.n_states} test states...")
    # NOTE: include n_states in the cache file name so a small run (e.g. a
    # 5-state smoke test) never overwrites the cache a larger run needs
    # (the original evaluate.py uses a fixed name and gets clobbered).
    cache_path = os.path.join(
        args.data_dir, f"qst_n{n_qubits}_test_eval_{args.n_states}.pkl"
    )
    test_dataset = QSTDataset(
        n_qubits=n_qubits,
        n_states=args.n_states,
        n_shots=config["data"]["n_measurement_shots"],
        state_types=config["data"]["state_types"],
        regularization_eps=config["data"]["regularization_eps"],
        seed=config["data"]["seed"] + 2000,
        cache_path=cache_path,
    )

    results = {
        "n_qubits": n_qubits,
        "n_states": args.n_states,
        "shot_levels": shot_levels,
        "ddpm": {s: {"fidelity": [], "trace_distance": [], "time": []} for s in shot_levels},
        "mle": {s: {"fidelity": [], "trace_distance": [], "time": []} for s in shot_levels},
        "linear": {s: {"fidelity": [], "trace_distance": [], "time": []} for s in shot_levels},
    }

    # ---- Batched evaluation loop ----
    print(f"\nEvaluating {len(shot_levels)} shot levels x {args.n_states} states "
          f"(batched {n_repeats}x{K} per forward pass)...")
    print("=" * 60)

    for shot_idx, n_shots in enumerate(shot_levels):
        t_shot_start = time.time()
        print(f"\nShot level: {n_shots} ({shot_idx + 1}/{len(shot_levels)})")

        for state_idx in range(args.n_states):
            rho_true = test_dataset.density_matrices[state_idx]

            # Build the condition for EVERY repeat up front, then stack into
            # a single (n_repeats*K, cond_dim) batch for ONE forward pass.
            cond_list = []
            for rep in range(n_repeats):
                meas_seed = state_idx * 10000 + shot_idx * 1000 + rep
                use_counts = config.get("use_counts_channel", False)
                meas = simulate_measurements(
                    rho_true, n_shots=n_shots, seed=meas_seed,
                    return_counts=use_counts,
                )
                if use_counts:
                    meas_freqs, meas_counts = meas
                else:
                    meas_freqs, meas_counts = meas, None
                cond_list.append(
                    build_condition(config, n_qubits, n_shots, meas_freqs, meas_counts)
                )

            cond_np = np.stack(cond_list)                       # (R, D)
            cond_batch = np.repeat(cond_np, K, axis=0)          # (R*K, D)
            cond_t = torch.from_numpy(cond_batch).to(device)

            t0 = time.time()
            with torch.no_grad():
                if model_type == "vdm":
                    n_steps = args.ddim_steps or eval_cfg.get("ddim_steps") or 100
                    x_pred = model.sample(cond_t, n_steps=n_steps, progress=False)
                elif model_type == "edm":
                    n_steps = args.ddim_steps or config.get("n_steps_eval", 35)
                    w = shot_cfg_weight(n_shots)
                    if w is None:
                        x_pred = model.sample(cond_t, n_steps=n_steps, progress=False)
                    else:
                        x_pred = model.sample_with_cfg(
                            cond_t, cfg_weight=w, n_steps=n_steps, progress=False
                        )
                elif model_type == "flow":
                    n_steps = args.ddim_steps or 4
                    x_pred = model.sample(cond_t, n_steps=n_steps, progress=False)
                elif args.ddim_steps and args.ddim_steps < config["diffusion_timesteps"]:
                    x_pred = model.ddim_sample(cond_t, ddim_steps=args.ddim_steps, progress=False)
                else:
                    x_pred = model.sample(cond_t, progress=False)

            # cholesky_to_dm accepts (..., d*d) -> (..., d, d); average over K
            # samples in density-matrix space (MMSE posterior mean).
            rho_all = cholesky_to_dm(x_pred.cpu().numpy())      # (R*K, d, d)
            rho_avg = rho_all.reshape(n_repeats, K, d, d).mean(axis=1)  # (R, d, d)
            ddpm_time = (time.time() - t0) / n_repeats

            for rep in range(n_repeats):
                # Baselines reuse the SAME measurement condition as the model,
                # so re-simulate per rep (identical seeds as above).
                meas_seed = state_idx * 10000 + shot_idx * 1000 + rep
                meas = simulate_measurements(
                    rho_true, n_shots=n_shots, seed=meas_seed,
                    return_counts=config.get("use_counts_channel", False),
                )
                if config.get("use_counts_channel", False):
                    meas_freqs = meas[0]
                else:
                    meas_freqs = meas

                if rep < mle_repeats and not args.skip_mle:
                    t_mle = time.time()
                    rho_mle = mle_reconstruct(
                        meas_freqs, n_shots=n_shots,
                        max_iter=eval_cfg["mle_max_iter"],
                        regularization=eval_cfg["mle_regularization"],
                    )
                    results["mle"][n_shots]["fidelity"].append(fidelity(rho_true, rho_mle))
                    results["mle"][n_shots]["trace_distance"].append(trace_distance(rho_true, rho_mle))
                    results["mle"][n_shots]["time"].append(time.time() - t_mle)

                if rep < mle_repeats and not args.skip_linear:
                    t_lin = time.time()
                    rho_linear = linear_inversion(meas_freqs, n_shots=n_shots)
                    results["linear"][n_shots]["fidelity"].append(fidelity(rho_true, rho_linear))
                    results["linear"][n_shots]["trace_distance"].append(trace_distance(rho_true, rho_linear))
                    results["linear"][n_shots]["time"].append(time.time() - t_lin)

                fid_ddpm = fidelity(rho_true, rho_avg[rep])
                td_ddpm = trace_distance(rho_true, rho_avg[rep])
                results["ddpm"][n_shots]["fidelity"].append(fid_ddpm)
                results["ddpm"][n_shots]["trace_distance"].append(td_ddpm)
                results["ddpm"][n_shots]["time"].append(ddpm_time)

            if (state_idx + 1) % 20 == 0:
                print(f"  State {state_idx + 1}/{args.n_states} "
                      f"(shot={n_shots}, {time.time() - t_shot_start:.0f}s elapsed)")

        # ---- Checkpoint after EVERY shot level: nothing is lost on kill ----
        results_path = save_results(results, shot_levels, args.output_dir)
        print(f"  [checkpointed] {results_path} (shot level {n_shots} done, "
              f"{time.time() - t_shot_start:.0f}s for this level)")

    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    for n_shots in shot_levels:
        fids = np.array(results["ddpm"][n_shots]["fidelity"])
        if len(fids) == 0:
            continue
        print(f"\nShots: {n_shots}")
        print(f"  DDPM:   Fidelity = {np.mean(fids):.4f} ± {np.std(fids):.4f}")
        if not args.skip_mle and results["mle"][n_shots]["fidelity"]:
            mf = np.array(results["mle"][n_shots]["fidelity"])
            print(f"  MLE:    Fidelity = {np.mean(mf):.4f} ± {np.std(mf):.4f}")
        if not args.skip_linear and results["linear"][n_shots]["fidelity"]:
            lf = np.array(results["linear"][n_shots]["fidelity"])
            print(f"  Linear: Fidelity = {np.mean(lf):.4f} ± {np.std(lf):.4f}")

    print(f"\nResults saved to: {results_path}")
    print("Evaluation complete!")


if __name__ == "__main__":
    main()
