"""T1 + T2 posterior diagnostics for diffusion-based QST.

Collects K posterior samples per (state, shot) pair from a trained
checkpoint and computes:

  T1 -- sample diversity / posterior-collapse check
        (experiments expect: diversity DECREASES with shot count, i.e. the
        posterior narrows as measurements accumulate; near-constant high
        pairwise fidelity at ALL shot levels = mode collapse, which voids
        the MMSE claim)

  T2 -- calibration of the model posterior
        PIT-based shape calibration (reliability curve + ECE) and absolute
        coverage of the true state by the sample cloud, per shot level.

Known-true-state synthetic scenario: test states are drawn from the same
ensembles used in training (the "prior"), so a calibrated model must behave
exactly as its posterior predicts.

Usage:
    python experiments/diagnose_posterior.py \
        [--ckpt outputs/abl_stage0/checkpoints/best.pt] \
        [--n-states 30] [--shots 50,100,300,1000] [--K 20] [--plot]

Outputs (JSON):
    outputs/eval_posterior/diversity.json      (T1, per shot level)
    outputs/eval_posterior/calibration.json    (T2, per shot level)
    outputs/eval_posterior/meta.json           (checkpoint + args snapshot)
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

from src.data.dataset import QSTDataset, counts_to_condition_channel
from src.data.measurements import (
    simulate_measurements,
    fisher_z_transform,
    arcsin_sqrt_transform,
)
from src.models.diffusion import DDPM
from src.models.edm import EDM
from src.models.vdm import VDM
from src.models.flow_matching import FlowMatching
from src.representation.cholesky import cholesky_to_dm
from src.evaluation.metrics import fidelity
from src.evaluation.posterior_diagnostics import (
    aggregate_diversity,
    sample_diversity_stats,
    collect_pit,
    pit_calibration,
    cloud_zscore_diagnostics,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CKPT = os.path.join(ROOT, "outputs", "abl_stage0", "checkpoints", "best.pt")
OUT_DIR = os.path.join(ROOT, "outputs", "eval_posterior")


# ---------------------------------------------------------------------------
# Model construction (must match evaluate.py / training)
# ---------------------------------------------------------------------------

def build_model(config):
    """Construct the model exactly as evaluate.py does."""
    n_qubits = config["n_qubits"]
    d = 2 ** n_qubits
    cond_input_dim = 6 ** n_qubits
    if config.get("use_counts_channel", False):
        cond_input_dim *= 2
    if config.get("use_shot_channel", False):
        cond_input_dim += 1
    model_type = config.get("model_type", "ddpm")

    if model_type == "flow":
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
    return model, model_type


def build_condition(meas_freqs, meas_counts, config, n_qubits, n_shots):
    """Build the condition vector exactly as in training / evaluate.py."""
    if config.get("use_arcsin_sqrt", False):
        cond_arr = arcsin_sqrt_transform(meas_freqs).astype(np.float32)
    elif config.get("use_fisher_z", True):
        cond_arr = fisher_z_transform(meas_freqs, eps=1e-6).astype(np.float32)
    else:
        cond_arr = meas_freqs.astype(np.float32)
    if config.get("use_counts_channel", False):
        counts_cond = counts_to_condition_channel(
            meas_counts, n_qubits, n_shots
        )
        cond_arr = np.concatenate([cond_arr, counts_cond], axis=0)
    if config.get("use_shot_channel", False):
        n_shots_max = config.get("n_shots_max", 50000)
        shot_feat = np.log1p(n_shots) / np.log1p(max(n_shots_max, 1))
        cond_arr = np.concatenate(
            [cond_arr, np.asarray([shot_feat], dtype=np.float32)], axis=0
        )
    return cond_arr


def sample_batch(model, model_type, cond_batch, config, eval_cfg,
                 meas_freqs, n_shots, device, n_steps_override=None):
    """Sample K posterior samples in one batched diffusion call (as
    evaluate.py). Returns (K, d*d) Cholesky vectors on CPU numpy."""
    with torch.no_grad():
        if model_type == "vdm":
            n_steps = n_steps_override or config.get("evaluation", {}).get(
                "ddim_steps") or 100
            x_pred = model.sample(cond_batch, n_steps=n_steps, progress=False)
        elif model_type == "edm":
            n_steps = n_steps_override or config.get("n_steps_eval", 35)
            dps_scale = eval_cfg.get("dps_scale", 0.0)
            if dps_scale > 0:
                x_pred = model.sample_dps(
                    cond_batch,
                    torch.from_numpy(meas_freqs.astype(np.float32)).to(device),
                    n_steps=n_steps, dps_scale=dps_scale, progress=False,
                )
            else:
                cfg_mode = eval_cfg.get("cfg_mode", "none")
                w = None
                if cfg_mode == "fixed":
                    w = float(eval_cfg.get("cfg_weight", 2.0))
                elif cfg_mode == "adaptive":
                    w_max = eval_cfg.get("cfg_weight_max", 2.0)
                    n_ref = eval_cfg.get("cfg_n_ref", 1000.0)
                    w = float(w_max) * np.sqrt(n_shots / (n_shots + n_ref))
                if w is None:
                    x_pred = model.sample(cond_batch, n_steps=n_steps,
                                          progress=False)
                else:
                    x_pred = model.sample_with_cfg(
                        cond_batch, cfg_weight=w, n_steps=n_steps,
                        progress=False,
                    )
        elif model_type == "flow":
            n_steps = n_steps_override or 4
            x_pred = model.sample(cond_batch, n_steps=n_steps, progress=False)
        else:
            x_pred = model.sample(cond_batch, progress=False)
    return x_pred.cpu().numpy()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="T1+T2 posterior diagnostics")
    parser.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    parser.add_argument("--n-states", type=int, default=30)
    parser.add_argument("--shots", type=str, default="50,100,300,1000")
    parser.add_argument("--K", type=int, default=20,
                        help="posterior samples per (state, shot)")
    parser.add_argument("--repeats", type=int, default=1,
                        help="independent measurement repeats (averaged)")
    parser.add_argument("--data-dir", type=str,
                        default=os.path.join(ROOT, "data"))
    parser.add_argument("--out-dir", type=str, default=OUT_DIR)
    parser.add_argument("--n-steps", type=int, default=None,
                        help="override ODE steps (evaluate default: 35)")
    parser.add_argument("--plot", action="store_true",
                        help="write PNG plots (needs matplotlib)")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    shots = [int(x) for x in args.shots.split(",") if x.strip()]
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading checkpoint: {args.ckpt}")
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    config = ckpt["config"]
    eval_cfg = config.get("evaluation", {})
    n_qubits = config["n_qubits"]

    model, model_type = build_model(config)
    model.load_state_dict(ckpt["model_state_dict"])
    if "ema_shadow" in ckpt:
        model.load_state_dict(ckpt["ema_shadow"], strict=False)
        print("  Using EMA weights from checkpoint")
    model = model.to(device)
    model.eval()
    print(f"Model: {model_type} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M "
          f"params | cond_dim="
          f"{6**n_qubits * (2 if config.get('use_counts_channel') else 1) + (1 if config.get('use_shot_channel') else 0)}")

    # Test states: same ensembles as training ("known prior" scenario)
    print(f"\nPreparing {args.n_states} test states...")
    eval_cache_path = os.path.join(
        args.data_dir, f"qst_n{n_qubits}_test_eval_{args.n_states}.pkl"
    )
    test_rep_kwargs = dict(
        use_fisher_z=config.get("use_fisher_z", True),
        use_arcsin_sqrt=config.get("use_arcsin_sqrt", False),
        use_counts_channel=config.get("use_counts_channel", False),
        use_shot_channel=config.get("use_shot_channel", False),
    )
    test_dataset = QSTDataset(
        n_qubits=n_qubits,
        n_states=args.n_states,
        n_shots=config["data"]["n_measurement_shots"],
        state_types=config["data"]["state_types"],
        regularization_eps=config["data"]["regularization_eps"],
        seed=config["data"]["seed"] + 2000,
        cache_path=eval_cache_path,
        **test_rep_kwargs,
    )

    diversity_by_shot, calibration_by_shot = {}, {}
    print(f"\nSampling K={args.K} per (state, shot), {args.repeats} repeat(s)...")
    for shot_idx, n_shots in enumerate(shots):
        per_state_stats = []
        pits, f_true_mmse, f_samples_mmse = [], [], []
        t0 = time.time()
        for state_idx in range(args.n_states):
            rho_true = test_dataset.density_matrices[state_idx]
            for rep in range(args.repeats):
                meas_seed = state_idx * 10000 + shot_idx * 1000 + rep
                use_counts = config.get("use_counts_channel", False)
                meas_freqs = simulate_measurements(
                    rho_true, n_shots=n_shots, seed=meas_seed,
                    return_counts=use_counts,
                )
                if use_counts:
                    meas_freqs, meas_counts = meas_freqs
                cond_arr = build_condition(
                    meas_freqs, meas_counts if use_counts else None,
                    config, n_qubits, n_shots,
                )
                condition = torch.from_numpy(cond_arr).unsqueeze(0).to(device)
                cond_batch = condition.repeat(args.K, 1)

                x_pred_np = sample_batch(
                    model, model_type, cond_batch, config, eval_cfg,
                    meas_freqs, n_shots, device, n_steps_override=args.n_steps,
                )
                rho_samples = np.stack([
                    cholesky_to_dm(x_pred_np[i]) for i in range(args.K)
                ])

                # --- T1: diversity / collapse ---
                stats = sample_diversity_stats(
                    rho_samples, rho_true, cholesky_vectors=x_pred_np
                )
                per_state_stats.append(stats)

                # --- T2: PIT (score = fidelity to the MMSE mean) ---
                rho_mmse = np.mean(rho_samples, axis=0)
                f_true = float(fidelity(rho_true, rho_mmse))
                f_samples = [
                    float(fidelity(rho_samples[k], rho_mmse))
                    for k in range(args.K)
                ]
                pits.append(collect_pit(f_true, f_samples))
                # Same scores feed the cloud-drift diagnostic (z-score of the
                # true state within the sample score distribution).
                f_true_mmse.append(f_true)
                f_samples_mmse.append(f_samples)

            if (state_idx + 1) % 10 == 0:
                print(f"  shots={n_shots}: state {state_idx+1}/{args.n_states}", flush=True)

        diversity_by_shot[str(n_shots)] = aggregate_diversity(per_state_stats)
        calibration_by_shot[str(n_shots)] = {
            "pit": pit_calibration(pits),
            "cloud": cloud_zscore_diagnostics(f_true_mmse, f_samples_mmse),
        }
        print(f"  shots={n_shots} done in {time.time()-t0:.1f}s "
              f"(pairwise_fid_mean="
              f"{diversity_by_shot[str(n_shots)]['pairwise_fid_mean_mean']:.4f}, "
              f"PIT ECE="
              f"{calibration_by_shot[str(n_shots)]['pit']['ece']:.4f})",
              flush=True)

    # Save outputs
    with open(os.path.join(args.out_dir, "diversity.json"), "w") as f:
        json.dump(diversity_by_shot, f, indent=2)
    with open(os.path.join(args.out_dir, "calibration.json"), "w") as f:
        json.dump(calibration_by_shot, f, indent=2)
    meta = {
        "ckpt": args.ckpt,
        "n_states": args.n_states,
        "shots": shots,
        "K": args.K,
        "repeats": args.repeats,
        "model_type": model_type,
        "cfg_mode": eval_cfg.get("cfg_mode", "none"),
        "dps_scale": eval_cfg.get("dps_scale", 0.0),
        "iw_mmse": eval_cfg.get("iw_mmse", False),
        "config": {k: v for k, v in config.items()
                   if k not in ("model", "data", "evaluation")},
    }
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nDONE -> {args.out_dir}/{{diversity,calibration,meta}}.json")

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

            # Panel 1: pairwise fidelity (diversity) vs shots
            xs = [int(s) for s in diversity_by_shot]
            ys = [diversity_by_shot[s]["pairwise_fid_mean_mean"] for s in diversity_by_shot]
            err = [diversity_by_shot[s]["pairwise_fid_mean_std"] for s in diversity_by_shot]
            axes[0].errorbar(xs, ys, yerr=err, marker="o", capsize=4)
            axes[0].set_xscale("log"); axes[0].set_xlabel("shots")
            axes[0].set_ylabel("mean pairwise fidelity (samples)")
            axes[0].set_title("T1: posterior diversity vs shot count")
            axes[0].set_ylim(0, 1)

            # Panel 2: PIT calibration curves per shot level
            for s in calibration_by_shot:
                d = calibration_by_shot[s]["pit"]
                axes[1].plot(d["claimed_coverage"], d["empirical_coverage"],
                             marker=".", label=f"{s} shots (ECE={d['ece']:.3f})")
            axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[1].set_xlabel("claimed coverage 1-alpha")
            axes[1].set_ylabel("empirical coverage")
            axes[1].set_title("T2: PIT reliability curves")
            axes[1].legend(fontsize=8)

            # Panel 3: cloud-drift z-score (absolute calibration)
            xs3 = [int(s) for s in calibration_by_shot]
            zs = [calibration_by_shot[s]["cloud"]["z_mean"] for s in calibration_by_shot]
            ze = [calibration_by_shot[s]["cloud"]["z_std"] for s in calibration_by_shot]
            axes[2].errorbar(xs3, zs, yerr=ze, marker="o", capsize=4)
            axes[2].axhline(0, color="k", linestyle="--", alpha=0.5)
            axes[2].set_xscale("log"); axes[2].set_xlabel("shots")
            axes[2].set_ylabel("z-score of true state in sample cloud")
            axes[2].set_title("T2: cloud-drift (0 = honest posterior)")
            for s in calibration_by_shot:
                c = calibration_by_shot[s]["cloud"]
                axes[2].annotate(f"F={c['fid_true_to_mmse_mean']:.3f}",
                                 (int(s), c["z_mean"]), fontsize=8,
                                 textcoords="offset points", xytext=(0, 6))

            fig.tight_layout()
            fig.savefig(os.path.join(args.out_dir, "diagnostics.png"), dpi=150)
            print(f"Plot saved -> {args.out_dir}/diagnostics.png")
        except ImportError:
            print("matplotlib not available; skipping plot.")


if __name__ == "__main__":
    main()
