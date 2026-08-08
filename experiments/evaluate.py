"""
Evaluate trained DDPM-QST model against baselines.

Usage:
    python experiments/evaluate.py --checkpoint outputs/abl_stage0/checkpoints/best.pt
    python experiments/evaluate.py --checkpoint outputs/abl_stage0/checkpoints/best.pt --n_states 100
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
from src.data.dataset import QSTDataset
from src.data.measurements import simulate_measurements
from src.models.diffusion import DDPM
from src.representation.cholesky import cholesky_to_dm
from src.evaluation.metrics import (
    fidelity,
    trace_distance,
    purity,
    hilbert_schmidt_distance,
)
from src.evaluation.baselines import mle_reconstruct, linear_inversion
from src.evaluation.plotting import (
    plot_fidelity_vs_shots,
    plot_density_matrix,
    plot_state_type_breakdown,
)


def main():
    parser = argparse.ArgumentParser(description="Evaluate DDPM-QST")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to config YAML")
    parser.add_argument("--n_states", type=int, default=100,
                        help="Number of test states to evaluate")
    parser.add_argument("--data_dir", type=str, default="./data",
                        help="Data cache directory")
    parser.add_argument("--output_dir", type=str, default="./outputs/eval",
                        help="Output directory for results")
    parser.add_argument("--device", type=str, default=None,
                        help="Device (cpu, cuda)")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="Sampling steps. Default: use config (n_steps_eval / ddim_steps / model default)")
    parser.add_argument("--skip_mle", action="store_true",
                        help="Skip slow MLE baseline")
    parser.add_argument("--skip_linear", action="store_true",
                        help="Skip linear inversion baseline")
    parser.add_argument("--mle_repeats", type=int, default=None,
                        help="Number of measurement repeats for MLE/linear baselines "
                             "(default: same as n_repeats, for a fair comparison)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Random seed for sampling reproducibility")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature: amplifies initial noise, widening the "
                             "sampled posterior (counters over-confidence; 1.0 = default)")
    args = parser.parse_args()

    # Deterministic sampling (measurement seeds are already fixed by
    # state_idx/shot_idx/rep; this also seeds torch RNG for DDPM noise).
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.output_dir, exist_ok=True)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = checkpoint.get("config", {})
    # An explicit --config overrides the checkpoint's stored config for the
    # evaluation-relevant fields (evaluation grid, shot range, representation
    # flags). This lets us re-evaluate an old checkpoint on a new shot grid
    # without retraining.
    if args.config:
        cli_cfg = load_config(args.config)
        for k, v in cli_cfg.items():
            if k == "evaluation" and isinstance(v, dict) and isinstance(config.get("evaluation"), dict):
                config["evaluation"].update(v)
            else:
                config[k] = v
    n_qubits = config["n_qubits"]
    d = 2 ** n_qubits

    print(f"  Qubits: {n_qubits}, Hilbert dim: {d}")

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Build model (supports both DDPM and FlowMatching)
    cond_input_dim = 6 ** n_qubits
    # Low-shot representation (scheme 1): counts channel doubles the dim,
    # and the shot-budget feature appends one scalar.
    if config.get("use_counts_channel", False):
        cond_input_dim *= 2
    if config.get("use_shot_channel", False):
        cond_input_dim += 1
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
        )
        print("  Model type: VDM")
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
            loss_type=config.get("loss_type", "l2"),
            sigma_min=config.get("sigma_min", 0.001),
            sigma_max=config.get("sigma_max", 0.8),
            sigma_data_diag=config.get("sigma_data_diag", 0.3),
            sigma_data_off=config.get("sigma_data_off", 0.2),
            sigma_data_per_dim=config.get("sigma_data_per_dim"),
            P_mean=config.get("P_mean", -1.2),
            P_std=config.get("P_std", 1.2),
            rho=config.get("rho", 7.0),
            # CRITICAL: use_preconditioning must match training. A checkpoint
            # trained with use_preconditioning: false (ablation B) would be
            # silently evaluated with preconditioning enabled otherwise.
            use_preconditioning=config.get("use_preconditioning", True),
            lambda_rank=config.get("lambda_rank", 0.0),
            lambda_rank_warmup=config.get("lambda_rank_warmup", 1000),
            lambda_meas=config.get("lambda_meas", 0.0),
            lambda_meas_warmup=config.get("lambda_meas_warmup", 1000),
            # ERDM-style loss reweighting (must match training config)
            use_loss_reweighting=config.get("use_loss_reweighting", True),
            # Representation must match training: "cholesky" or "hermitian".
            # Without this the EDM defaults to cholesky and samples get decoded
            # with the wrong map (Hermitian vectors passed to cholesky_to_dm).
            representation=config.get("representation", "cholesky"),
            out_dim=(d * d - 1 if config.get("representation", "cholesky") in ("bloch", "hermitian_fix_trace") else d * d),
        )
        print("  Model type: QST-EDM")
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
        )
        print("  Model type: DDPM")
    # Load the raw weights first (full state_dict), then overlay the EMA
    # shadow (which only contains requires_grad parameters) if present.
    # EMA weights are what the trainer used for best-model selection, so
    # evaluating with them matches training and generally performs better.
    model.load_state_dict(checkpoint["model_state_dict"])
    if "ema_shadow" in checkpoint:
        model.load_state_dict(checkpoint["ema_shadow"], strict=False)
        print("  Using EMA weights from checkpoint")
    model = model.to(device)
    model.eval()

    eval_cfg = config["evaluation"]
    shot_levels = eval_cfg["shot_levels"]
    n_repeats = eval_cfg.get("n_repeats", 5)
    # Baselines should average over the same measurement repeats as the
    # diffusion model for a fair comparison (previously MLE only used rep==0).
    mle_repeats = args.mle_repeats if args.mle_repeats is not None else n_repeats

    # Load or generate test states.
    # NOTE: use an _eval-suffixed cache file so that a small evaluation run
    # (e.g. --n_states 100) never overwrites the training pipeline's
    # qst_n{n}_test.pkl (which create_dataloaders expects to hold n_test_states).
    # The n_states suffix prevents runs of different sizes from silently
    # reusing each other's cache (the cache validator does not check n_states).
    print(f"\nPreparing {args.n_states} test states...")
    eval_cache_path = os.path.join(
        args.data_dir, f"qst_n{n_qubits}_test_eval_{args.n_states}.pkl"
    )
    # 与训练一致的表示参数，否则缓存校验不匹配会反复重新生成。
    test_rep_kwargs = dict(
        use_fisher_z=config.get("use_fisher_z", True),
        use_arcsin_sqrt=config.get("use_arcsin_sqrt", False),
        use_counts_channel=config.get("use_counts_channel", False),
        use_shot_channel=config.get("use_shot_channel", False),
        cholesky_normalize=config.get("cholesky_normalize", False),
    )
    try:
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
    except FileNotFoundError:
        print("  Test cache not found, generating fresh...")
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

    # Results storage
    results = {
        "n_qubits": n_qubits,
        "n_states": args.n_states,
        "shot_levels": shot_levels,
        "ddpm": {s: {"fidelity": [], "trace_distance": [], "hs_distance": [], "time": []} for s in shot_levels},
        "mle": {s: {"fidelity": [], "trace_distance": [], "hs_distance": [], "time": []} for s in shot_levels},
        "linear": {s: {"fidelity": [], "trace_distance": [], "hs_distance": [], "time": []} for s in shot_levels},
    }

    # Evaluate
    print(f"\nEvaluating over {len(shot_levels)} shot levels x {args.n_states} states...")
    print("=" * 60)

    # --- Shot-adaptive CFG ---
    # Guidance strength grows with shot count: low-shot measurements are
    # noisy, so the model should lean on the learned prior (weak guidance);
    # high-shot measurements are reliable, so conditioning can be amplified
    # (strong guidance). Modes:
    #   "none"     - plain conditional sampling (default, no CFG)
    #   "fixed"    - constant cfg_weight at all shot levels
    #   "adaptive" - w(n) = w_max * sqrt(n / (n + n_ref))
    cfg_mode = eval_cfg.get("cfg_mode", "none")
    cfg_weight_fixed = eval_cfg.get("cfg_weight", 2.0)
    cfg_weight_max = eval_cfg.get("cfg_weight_max", 2.0)
    cfg_n_ref = eval_cfg.get("cfg_n_ref", 1000.0)

    def shot_cfg_weight(n_shots: int):
        if cfg_mode == "fixed":
            return float(cfg_weight_fixed)
        if cfg_mode == "adaptive":
            return float(cfg_weight_max) * np.sqrt(n_shots / (n_shots + cfg_n_ref))
        return None

    print(f"  CFG mode: {cfg_mode}"
          + (f" (w_max={cfg_weight_max}, n_ref={cfg_n_ref})" if cfg_mode == "adaptive"
             else f" (w={cfg_weight_fixed})" if cfg_mode == "fixed" else ""))

    for shot_idx, n_shots in enumerate(shot_levels):
        print(f"\nShot level: {n_shots} ({shot_idx + 1}/{len(shot_levels)})")

        for state_idx in range(args.n_states):
            if (state_idx + 1) % 20 == 0:
                print(f"  State {state_idx + 1}/{args.n_states}")

            rho_true = test_dataset.density_matrices[state_idx]

            for rep in range(n_repeats):
                # Simulate fresh measurements at this shot level. When the
                # counts channel is used, also fetch the raw counts (the
                # sufficient statistic) to build the training-consistent
                # condition vector.
                meas_seed = state_idx * 10000 + shot_idx * 1000 + rep
                use_counts = config.get("use_counts_channel", False)
                meas_freqs = simulate_measurements(
                    rho_true, n_shots=n_shots, seed=meas_seed,
                    return_counts=use_counts,
                )
                if use_counts:
                    meas_freqs, meas_counts = meas_freqs

                # Build the condition exactly as in training:
                #   arcsin-sqrt (if enabled) > Fisher z (default) > raw freq,
                #   optionally concatenated with the log1p-normalized counts.
                from src.data.dataset import counts_to_condition_channel
                from src.data.measurements import (
                    fisher_z_transform,
                    arcsin_sqrt_transform,
                )

                if config.get("use_arcsin_sqrt", False):
                    cond_arr = arcsin_sqrt_transform(meas_freqs).astype(np.float32)
                elif config.get("use_fisher_z", True):
                    cond_arr = fisher_z_transform(meas_freqs, eps=1e-6).astype(np.float32)
                else:
                    cond_arr = meas_freqs.astype(np.float32)
                if use_counts:
                    counts_cond = counts_to_condition_channel(
                        meas_counts, n_qubits, n_shots
                    )
                    cond_arr = np.concatenate([cond_arr, counts_cond], axis=0)
                if config.get("use_shot_channel", False):
                    # Shot-budget feature (must match training normalization:
                    # log1p(n_shots)/log1p(n_shots_max), n_shots_max from config).
                    n_shots_max = config.get("n_shots_max", 50000)
                    shot_feat = np.log1p(n_shots) / np.log1p(max(n_shots_max, 1))
                    cond_arr = np.concatenate(
                        [cond_arr, np.asarray([shot_feat], dtype=np.float32)], axis=0
                    )
                condition = torch.from_numpy(cond_arr).unsqueeze(0).to(device)

                # --- Posterior-mean (MMSE) sampling ---
                # Sample K times and average in DENSITY-MATRIX space: the
                # average of legal density matrices is a legal state, and
                # converges to the Bayesian posterior mean (MMSE estimate
                # under Hilbert-Schmidt loss) — a theoretical advantage over
                # MLE's mode estimate that grows as the posterior widens
                # (i.e. at low shot counts). Averaging Cholesky vectors
                # instead would break legality and the MMSE interpretation.
                K = eval_cfg.get("n_samples_per_state", 1)
                t0 = time.time()
                with torch.no_grad():
                    # Batch-sample all K posterior samples in ONE diffusion
                    # call (~20x faster than K serial calls: ~51 vs ~2.5
                    # samples/s on RTX 3090). condition is (1, cond_dim);
                    # repeat to (K, cond_dim) and average in DM space.
                    cond_batch = condition.repeat(K, 1)
                    if model_type == "vdm":
                        n_vdm_steps = args.ddim_steps or config.get("evaluation", {}).get("ddim_steps") or 100
                        x_pred = model.sample(cond_batch, n_steps=n_vdm_steps, progress=False)
                    elif model_type == "edm":
                        n_edm_steps = args.ddim_steps or config.get("n_steps_eval", 35)
                        dps_scale = eval_cfg.get("dps_scale", 0.0)
                        if dps_scale > 0:
                            # Stage 4: DPS guidance with explicit Born-rule
                            # likelihood (overrides the CFG path when enabled).
                            x_pred = model.sample_dps(
                                cond_batch,
                                torch.from_numpy(meas_freqs.astype(np.float32)).to(device),
                                n_steps=n_edm_steps, dps_scale=dps_scale,
                                progress=False,
                            )
                        else:
                            w = shot_cfg_weight(n_shots)
                            if w is None:
                                x_pred = model.sample(cond_batch, n_steps=n_edm_steps, progress=False, temperature=args.temperature)
                            else:
                                # Classifier-free guidance: amplify conditioning
                                # (strength from cfg_mode; shot-adaptive when set).
                                x_pred = model.sample_with_cfg(
                                    cond_batch, cfg_weight=w,
                                    n_steps=n_edm_steps, progress=False,
                                    temperature=args.temperature,
                                )
                    elif model_type == "flow":
                        n_flow_steps = args.ddim_steps or 4
                        x_pred = model.sample(cond_batch, n_steps=n_flow_steps, progress=False)
                    elif args.ddim_steps and args.ddim_steps < config["diffusion_timesteps"]:
                        x_pred = model.ddim_sample(
                            cond_batch,
                            ddim_steps=args.ddim_steps,
                            progress=False,
                        )
                    else:
                        x_pred = model.sample(cond_batch, progress=False)
                    x_pred_np = x_pred.cpu().numpy()
                    # Decode in the representation the model was trained in.
                    # Hermitian outputs are raw rho params: build rho, then
                    # project to the nearest valid state (trace-1, PSD) -- the
                    # same path train.py's in-training fidelity monitor uses.
                    model_rep = config.get("representation", "cholesky")
                    if model_rep == "hermitian":
                        from src.representation import hermitian as _herm_rep
                        rho_samples = [
                            _herm_rep.project_to_valid_np(
                                _herm_rep.vec_to_dm_np(x_pred_np[i])
                            )
                            for i in range(K)
                        ]
                    elif model_rep == "bloch":
                        from src.representation import bloch as _bloch_rep
                        rho_samples = [
                            _bloch_rep.project_to_valid_np(
                                _bloch_rep.vec_to_dm_np(x_pred_np[i])
                            )
                            for i in range(K)
                        ]
                    else:
                        rho_samples = [
                            cholesky_to_dm(x_pred_np[i]) for i in range(K)
                        ]

                # --- Importance-weighted posterior mean (IW-MMSE) ---
                # Weight each sample by its Born-rule likelihood under the
                # observed frequencies,  w_k = p(m | rho_k) = prod_j m_j^...
                # (log-space:  log p(m|rho) = sum_j m_j log p_j). Samples that
                # contradict the measurement contribute less than under uniform
                # averaging, correcting the bias of the ODE-induced sample
                # distribution toward the true posterior mean E[rho | m].
                if eval_cfg.get("iw_mmse", False) and K > 1:
                    from src.data.measurements import compute_measurement_probabilities
                    ll = np.array([
                        float(np.sum(
                            meas_freqs * np.log(np.clip(
                                compute_measurement_probabilities(r), 1e-12, 1.0))
                        ))
                        for r in rho_samples
                    ])
                    ll -= ll.max()                      # numerical stability
                    wts = np.exp(ll)
                    wts = wts / (wts.sum() + 1e-12)
                    rho_ddpm = np.tensordot(wts, np.stack(rho_samples), axes=([0], [0]))
                else:
                    rho_ddpm = np.mean(rho_samples, axis=0)
                ddpm_time = time.time() - t0

                # MLE / linear reconstruction: run on every repeat up to
                # mle_repeats (default = n_repeats) so that the baselines are
                # averaged over the SAME measurement noise realizations as the
                # diffusion model. Previously they ran only on rep==0, which
                # systematically under-averaged the baseline shot noise.
                if rep < mle_repeats and not args.skip_mle:
                    t0 = time.time()
                    rho_mle = mle_reconstruct(
                        meas_freqs,
                        n_shots=n_shots,
                        max_iter=eval_cfg["mle_max_iter"],
                        regularization=eval_cfg["mle_regularization"],
                    )
                    mle_time = time.time() - t0
                    fid_mle = fidelity(rho_true, rho_mle)
                    td_mle = trace_distance(rho_true, rho_mle)
                    hs_mle = hilbert_schmidt_distance(rho_true, rho_mle)
                    results["mle"][n_shots]["fidelity"].append(fid_mle)
                    results["mle"][n_shots]["trace_distance"].append(td_mle)
                    results["mle"][n_shots]["hs_distance"].append(hs_mle)
                    results["mle"][n_shots]["time"].append(mle_time)

                if rep < mle_repeats and not args.skip_linear:
                    t0 = time.time()
                    rho_linear = linear_inversion(meas_freqs, n_shots=n_shots)
                    linear_time = time.time() - t0
                    fid_lin = fidelity(rho_true, rho_linear)
                    td_lin = trace_distance(rho_true, rho_linear)
                    hs_lin = hilbert_schmidt_distance(rho_true, rho_linear)
                    results["linear"][n_shots]["fidelity"].append(fid_lin)
                    results["linear"][n_shots]["trace_distance"].append(td_lin)
                    results["linear"][n_shots]["hs_distance"].append(hs_lin)
                    results["linear"][n_shots]["time"].append(linear_time)

                # Compute metrics for DDPM
                fid_ddpm = fidelity(rho_true, rho_ddpm)
                td_ddpm = trace_distance(rho_true, rho_ddpm)
                hs_ddpm = hilbert_schmidt_distance(rho_true, rho_ddpm)
                results["ddpm"][n_shots]["fidelity"].append(fid_ddpm)
                results["ddpm"][n_shots]["trace_distance"].append(td_ddpm)
                results["ddpm"][n_shots]["hs_distance"].append(hs_ddpm)
                results["ddpm"][n_shots]["time"].append(ddpm_time)

    # Summary statistics
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)

    ddpm_mean_fid = []
    ddpm_std_fid = []
    mle_mean_fid = []
    mle_std_fid = []
    linear_mean_fid = []
    linear_std_fid = []

    for n_shots in shot_levels:
        ddpm_fids = np.array(results["ddpm"][n_shots]["fidelity"])
        ddpm_mean = np.mean(ddpm_fids)
        ddpm_std = np.std(ddpm_fids)
        ddpm_mean_fid.append(ddpm_mean)
        ddpm_std_fid.append(ddpm_std)

        print(f"\nShots: {n_shots}")
        print(f"  DDPM:  Fidelity = {ddpm_mean:.4f} ± {ddpm_std:.4f}")

        if not args.skip_mle:
            mle_fids = np.array(results["mle"][n_shots]["fidelity"])
            mle_mean = np.mean(mle_fids)
            mle_std = np.std(mle_fids)
            mle_mean_fid.append(mle_mean)
            mle_std_fid.append(mle_std)
            print(f"  MLE:   Fidelity = {mle_mean:.4f} ± {mle_std:.4f}")

        if not args.skip_linear:
            lin_fids = np.array(results["linear"][n_shots]["fidelity"])
            lin_mean = np.mean(lin_fids)
            lin_std = np.std(lin_fids)
            linear_mean_fid.append(lin_mean)
            linear_std_fid.append(lin_std)
            print(f"  Linear:Fidelity = {lin_mean:.4f} ± {lin_std:.4f}")

        # Time comparison
        ddpm_times = np.array(results["ddpm"][n_shots]["time"])
        print(f"  DDPM time:  {np.mean(ddpm_times):.3f}s ± {np.std(ddpm_times):.3f}s")
        if not args.skip_mle:
            mle_times = np.array(results["mle"][n_shots]["time"])
            print(f"  MLE time:   {np.mean(mle_times):.3f}s ± {np.std(mle_times):.3f}s")

    # Plot fidelity vs shots
    plot_kwargs = {
        "shot_levels": shot_levels,
        "ddpm_fidelities": np.array(ddpm_mean_fid),
        "ddpm_std": np.array(ddpm_std_fid),
        "title": f"QST Fidelity Comparison (n={n_qubits})",
        "save_path": os.path.join(args.output_dir, "fidelity_vs_shots.png"),
    }

    if not args.skip_mle:
        plot_kwargs["mle_fidelities"] = np.array(mle_mean_fid)
        plot_kwargs["mle_std"] = np.array(mle_std_fid)
    else:
        plot_kwargs["mle_fidelities"] = np.zeros_like(ddpm_mean_fid)
        plot_kwargs["mle_std"] = np.zeros_like(ddpm_mean_fid)

    if not args.skip_linear:
        plot_kwargs["linear_fidelities"] = np.array(linear_mean_fid)
        plot_kwargs["linear_std"] = np.array(linear_std_fid)

    plot_fidelity_vs_shots(**plot_kwargs)
    print(f"\nFidelity plot saved to: {plot_kwargs['save_path']}")

    # Save raw results
    results_path = os.path.join(args.output_dir, "results.json")
    # Convert numpy arrays for JSON serialization
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
            results_json[method][str(n_shots)] = {
                "fidelity_mean": float(np.mean(results[method][n_shots]["fidelity"])),
                "fidelity_std": float(np.std(results[method][n_shots]["fidelity"])),
                "trace_distance_mean": float(np.mean(results[method][n_shots]["trace_distance"])),
                "trace_distance_std": float(np.std(results[method][n_shots]["trace_distance"])),
                "hs_distance_mean": float(np.mean(results[method][n_shots]["hs_distance"])),
                "hs_distance_std": float(np.std(results[method][n_shots]["hs_distance"])),
            }
            if results[method][n_shots]["time"]:
                results_json[method][str(n_shots)]["time_mean"] = float(
                    np.mean(results[method][n_shots]["time"])
                )

    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Results saved to: {results_path}")

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
