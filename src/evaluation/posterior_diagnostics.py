"""
Posterior diagnostics for diffusion-based QST (T1: sample diversity /
posterior-collapse check; T2: calibration of the model posterior).

Pure NumPy -- no torch, no model. The sampling script
(experiments/diagnose_posterior.py) collects K posterior samples per
(state, shot) pair and feeds them into these functions.

T1 -- diversity / collapse:
    For each state, the K samples rho_k should SPREAD when the posterior is
    wide (low shot) and CONVERGE when it is narrow (high shot). If the model
    has collapsed (mode collapse), the samples are nearly identical at ALL
    shot levels and the MMSE claim (sample average = posterior mean) is void.

T2 -- calibration (probability integral transform):
    If the model posterior p_theta(rho | m) were exact, the true state rho*
    would be indistinguishable from a fresh sample. Hence the PIT value
        u_i = (1/K) sum_k 1{ f(rho_k, rho*_i) <= f(rho_i, rho*_i) }
    (rank of the true state's fidelity among the sample fidelities) must be
    Uniform[0,1] across test states. A reliability curve over u and the
    expected calibration error (ECE) quantify deviations: u concentrated
    near 0 => the model is over-confident (samples too close to rho*);
    u concentrated near 1 => under-confident (samples too far).
"""

from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


# --------------------------------------------------------------------------
# T1: sample diversity / posterior collapse
# --------------------------------------------------------------------------

def sample_diversity_stats(
    rho_samples: np.ndarray,
    rho_true: np.ndarray,
    cholesky_vectors: np.ndarray | None = None,
) -> Dict[str, float]:
    """
    Diversity statistics of K posterior samples around their MMSE mean.

    Args:
        rho_samples: (K, d, d) complex density matrices.
        rho_true: (d, d) complex true state.
        cholesky_vectors: optional (K, d*d) Cholesky vectors of the samples,
            used for coordinate-wise variance analysis.

    Returns:
        Dict with keys:
          fid_to_true_mean/std   : fidelity of samples to the true state
          fid_spread_mean/std    : fidelity of samples to the MMSE mean
          pairwise_fid_mean      : mean pairwise fidelity among samples
                                   (1.0 => complete collapse)
          pairwise_fid_std
          trace_spread_mean      : mean trace distance sample->MMSE mean
          cholesky_var_mean      : mean over coords of sample variance
          cholesky_var_max       : max coordinate sample variance
          cholesky_var_frac_active: fraction of coords with var > 0.01*max
    """
    from src.evaluation.metrics import fidelity, trace_distance

    K = rho_samples.shape[0]
    rho_mmse = np.mean(rho_samples, axis=0)

    # Fidelity of each sample to the true state
    fid_true = np.array([fidelity(rho_samples[k], rho_true) for k in range(K)])

    # Fidelity of each sample to the MMSE mean (spread around the estimator)
    fid_spread = np.array([fidelity(rho_samples[k], rho_mmse) for k in range(K)])

    # Pairwise fidelity among samples (collapse indicator)
    pw = np.zeros((K, K))
    for a in range(K):
        for b in range(a + 1, K):
            f = fidelity(rho_samples[a], rho_samples[b])
            pw[a, b] = pw[b, a] = f
    pw_diag = pw[np.triu_indices(K, k=1)]

    # Trace distance to the MMSE mean
    trace_spread = np.array([
        trace_distance(rho_samples[k], rho_mmse) for k in range(K)
    ])

    stats = {
        "K": int(K),
        "fid_to_true_mean": float(fid_true.mean()),
        "fid_to_true_std": float(fid_true.std()),
        "fid_spread_mean": float(fid_spread.mean()),
        "fid_spread_std": float(fid_spread.std()),
        "pairwise_fid_mean": float(pw_diag.mean()) if K > 1 else 1.0,
        "pairwise_fid_std": float(pw_diag.std()) if K > 1 else 0.0,
        "trace_spread_mean": float(trace_spread.mean()),
        "trace_spread_std": float(trace_spread.std()),
    }

    if cholesky_vectors is not None:
        # Coordinate-wise variance of the Cholesky vectors (which coordinates
        # of the representation are still being explored by the sampler).
        var = np.var(cholesky_vectors, axis=0)          # (d*d,)
        vmax = float(var.max()) if var.size else 0.0
        stats["cholesky_var_mean"] = float(var.mean())
        stats["cholesky_var_max"] = vmax
        stats["cholesky_var_frac_active"] = float(
            np.mean(var > 0.01 * vmax) if vmax > 0 else 0.0
        )
        stats["cholesky_var_profile"] = (var / vmax if vmax > 0
                                         else np.zeros_like(var)).tolist()

    return stats


def aggregate_diversity(
    per_state_stats: Sequence[Dict[str, float]],
) -> Dict[str, float]:
    """
    Aggregate per-state diversity stats into (mean, std) across states.
    Scalar keys are mean/std-averaged; profile keys are element-wise averaged.
    """
    keys = [k for k in per_state_stats[0] if k not in ("K", "cholesky_var_profile")]
    out: Dict[str, float] = {}
    for k in keys:
        vals = np.array([s[k] for s in per_state_stats])
        out[f"{k}_mean"] = float(vals.mean())
        out[f"{k}_std"] = float(vals.std())
    # Profile (coordinate-wise) averaged element-wise
    if "cholesky_var_profile" in per_state_stats[0]:
        prof = np.mean(
            [np.asarray(s["cholesky_var_profile"]) for s in per_state_stats], axis=0
        )
        out["cholesky_var_profile_mean"] = prof.tolist()
    return out


# --------------------------------------------------------------------------
# T2: posterior calibration via probability integral transform
# --------------------------------------------------------------------------

def collect_pit(f_true: float, f_samples: Sequence[float]) -> float:
    """
    PIT value for one state: the rank of the TRUE state's score among the
    K sample scores.

    The score must be measured against a common reference. We use fidelity
    to the MMSE mean:  f_true = F(rho*, rho_mmse),  f_k = F(rho_k, rho_mmse).
    Then u = (1/K) sum_k 1{f_k <= f_true} is the fraction of samples at least
    as close to the MMSE mean as the true state is.

    Under an exact posterior, rho* is indistinguishable from a fresh sample,
    so u ~ Uniform[0, 1] across test states. u concentrated near 0 => the
    model is over-confident (samples hug the estimator closer than rho*
    does); u concentrated near 1 => under-confident.
    """
    f_true = float(f_true)
    f_s = np.asarray(f_samples, dtype=np.float64)
    u = float(np.mean(f_s <= f_true))
    # Edge handling: avoid exact 0/1 (finite-sample artifacts)
    K = f_s.size
    return max(min(u, 1.0 - 1.0 / (2 * K)), 1.0 / (2 * K))


def pit_calibration(
    pit_values: Sequence[float],
    alpha_grid: Sequence[float] | None = None,
) -> Dict[str, object]:
    """
    Calibration curve and ECE from PIT values.

    Claimed coverage at level alpha: the model claims the true state ranks at
    or above the alpha-quantile of its posterior with probability (1 - alpha).
    Under perfect calibration, P(u >= alpha) = 1 - alpha.

    Args:
        pit_values: PIT values across all (state, shot) pairs.
        alpha_grid: grid of alpha in [0,1] (default linspace 0.05..0.95, 19 pts)

    Returns:
        Dict with "alpha", "coverage" (empirical P(u >= alpha)),
        "ece" (mean |coverage - (1-alpha)|), and basic PIT statistics
        (mean, std, KS-like max deviation).
    """
    u = np.asarray(pit_values, dtype=np.float64)
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.95, 19)
    else:
        alpha_grid = np.asarray(alpha_grid, dtype=np.float64)

    # Empirical CDF of u, evaluated on the grid
    cdf = np.array([np.mean(u <= a) for a in alpha_grid])
    coverage = 1.0 - cdf                      # empirical P(u >= alpha)
    claimed = 1.0 - alpha_grid
    ece = float(np.mean(np.abs(coverage - claimed)))

    # Basic PIT diagnostics: should be Uniform(0,1) under calibration
    pit_mean = float(u.mean())
    pit_std = float(u.std())
    # Max deviation of the empirical CDF from the identity (KS-like)
    grid_full = np.sort(np.concatenate([alpha_grid, [0.0, 1.0]]))
    ks = float(np.max(np.abs(
        np.array([np.mean(u <= g) for g in grid_full]) - grid_full
    )))

    return {
        "n_pairs": int(u.size),
        "pit_mean": pit_mean,
        "pit_std": pit_std,
        "ks_max_dev": ks,
        "alpha": alpha_grid.tolist(),
        "claimed_coverage": claimed.tolist(),
        "empirical_coverage": coverage.tolist(),
        "ece": ece,
    }


def cloud_zscore_diagnostics(
    f_true_all: Sequence[float],
    f_samples_all: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """
    Sample-cloud drift diagnostic (absolute calibration, complement of the
    shape-only PIT): how anomalous is the TRUE state's score within the
    distribution of sample scores?

    Scores are fidelity to the MMSE mean:  f_true = F(rho*, rho_mmse),
    f_k = F(rho_k, rho_mmse). For an honest posterior, the true state sits
    like a typical sample, so
        z = (f_true - mean(f_k)) / (std(f_k) + eps)
    is centered near 0 across test states. z >> 0 means the true state is
    anomalously CLOSE to the estimator while the samples wander far -- a
    sign that the sample cloud is systematically off (e.g. ODE
    discretization bias, or over-dispersed sampling). z << 0 means the
    model is over-confident (samples hug the estimator while rho* lies
    outside the cloud).

    Returns:
        Dict with z-score mean/std, fraction |z| > 2 (outliers), and the
        mean true-state-to-estimator fidelity (the estimator's absolute
        quality, expected to rise with shot count).
    """
    zs, f_true_vals = [], []
    for ft, fs in zip(f_true_all, f_samples_all):
        fs = np.asarray(fs, dtype=np.float64)
        if fs.size < 2:
            continue
        mu, sd = fs.mean(), fs.std()
        # Floor the standard deviation (fidelity lives in [0,1]; a sample
        # cloud with sd below 1e-4 is numerically degenerate).
        sd = max(sd, 1e-4)
        zs.append((ft - mu) / sd)
        f_true_vals.append(ft)
    zs = np.asarray(zs, dtype=np.float64)
    f_true_vals = np.asarray(f_true_vals, dtype=np.float64)
    return {
        "n_pairs": int(zs.size),
        "z_mean": float(zs.mean()),
        "z_std": float(zs.std()),
        "frac_outlier": float(np.mean(np.abs(zs) > 2.0)),
        "fid_true_to_mmse_mean": float(f_true_vals.mean()),
        "fid_true_to_mmse_std": float(f_true_vals.std()),
    }
