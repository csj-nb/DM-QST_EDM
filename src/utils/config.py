"""
Configuration management using YAML files.

Provides default values and hierarchical config loading, so experiments
only need to specify the parameters they want to override.
"""

import yaml
import os
from typing import Dict, Any, Optional


# Default configuration values
DEFAULT_CONFIG = {
    "n_qubits": 2,
    "diffusion_timesteps": 1000,
    "beta_schedule": "cosine",
    "beta_start": 1.0e-4,
    "beta_end": 0.02,
    "model": {
        "dim": None,
        "hidden_dim": 256,
        "dim_mults": [1, 2, 4],
        "cond_dim": 128,
        "cond_dropout_prob": 0.1,
        "time_emb_dim": 256,
        "attn_resolutions": [8],
        "num_res_blocks": 2,
    },
    "training": {
        "batch_size": 128,
        "learning_rate": 2.0e-4,
        "lr_scheduler": "cosine",
        "warmup_steps": 500,
        "epochs": 300,
        "ema_decay": 0.9999,
        "gradient_clip": 1.0,
        "save_every": 50,
        "val_every": 10,
        "log_every": 100,
    },
    "data": {
        "n_train_states": 50000,
        "n_val_states": 5000,
        "n_test_states": 5000,
        "n_measurement_shots": 10000,
        "state_types": {
            "pure_haar": 0.40,
            "mixed_hs": 0.30,
            "mixed_ginibre": 0.10,
            "thermal": 0.10,
            "product": 0.10,
        },
        "regularization_eps": 1.0e-6,
        "seed": 42,
    },
    "evaluation": {
        "n_samples_per_state": 1,
        "ddim_steps": 100,
        "mle_max_iter": 5000,
        "mle_regularization": 0.01,
        "shot_levels": [100, 500, 1000, 5000, 10000, 50000],
        "n_repeats": 10,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base dict."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Load configuration from a YAML file, falling back to defaults.

    Args:
        config_path: Path to YAML config file. If None, use defaults only.

    Returns:
        Configuration dictionary.
    """
    config = DEFAULT_CONFIG.copy()

    if config_path is not None and os.path.exists(config_path):
        with open(config_path, "r") as f:
            yaml_config = yaml.safe_load(f)
        if yaml_config:
            config = _deep_merge(config, yaml_config)

    # Auto-compute derived values
    if config["model"]["dim"] is None:
        d = 2 ** config["n_qubits"]
        config["model"]["dim"] = d * d

    config["_derived"] = {
        "hilbert_dim": 2 ** config["n_qubits"],
        "cholesky_dim": (2 ** config["n_qubits"]) ** 2,
        "cond_input_dim": 6 ** config["n_qubits"],
    }

    return config


def save_config(config: Dict[str, Any], path: str):
    """Save configuration to a YAML file (excluding derived values)."""
    # Remove derived values before saving
    clean = {k: v for k, v in config.items() if not k.startswith("_")}
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(clean, f, default_flow_style=False, sort_keys=False)
