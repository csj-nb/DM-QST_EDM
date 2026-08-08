"""
Training loop for the QST-EDM model.

Supports multiple model types:
    - DDPM (Denoising Diffusion Probabilistic Models)
    - EDM (Elucidating the Design Space of Diffusion Models)
    - VDM (Variational Diffusion Models)
    - FlowMatching
    - QST-MAD

Features:
    - EMA (Exponential Moving Average) of model weights
    - Cosine learning rate schedule with warmup
    - Periodic validation and checkpointing
    - Gradient clipping
    - TensorBoard logging (optional)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
import os
import time
from typing import Dict, Any, Optional, Union
import numpy as np
from tqdm import tqdm

from ..models.edm import EDM
from ..representation.cholesky import cholesky_to_dm


class EMA:
    """Exponential Moving Average for model parameters."""

    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """Initialize shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """Update shadow parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                new_average = (
                    self.decay * self.shadow[name] + (1.0 - self.decay) * param.data
                )
                self.shadow[name] = new_average

    def apply_shadow(self):
        """Apply shadow parameters to model (for evaluation)."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        """Restore original parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """
    Trainer for diffusion-model-based quantum state tomography.

    Supports DDPM, EDM, VDM, FlowMatching, and QST-MAD models.
    """

    def __init__(
        self,
        model: Union[EDM, nn.Module],
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        output_dir: str = "./outputs",
    ):
        """
        Args:
            model: Diffusion model (EDM, DDPM, VDM, FlowMatching, etc.).
            config: Full configuration dict.
            device: Torch device.
            output_dir: Directory for checkpoints and logs.
        """
        self.model = model
        self.config = config
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.output_dir = output_dir
        self.train_cfg = config["training"]

        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "checkpoints"), exist_ok=True)

        # Move model to device
        self.model = self.model.to(self.device)

        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.train_cfg["learning_rate"],
            weight_decay=1e-5,
        )

        # EMA
        self.ema = EMA(self.model, decay=self.train_cfg["ema_decay"])
        self.ema.register()

        # Learning rate scheduler (built lazily in train() once we know
        # the number of steps per epoch; see _build_scheduler).
        self.scheduler = None
        self._pending_scheduler_state = None

        # Training state
        self.epoch = 0
        self.global_step = 0
        self.best_val_loss = float("inf")
        # Optional in-training QST-quality monitor. train.py injects a function
        # fn(model, ema, device) -> mean fidelity over a few val states at a
        # fixed shot count. Loss alone is not informative: Cholesky L2 can fall
        # while fidelity stalls (the v3 failure mode: flat fidelity curves).
        self.val_fidelity_fn = None
        self.metrics_history = {
            "train_loss": [],
            "val_loss": [],
            "val_fidelity": [],
            "lr": [],
        }

        # Try to use TensorBoard (may fail due to numpy/JAX conflicts)
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(os.path.join(output_dir, "logs"))
        except Exception:
            pass

    def _build_scheduler(self, steps_per_epoch: int, epochs: int):
        """
        Build a cosine LR schedule with linear warmup, stepped once per
        training STEP (batch), not per epoch.

        ``warmup_steps`` and ``T_max`` are expressed in optimizer steps so
        that ``warmup_steps: 500`` (as written in the configs) actually
        covers the first 500 batches instead of 500 epochs. Previously the
        scheduler was stepped once per epoch, which meant a 300-epoch run
        never left the warmup phase and the cosine decay never activated.

        Args:
            steps_per_epoch: Number of optimizer steps in one epoch.
            epochs: Total number of epochs (from config/CLI).
        """
        total_steps = max(1, epochs * steps_per_epoch)
        warmup_steps = int(self.train_cfg.get("warmup_steps", 0))
        # Guard against configs where warmup >= total steps.
        warmup_steps = min(warmup_steps, max(0, total_steps - 1))

        if warmup_steps > 0:
            warmup = LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_steps,
            )
            cosine = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=1e-6,
            )
            return SequentialLR(
                self.optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_steps],
            )
        else:
            return CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps,
                eta_min=1e-6,
            )

    def _compute_val_loss(self, val_loader: DataLoader) -> float:
        """Compute average validation loss (using EMA weights)."""
        self.model.eval()
        self.ema.apply_shadow()

        # Pin the validation noise level to a representative sigma (if the
        # model supports it) so the val curve is not dominated by random
        # sigma sampling, and best-model selection is stable across epochs.
        had_fixed_sigma = hasattr(self.model, "fixed_sigma")
        if had_fixed_sigma and getattr(self.model, "fixed_sigma", None) is None:
            self.model.fixed_sigma = float(
                np.exp(getattr(self.model, "P_mean", -1.2))
            )

        try:
            total_loss = 0.0
            n_batches = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_0 = batch["x_0"].to(self.device)
                    condition = batch["condition"].to(self.device)
                    loss = self.model.training_loss(x_0, condition)
                    total_loss += loss.item()
                    n_batches += 1
            return total_loss / max(n_batches, 1)
        finally:
            if had_fixed_sigma:
                self.model.fixed_sigma = None
            self.ema.restore()
            self.model.train()

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(train_loader, desc=f"Epoch {self.epoch + 1}")
        for batch in pbar:
            x_0 = batch["x_0"].to(self.device)
            condition = batch["condition"].to(self.device)

            # Measurement-consistency loss: only pass the observed frequencies
            # to models that have enabled it (EDM with lambda_meas > 0); other
            # model families do not accept this argument.
            loss_kwargs = {}
            if getattr(self.model, "lambda_meas", 0) > 0:
                loss_kwargs["measurement_target"] = batch["measurement_target"].to(self.device)

            # Forward pass
            loss = self.model.training_loss(x_0, condition, **loss_kwargs)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if self.train_cfg["gradient_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.train_cfg["gradient_clip"],
                )

            self.optimizer.step()
            self.ema.update()
            # Step the LR scheduler once per training step (batch), not per
            # epoch. This makes warmup_steps a true step count and ensures
            # the cosine decay actually runs over the whole training.
            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            # Logging
            if self.global_step % self.train_cfg["log_every"] == 0:
                pbar.set_postfix({
                    "loss": f"{loss.item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                })
                if self.writer:
                    self.writer.add_scalar("train/loss_step", loss.item(), self.global_step)
                    self.writer.add_scalar(
                        "train/lr",
                        self.optimizer.param_groups[0]["lr"],
                        self.global_step,
                    )

            self.global_step += 1

        return total_loss / max(n_batches, 1)

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: Optional[int] = None,
    ):
        """
        Full training loop.

        Args:
            train_loader: Training data loader.
            val_loader: Validation data loader.
            epochs: Number of epochs. If None, use config value.
        """
        if epochs is None:
            epochs = self.train_cfg["epochs"]

        print(f"Training on {self.device}")
        print(f"  Epochs: {epochs}")
        print(f"  Batch size: {self.train_cfg['batch_size']}")
        print(f"  Learning rate: {self.train_cfg['learning_rate']}")
        print(f"  Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")

        # Build the LR scheduler now that we know the number of steps per
        # epoch. If we resumed from a checkpoint, restore its scheduler state.
        steps_per_epoch = max(1, len(train_loader))
        self.scheduler = self._build_scheduler(steps_per_epoch, epochs)
        if self._pending_scheduler_state is not None:
            try:
                self.scheduler.load_state_dict(self._pending_scheduler_state)
                print("  Restored LR scheduler state from checkpoint")
            except Exception as e:
                print(f"  Warning: could not restore scheduler state: {e}")
            self._pending_scheduler_state = None

        start_time = time.time()

        for epoch in range(self.epoch, epochs):
            self.epoch = epoch

            # Train
            train_loss = self.train_epoch(train_loader)
            self.metrics_history["train_loss"].append(train_loss)
            self.metrics_history["lr"].append(
                self.optimizer.param_groups[0]["lr"]
            )

            # NOTE: scheduler.step() is now called inside train_epoch, once
            # per optimizer step (batch), so warmup/cosine follow true steps.

            # Validation
            if (epoch + 1) % self.train_cfg["val_every"] == 0:
                val_loss = self._compute_val_loss(val_loader)
                self.metrics_history["val_loss"].append(val_loss)

                # QST-quality monitor (if injected by train.py). Wrapped in
                # try/except so a broken monitor never kills training.
                val_fid = None
                if self.val_fidelity_fn is not None:
                    try:
                        val_fid = self.val_fidelity_fn(
                            self.model, self.ema, self.device
                        )
                        self.metrics_history["val_fidelity"].append(val_fid)
                    except Exception as exc:  # pragma: no cover - defensive
                        print(f"  (val fidelity monitor failed: {exc})")

                elapsed = time.time() - start_time
                fid_str = f" | Val Fid: {val_fid:.4f}" if val_fid is not None else ""
                print(
                    f"Epoch {epoch + 1}/{epochs} | "
                    f"Train Loss: {train_loss:.6f} | "
                    f"Val Loss: {val_loss:.6f}"
                    f"{fid_str} | "
                    f"Time: {elapsed:.0f}s"
                )

                if self.writer:
                    self.writer.add_scalar("train/loss_epoch", train_loss, epoch)
                    self.writer.add_scalar("val/loss", val_loss, epoch)
                    if val_fid is not None:
                        self.writer.add_scalar("val/fidelity", val_fid, epoch)
                    self.writer.add_scalar(
                        "train/lr_epoch",
                        self.optimizer.param_groups[0]["lr"],
                        epoch,
                    )

                # Save best model
                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    self.save_checkpoint("best.pt")
                    print(f"  -> New best model (val_loss: {val_loss:.6f})")

            # Periodic checkpoint
            if (epoch + 1) % self.train_cfg["save_every"] == 0:
                self.save_checkpoint(f"epoch_{epoch + 1}.pt")

        # Save final model
        self.save_checkpoint("final.pt")

        elapsed = time.time() - start_time
        print(f"Training complete. Total time: {elapsed:.0f}s ({elapsed / 3600:.1f}h)")

        if self.writer:
            self.writer.close()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint (including EMA shadow and scheduler state)."""
        path = os.path.join(self.output_dir, "checkpoints", filename)
        checkpoint = {
            "epoch": self.epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "ema_shadow": {k: v.cpu() for k, v in self.ema.shadow.items()},
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler is not None else None
            ),
            "best_val_loss": self.best_val_loss,
            "metrics_history": self.metrics_history,
            "config": self.config,
        }
        # Atomically write the checkpoint (avoid corrupting it on crash).
        tmp_path = path + ".tmp"
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, path)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint (restores EMA shadow; scheduler on next train()).

        Accepts either a bare filename (resolved under the run's checkpoints/
        directory) or an absolute / already-existing path, so both
        ``--resume best.pt`` and ``--resume outputs/abl_stage0/checkpoints/best.pt``
        work.
        """
        if os.path.isabs(filename) or os.path.exists(filename):
            path = filename
        else:
            path = os.path.join(self.output_dir, "checkpoints", filename)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict") is not None:
            # The scheduler is built lazily in train(); keep the state until then.
            # Guard: if the checkpoint was trained with a different number of
            # epochs, its cosine scheduler state is meaningless (T_max mismatch,
            # LR can explode or freeze). Drop it and restart the schedule.
            ckpt_cfg = checkpoint.get("config") or {}
            ckpt_epochs = (ckpt_cfg.get("training") or {}).get("epochs")
            cur_epochs = self.train_cfg.get("epochs")
            if ckpt_epochs is not None and cur_epochs is not None and ckpt_epochs != cur_epochs:
                print(
                    f"  Warning: checkpoint epochs ({ckpt_epochs}) != current "
                    f"epochs ({cur_epochs}); scheduler state NOT restored "
                    f"(LR schedule restarts from scratch)"
                )
                self._pending_scheduler_state = None
            else:
                self._pending_scheduler_state = checkpoint["scheduler_state_dict"]
        self.epoch = checkpoint["epoch"]
        self.global_step = checkpoint["global_step"]
        self.best_val_loss = checkpoint["best_val_loss"]
        self.metrics_history = checkpoint["metrics_history"]

        # Restore EMA shadow if present, otherwise re-register from current
        # weights (previous behavior, loses EMA history).
        if "ema_shadow" in checkpoint:
            self.ema.shadow = {
                k: v.to(self.device) for k, v in checkpoint["ema_shadow"].items()
            }
        else:
            self.ema.register()
        print(f"Loaded checkpoint from {path} (epoch {self.epoch + 1})")