"""
Conditioning network for measurement-based quantum state tomography.

Maps measurement outcome frequencies (or Fisher z-transformed values)
to a conditioning embedding that guides the diffusion denoising process.

Uses FiLM (Feature-wise Linear Modulation) conditioning: the network
produces scale and shift parameters that modulate the UNet features
at each resolution level.

Supports classifier-free guidance via random dropout of the condition
during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ConditioningNetwork(nn.Module):
    """
    MLP-based conditioning network.

    Maps the measurement vector (6^n dims) to a latent conditioning
    embedding and FiLM parameters for each UNet resolution level.

    Architecture:
        measurement -> [Linear -> SiLU -> Linear -> SiLU -> ...] -> cond_emb
        cond_emb -> [Linear] -> FiLM params per resolution level
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        cond_dim: int = 128,
        num_resolutions: int = 3,
        base_channels: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        cond_dropout_prob: float = 0.1,
    ):
        """
        Args:
            input_dim: Dimension of measurement vector (6^n).
            hidden_dim: Hidden dimension of the MLP.
            cond_dim: Output conditioning embedding dimension.
            num_resolutions: Number of UNet resolution levels.
            base_channels: Base channel count of UNet.
            dim_mults: Channel multipliers per UNet level.
            cond_dropout_prob: Probability of dropping the condition (for CFG).
        """
        super().__init__()
        self.cond_dropout_prob = cond_dropout_prob

        # Measurement encoder: measurement vector -> latent embedding
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, cond_dim),
        )

        # FiLM parameter generators: produce (scale, shift) per resolution
        self.num_resolutions = num_resolutions
        self.film_generators = nn.ModuleList()
        for i, mult in enumerate(dim_mults):
            ch = base_channels * mult
            self.film_generators.append(
                nn.Sequential(
                    nn.SiLU(),
                    nn.Linear(cond_dim, 2 * ch),
                )
            )

    def forward(
        self,
        measurement: torch.Tensor,
        force_dropout: bool = False,
    ) -> Tuple[torch.Tensor, List[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            measurement: Measurement vector of shape (B, input_dim).
            force_dropout: If True, force dropout (for unconditional generation).

        Returns:
            Tuple of:
                - cond_emb: Conditioning embedding of shape (B, cond_dim).
                - film_params: List of (scale, shift) tuples per resolution level.
                  Each tensor has shape (B, channels).
        """
        B = measurement.shape[0]

        # Dropout conditioning
        if self.training and self.cond_dropout_prob > 0:
            mask = torch.rand(B, 1, device=measurement.device) > self.cond_dropout_prob
            measurement = measurement * mask.float()
        if force_dropout:
            measurement = torch.zeros_like(measurement)

        # Encode measurement
        cond_emb = self.encoder(measurement)

        # Generate FiLM parameters
        film_params = []
        for generator in self.film_generators:
            raw = generator(cond_emb)  # (B, 2*ch)
            ch = raw.shape[-1] // 2
            scale = raw[:, :ch] + 1.0  # Initialize around 1 (identity)
            shift = raw[:, ch:]
            film_params.append((scale, shift))

        return cond_emb, film_params


class TimeEmbedding(nn.Module):
    """
    Sinusoidal time step embedding.

    Maps integer time step t to a continuous embedding vector using
    sinusoidal positional encodings, as in the Transformer and DDPM.
    """

    def __init__(self, dim: int):
        """
        Args:
            dim: Output embedding dimension.
        """
        super().__init__()
        self.dim = dim

        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: Time step tensor of shape (B,) or (B, 1), integer values.

        Returns:
            Time embedding of shape (B, dim).
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        half_dim = self.dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.float() * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return self.mlp(emb)
