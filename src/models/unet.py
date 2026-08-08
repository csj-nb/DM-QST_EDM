"""
UNet denoising network for Cholesky-space diffusion.

The network operates on the Cholesky matrix representation reshaped as a
2D "image" of shape (2, d, d), where channel 0 = real(L) and channel 1 = imag(L).

For small dimensions (d=2 for 1 qubit), downsampling is limited to preserve
spatial information.

Architecture:
    - 2D ResNet blocks with GroupNorm and SiLU activations
    - FiLM conditioning from measurement data (scale + shift per block)
    - Self-attention at specified resolutions
    - Time embedding via sinusoidal encoding + MLP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Optional
import math
import numpy as np

from .conditioning import TimeEmbedding, ConditioningNetwork


def _is_power_of_two(x: int) -> bool:
    return x > 0 and (x & (x - 1)) == 0


def _num_groups(channels: int, max_groups: int = 32) -> int:
    """Find the largest divisor of channels that is <= max_groups."""
    for g in range(max_groups, 0, -1):
        if channels % g == 0:
            return g
    return 1


class ResNetBlock(nn.Module):
    """
    Residual block with GroupNorm, SiLU, and optional FiLM conditioning.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        use_film: bool = True,
        cond_channels: Optional[int] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.use_film = use_film

        self.norm1 = nn.GroupNorm(_num_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_num_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Time embedding projection
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_channels),
        )

        # FiLM conditioning projection
        if use_film and cond_channels is not None:
            self.cond_proj = nn.Sequential(
                nn.SiLU(),
                nn.Linear(cond_channels, 2 * out_channels),
            )

        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        time_emb: torch.Tensor,
        cond_emb: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # First conv block
        h = self.norm1(x)
        h = F.silu(h)
        h = self.conv1(h)

        # Add time embedding
        h = h + self.time_proj(time_emb).unsqueeze(-1).unsqueeze(-1)

        # Second conv block
        h = self.norm2(h)
        h = F.silu(h)

        # Apply FiLM conditioning
        if self.use_film and cond_emb is not None:
            film_params = self.cond_proj(cond_emb)  # (B, 2*out_ch)
            scale, shift = film_params.chunk(2, dim=1)
            h = h * (1.0 + scale.unsqueeze(-1).unsqueeze(-1))
            h = h + shift.unsqueeze(-1).unsqueeze(-1)

        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.skip(x)


class SelfAttention(nn.Module):
    """Multi-head self-attention with residual connection."""

    def __init__(self, channels: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, C // self.num_heads, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (B, heads, c_per_head, HW)

        # Scaled dot-product attention
        scale = (C // self.num_heads) ** -0.5
        attn = torch.einsum("bhci,bhcj->bhij", q, k) * scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum("bhij,bhcj->bhci", attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return x + out


class CholeskyUNet(nn.Module):
    """
    2D UNet for denoising Cholesky matrix representations.

    Input: noisy Cholesky vector of shape (B, d*d), reshaped internally
           to (B, 2, d, d) tensor (real and imaginary channels).
    Output: predicted noise, same shape as input.

    The architecture adapts to the spatial dimension d:
        - d=2 (1 qubit): no downsampling, just residual blocks
        - d=4 (2 qubits): one downsampling level
        - d>=8 (3+ qubits): two or more downsampling levels
    """

    def __init__(
        self,
        d: int,
        base_channels: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        in_channels: int = 2,
        time_emb_dim: int = 256,
        cond_dim: int = 128,
        attn_resolutions: Tuple[int, ...] = (8,),
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        out_dim: Optional[int] = None,
    ):
        """
        Args:
            d: Hilbert space dimension (2^n).
            base_channels: Base channel count.
            dim_mults: Channel multipliers per resolution level.
            in_channels: Input channels (2 for real+imag).
            time_emb_dim: Time embedding dimension.
            cond_dim: Conditioning embedding dimension.
            attn_resolutions: Spatial resolutions to apply self-attention.
            num_res_blocks: Number of ResNet blocks per resolution level.
            dropout: Dropout probability.
            out_dim: Optional output vector dimension (e.g. 15 for the
                isometric Bloch / fix-trace representations, which have
                d*d-1 coordinates). Default d*d keeps the classic Cholesky
                path byte-identical.
        """
        super().__init__()
        self.d = d
        self.in_channels = in_channels
        self.out_dim = out_dim if out_dim is not None else d * d
        self.out_channels = self.out_dim  # Output vector dimension

        # Determine how many resolution levels we can use
        # Don't downsample below 4x4 (or 2x2 for very small states)
        min_resolution = 2 if d >= 4 else d
        num_resolutions = 0
        curr_res = d
        while curr_res > min_resolution and curr_res % 2 == 0:
            num_resolutions += 1
            curr_res //= 2
        num_resolutions = max(1, num_resolutions)  # At least 1 level

        # Adjust dim_mults to available resolutions
        dim_mults = dim_mults[:num_resolutions]
        if len(dim_mults) < num_resolutions:
            # Pad with the last multiplier
            dim_mults = dim_mults + (dim_mults[-1],) * (num_resolutions - len(dim_mults))

        self.num_resolutions = num_resolutions
        self.dim_mults = dim_mults

        # Time embedding
        self.time_mlp = TimeEmbedding(time_emb_dim)

        # Initial convolution
        ch = base_channels
        self.init_conv = nn.Conv2d(in_channels, ch, 3, padding=1)

        # --- Down-sampling path ---
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        chs = [ch]  # Channel counts at each resolution

        for level in range(num_resolutions):
            mult = dim_mults[level]
            out_ch = base_channels * mult
            resolution = d // (2 ** level)

            # ResNet blocks at this resolution
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(
                    ResNetBlock(
                        in_channels=ch,
                        out_channels=out_ch,
                        time_emb_dim=time_emb_dim,
                        cond_channels=cond_dim,
                        dropout=dropout,
                    )
                )
                ch = out_ch
            self.down_blocks.append(blocks)

            # Self-attention
            if resolution in attn_resolutions:
                blocks.append(SelfAttention(ch))

            chs.append(ch)

            # Downsample (except at last level)
            if level < num_resolutions - 1:
                self.downsamples.append(
                    nn.Conv2d(ch, ch, 3, stride=2, padding=1)
                )
            else:
                self.downsamples.append(nn.Identity())

        # --- Middle blocks ---
        mid_ch = ch
        self.mid_blocks = nn.ModuleList([
            ResNetBlock(mid_ch, mid_ch, time_emb_dim, cond_channels=cond_dim, dropout=dropout),
            SelfAttention(mid_ch) if d >= 4 else nn.Identity(),
            ResNetBlock(mid_ch, mid_ch, time_emb_dim, cond_channels=cond_dim, dropout=dropout),
        ])

        # --- Up-sampling path ---
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()

        for level in reversed(range(num_resolutions)):
            mult = dim_mults[level]
            out_ch = base_channels * mult

            # ResNet blocks (input has skip connection, so in_ch = ch + skip_ch)
            blocks = nn.ModuleList()
            skip_ch = chs[level + 1]
            for i in range(num_res_blocks):
                in_ch = ch + skip_ch if i == 0 else ch
                blocks.append(
                    ResNetBlock(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        time_emb_dim=time_emb_dim,
                        cond_channels=cond_dim,
                        dropout=dropout,
                    )
                )
                ch = out_ch
            self.up_blocks.append(blocks)
            chs.append(ch)

            # Upsample (insert at beginning since we iterate reversed)
            if level > 0:
                self.upsamples.insert(0,
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
                        nn.Conv2d(ch, ch, 3, padding=1),
                    )
                )
            else:
                self.upsamples.insert(0, nn.Identity())

        # --- Output ---
        self.out_norm = nn.GroupNorm(_num_groups(ch), ch)
        self.out_conv = nn.Conv2d(ch, in_channels, 3, padding=1)
        # Optional dimension projection: d*d -> out_dim (used by isometric
        # representations with d*d-1 coordinates; Identity for the classic
        # d*d Cholesky path so it stays byte-identical).
        self.out_proj = nn.Linear(d * d, self.out_dim) if self.out_dim != d * d else nn.Identity()

    @staticmethod
    def vector_to_matrix(x: torch.Tensor, d: int) -> torch.Tensor:
        """
        Convert Cholesky vector (B, d*d) to matrix representation (B, 2, d, d).

        The layout matches the Cholesky vector encoding:
            - First d values: real diagonal
            - Remaining: interleaved (re, im) of strictly lower-triangular entries,
              column-major order.
        """
        B = x.shape[0]
        device = x.device

        # Build real and imaginary matrices
        L_real = torch.zeros(B, d, d, device=device)
        L_imag = torch.zeros(B, d, d, device=device)

        # Diagonal (real only)
        L_real[:, torch.arange(d), torch.arange(d)] = x[:, :d]

        # Strictly lower triangular entries
        idx = d
        for col in range(d):
            for row in range(col + 1, d):
                L_real[:, row, col] = x[:, idx]
                L_imag[:, row, col] = x[:, idx + 1]
                idx += 2

        return torch.stack([L_real, L_imag], dim=1)  # (B, 2, d, d)

    @staticmethod
    def matrix_to_vector(m: torch.Tensor) -> torch.Tensor:
        """
        Convert matrix representation (B, 2, d, d) back to Cholesky vector (B, d*d).

        Inverse of vector_to_matrix.
        """
        B = m.shape[0]
        d = m.shape[2]
        device = m.device

        L_real = m[:, 0]  # (B, d, d)
        L_imag = m[:, 1]  # (B, d, d)

        parts = []
        # Diagonal (real only)
        parts.append(L_real[:, torch.arange(d), torch.arange(d)])  # (B, d)

        # Strictly lower triangular (interleaved real, imag)
        for col in range(d):
            for row in range(col + 1, d):
                parts.append(L_real[:, row, col].unsqueeze(1))  # (B, 1)
                parts.append(L_imag[:, row, col].unsqueeze(1))  # (B, 1)

        return torch.cat(parts, dim=1)  # (B, d*d)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: Noisy Cholesky vector of shape (B, d*d).
            t: Time steps of shape (B,).
            cond_emb: Conditioning embedding of shape (B, cond_dim).

        Returns:
            Predicted noise, same shape as x: (B, d*d).
        """
        # Convert vector to matrix (pad to d*d if the representation has
        # fewer coordinates, e.g. 15-dim isometric Bloch / fix-trace)
        if x.shape[-1] < self.d * self.d:
            x = F.pad(x, (0, self.d * self.d - x.shape[-1]))
        h = self.vector_to_matrix(x, self.d)  # (B, 2, d, d)

        # Time embedding
        time_emb = self.time_mlp(t)  # (B, time_emb_dim)

        # Initial conv
        h = self.init_conv(h)

        # Store skip connections
        skips = []

        # Down-sampling path
        for level in range(self.num_resolutions):
            for block in self.down_blocks[level]:
                if isinstance(block, SelfAttention):
                    h = block(h)
                else:
                    h = block(h, time_emb, cond_emb)
            skips.append(h)
            if level < self.num_resolutions - 1:
                h = self.downsamples[level](h)

        # Middle
        for block in self.mid_blocks:
            if isinstance(block, SelfAttention):
                h = block(h)
            elif isinstance(block, nn.Identity):
                pass
            else:
                h = block(h, time_emb, cond_emb)

        # Up-sampling path
        for level in range(self.num_resolutions):
            # Upsample
            if level > 0:
                h = self.upsamples[level](h)

            # Concatenate skip
            skip = skips[-(level + 1)]
            h = torch.cat([h, skip], dim=1)

            for i, block in enumerate(self.up_blocks[level]):
                h = block(h, time_emb, cond_emb)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)  # (B, 2, d, d)

        # Convert back to vector
        out = self.matrix_to_vector(h)  # (B, d*d)
        if self.out_dim != self.d * self.d:
            out = self.out_proj(out)    # (B, out_dim)

        return out
