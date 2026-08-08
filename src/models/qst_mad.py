"""
QST-MAD: Measurement-Aware Diffusion for Quantum State Tomography

A diffusion model natively designed for QST, combining EDM preconditioning
principles with QST-specific physical structure:

1. Measurement-Adaptive Noise Schedule: sigma_max = c/sqrt(n_shots)
2. Score Decomposition: score = prior (NN) + likelihood (analytical Born rule)
3. Pauli-Structured Encoder: Exploits tensor product structure
4. Cross-Attention Conditioning: More expressive than FiLM
5. Physics-Informed Loss: denoising + measurement consistency + purity
6. Shot-Adaptive Guidance: lambda proportional to 1/sqrt(n_shots)

Usage:
    model = QST_MAD(d=4, cond_input_dim=36)
    loss = model.training_loss(x_0, measurement, n_shots)
    x_pred = model.sample(measurement, n_shots, n_steps=35)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from .unet import CholeskyUNet, ResNetBlock, SelfAttention, _num_groups
from .conditioning import TimeEmbedding
from .vdm import _vector_to_dm_pytorch, _purity


# ============================================================================
# 1. MAD Preconditioning (Shot-Adaptive)
# ============================================================================

class MADPreconditioner(nn.Module):
    """
    EDM-style network preconditioning with shot-adaptive noise calibration.

    D_theta(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x)

    Key addition: sigma_max is calibrated to shot count via sigma_max = c / sqrt(n_shots)
    """

    def __init__(
        self,
        cholesky_dim: int,
        d: int,
        sigma_data_diag: float = 0.3,
        sigma_data_off: float = 0.2,
        sigma_max_base: float = 0.8,
        use_preconditioning: bool = True,
    ):
        super().__init__()
        self.cholesky_dim = cholesky_dim
        self.d = d
        self.sigma_data_diag = sigma_data_diag
        self.sigma_data_off = sigma_data_off
        self.sigma_max_base = sigma_max_base
        self.use_preconditioning = use_preconditioning
        self.sigma_data_global = np.sqrt(
            (d * sigma_data_diag**2 + (d*d - d) * sigma_data_off**2) / (d*d)
        )

    def get_sigma_max(self, n_shots: torch.Tensor) -> torch.Tensor:
        """
        Compute shot-adaptive sigma_max.

        sigma_max = sigma_max_base * sqrt(n_shots_ref / n_shots)
        where n_shots_ref = 10000 (reference shot count).
        """
        n_shots_ref = 10000.0
        return self.sigma_max_base * torch.sqrt(n_shots_ref / n_shots.float())

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        net: nn.Module,
        t_int: torch.Tensor,
        cond_emb: torch.Tensor,
        cross_attn_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        B = x.shape[0]
        D = self.cholesky_dim
        device = x.device

        if sigma.dim() == 0:
            sigma = sigma.expand(B)
        sigma = sigma.view(B, 1)

        if self.use_preconditioning:
            sigma_diag = torch.full((B, self.d), self.sigma_data_diag, device=device)
            sigma_off = torch.full((B, D - self.d), self.sigma_data_off, device=device)
            sigma_data = torch.cat([sigma_diag, sigma_off], dim=1)

            c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)
            c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)
            c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)
        else:
            c_skip = torch.zeros(B, D, device=device)
            c_out = torch.ones(B, D, device=device)
            c_in = torch.ones(B, D, device=device)

        # Network input
        x_in = c_in * x
        net_out = net(x_in, t_int, cond_emb, cross_attn_kv)

        # Combine
        return c_skip * x + c_out * net_out

# ============================================================================
# 2. Pauli-Structured Measurement Encoder
# ============================================================================

class PauliMeasurementEncoder(nn.Module):
    """
    Measurement encoder that exploits the tensor product structure of Pauli measurements.

    For n qubits:
        - 3^n measurement bases (each qubit in X, Y, or Z)
        - 2^n outcomes per basis
        - Total: 6^n measurement outcomes

    The encoder:
        1. Reshapes 6^n vector into (3^n, 2^n)
        2. Processes each basis with a shared MLP
        3. Aggregates across bases with self-attention
        4. Produces conditioning embedding + cross-attention keys/values
    """

    def __init__(
        self,
        n_qubits: int,
        cond_dim: int = 128,
        hidden_dim: int = 256,
        cond_dropout_prob: float = 0.1,
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.n_bases = 3 ** n_qubits
        self.n_outcomes = 2 ** n_qubits
        self.cond_dim = cond_dim
        self.cond_dropout_prob = cond_dropout_prob

        # Shared MLP for each measurement basis
        self.basis_encoder = nn.Sequential(
            nn.Linear(self.n_outcomes, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        # Shot count encoder
        self.shot_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 4),
            nn.SiLU(),
        )

        # Cross-basis self-attention
        self.cross_basis_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=4,
            batch_first=True,
        )

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 4, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim),
        )

        # Cross-attention key/value generators
        self.cross_k = nn.Linear(hidden_dim, cond_dim)
        self.cross_v = nn.Linear(hidden_dim, cond_dim)

    def forward(
        self,
        measurement: torch.Tensor,
        n_shots: torch.Tensor,
        force_dropout: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        B = measurement.shape[0]
        device = measurement.device

        # Dropout for classifier-free guidance
        if self.training and self.cond_dropout_prob > 0:
            mask = torch.rand(B, 1, device=device) > self.cond_dropout_prob
            measurement = measurement * mask.float()
        if force_dropout:
            measurement = torch.zeros_like(measurement)

        # Reshape into (B, n_bases, n_outcomes)
        x = measurement.view(B, self.n_bases, self.n_outcomes)

        # Encode each basis with shared MLP: (B, n_bases, hidden_dim)
        basis_encoded = self.basis_encoder(x)

        # Cross-basis self-attention
        basis_attn, _ = self.cross_basis_attn(
            basis_encoded, basis_encoded, basis_encoded
        )
        basis_attn = basis_attn + basis_encoded  # residual

        # Global pooling across bases
        basis_pooled = basis_attn.mean(dim=1)  # (B, hidden_dim)

        # Encode shot count
        shot_emb = self.shot_encoder(n_shots.float().unsqueeze(-1))  # (B, hidden_dim//4)

        # Combine for conditioning embedding
        cond_emb = self.output_proj(
            torch.cat([basis_pooled, shot_emb], dim=-1)
        )  # (B, cond_dim)

        # Cross-attention keys/values from per-basis encodings
        cross_k = self.cross_k(basis_attn)  # (B, n_bases, cond_dim)
        cross_v = self.cross_v(basis_attn)  # (B, n_bases, cond_dim)

        return cond_emb, shot_emb, (cross_k, cross_v)

# ============================================================================
# 3. Cross-Attention Module for UNet
# ============================================================================

class CrossAttention(nn.Module):
    """
    Multi-head cross-attention for conditioning the UNet on measurement data.
    """

    def __init__(self, channels: int, cond_dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(_num_groups(channels), channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k_proj = nn.Linear(cond_dim, channels)
        self.v_proj = nn.Linear(cond_dim, channels)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = (channels // num_heads) ** -0.5

    def forward(
        self,
        x: torch.Tensor,
        cross_attn_kv: Tuple[torch.Tensor, torch.Tensor],
    ) -> torch.Tensor:
        B, C, H, W = x.shape
        residual = x

        # Normalize
        h = self.norm(x)

        # Compute Q from x: (B, heads, C//H, HW)
        q = self.q(h).reshape(B, self.num_heads, C // self.num_heads, H * W)

        # Compute K, V from measurement encoding
        k, v = cross_attn_kv  # (B, n_bases, cond_dim)
        K = self.k_proj(k).reshape(B, self.num_heads, C // self.num_heads, -1)
        V = self.v_proj(v).reshape(B, self.num_heads, C // self.num_heads, -1)

        # Attention: (B, heads, HW, n_bases)
        attn = torch.einsum('bhdi,bhdj->bhij', q, K) * self.scale
        attn = F.softmax(attn, dim=-1)
        out = torch.einsum('bhij,bhdj->bhdi', attn, V)
        out = out.reshape(B, C, H, W)
        out = self.proj(out)

        return residual + out


# ============================================================================
# 4. CholeskyUNet with Cross-Attention
# ============================================================================

class CholeskyUNetWithCrossAttn(CholeskyUNet):
    """
    Extended CholeskyUNet that supports cross-attention conditioning.
    Inherits from CholeskyUNet and adds cross-attention at the middle resolution.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Determine middle channel count
        if len(self.mid_blocks) > 0 and hasattr(self.mid_blocks[0], 'conv1'):
            mid_ch = self.mid_blocks[0].conv1.out_channels
        else:
            mid_ch = self.base_channels * self.dim_mults[-1] if hasattr(self, 'dim_mults') else 64
        # Cross-attention K/V must match the CONDITIONING embedding dim
        # (cond_dim), not the time-embedding dim. Fall back to the time-MLP
        # output only when cond_dim is not passed (legacy MAD path).
        attn_cond_dim = kwargs.get("cond_dim", self.time_mlp.mlp[-1].out_features)
        self.cross_attn = CrossAttention(mid_ch, attn_cond_dim)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        cross_attn_kv: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        # Convert vector to matrix
        h = self.vector_to_matrix(x, self.d)

        # Time embedding
        time_emb = self.time_mlp(t)

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

        # Middle with cross-attention
        for block in self.mid_blocks:
            if isinstance(block, SelfAttention):
                h = block(h)
            elif isinstance(block, nn.Identity):
                pass
            else:
                h = block(h, time_emb, cond_emb)

        # Apply cross-attention to measurement encoding
        if cross_attn_kv is not None:
            h = self.cross_attn(h, cross_attn_kv)

        # Up-sampling path
        for level in range(self.num_resolutions):
            if level > 0:
                h = self.upsamples[level](h)
            skip = skips[-(level + 1)]
            h = torch.cat([h, skip], dim=1)
            for block in self.up_blocks[level]:
                h = block(h, time_emb, cond_emb)

        # Output
        h = self.out_norm(h)
        h = F.silu(h)
        h = self.out_conv(h)

        return self.matrix_to_vector(h)

# ============================================================================
# 5. Analytical QST Likelihood Module
# ============================================================================

class QSTLikelihood(nn.Module):
    """
    Computes the analytical likelihood gradient for QST.

    The likelihood is: P(data|rho) = prod_k Tr(rho * P_k)^(n_k)
    The log-likelihood gradient w.r.t. rho is:
        nabla_rho log P = sum_k (n_k / Tr(rho * P_k)) * P_k

    This is converted to Cholesky space via autograd.
    """

    def __init__(self, n_qubits: int):
        super().__init__()
        self.n_qubits = n_qubits
        self.d = 2 ** n_qubits
        self.n_bases = 3 ** n_qubits
        self.n_outcomes = 2 ** n_qubits

        # Precompute Pauli projectors as buffers
        self._register_projectors()

    def _register_projectors(self):
        """Precompute and register all Pauli measurement projectors."""
        from itertools import product as cartesian_product
        bases = list(cartesian_product(["X", "Y", "Z"], repeat=self.n_qubits))
        outcomes = list(cartesian_product([1, -1], repeat=self.n_qubits))

        projectors = []
        for basis in bases:
            for outcome in outcomes:
                P = self._build_projector(basis, outcome)
                projectors.append(torch.from_numpy(P.flatten()).to(torch.complex64))

        # (n_bases * n_outcomes, d*d)
        self.register_buffer("projectors", torch.stack(projectors))

    def _build_projector(self, basis, outcome):
        """Build tensor-product projector for a given basis and outcome."""
        from ..data.measurements import get_measurement_projector
        return get_measurement_projector(basis, outcome).astype(np.complex64)

    def log_likelihood(self, rho: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Compute log-likelihood log P(data|rho).

        Args:
            rho: (B, d, d) density matrices
            counts: (B, n_bases * n_outcomes) measurement counts

        Returns:
            log_lik: (B,) log-likelihood values
        """
        B = rho.shape[0]
        d = self.d
        device = rho.device

        # Flatten rho: (B, d*d)
        rho_flat = rho.reshape(B, -1)

        # Compute Tr(rho * P_k) for all k
        projs = self.projectors.to(device)  # (K, d*d)
        projs_matrix = projs.reshape(-1, d, d)
        traces = torch.real(
            torch.einsum("bij,kji->bk", rho_flat.reshape(B, d, d), projs_matrix)
        )  # (B, K)
        traces = torch.clamp(traces, min=1e-15)

        # log-likelihood
        log_lik = torch.sum(counts * torch.log(traces), dim=-1)  # (B,)
        return log_lik

    def gradient(self, rho: torch.Tensor, counts: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient of log-likelihood w.r.t. rho.

        nabla_rho log P = sum_k (n_k / Tr(rho * P_k)) * P_k

        Args:
            rho: (B, d, d) density matrices
            counts: (B, n_bases * n_outcomes) measurement counts

        Returns:
            grad: (B, d, d) gradient w.r.t. rho
        """
        B = rho.shape[0]
        d = self.d
        device = rho.device

        rho_flat = rho.reshape(B, -1)
        projs = self.projectors.to(device)
        projs_matrix = projs.reshape(-1, d, d)  # (K, d, d)

        # Traces: (B, K)
        traces = torch.real(
            torch.einsum("bij,kji->bk", rho_flat.reshape(B, d, d), projs_matrix)
        )
        traces = torch.clamp(traces, min=1e-15)

        # Weights: (B, K)
        weights = counts / traces

        # Gradient: sum_k w_k * P_k
        grad = torch.einsum("bk,kij->bij", weights, projs_matrix)

        # Ensure Hermitian
        grad = (grad + grad.transpose(-2, -1).conj()) / 2.0

        return grad
