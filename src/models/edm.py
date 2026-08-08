"""
QST-EDM: Elucidating the Design Space of Diffusion Models for QST.

Built on Karras et al. (NeurIPS 2022) design principles, adapted for
Cholesky-space quantum state tomography with physics-aware modifications:

1. Fixed sigma(t)=t noise schedule (no learnable schedule needed)
2. Grouped sigma_data preconditioning (diagonal vs off-diagonal)
3. EDM loss weighting: lambda(sigma) = 1/c_out(sigma)^2
4. Log-normal training noise distribution
5. Heun 2nd-order ODE solver with rho-scheduling
6. Diagonal positivity protection via softplus gate
7. Physics-informed auxiliary losses (low-rank, measurement consistency)

Key differences from DDPM/VDM:
    - Denoiser parameterization D_theta(x; sigma) → clean Cholesky vector
    - No discrete timesteps; continuous sigma in [sigma_min, sigma_max]
    - Deterministic ODE sampling (no SDE, no stochasticity)
    - Built-in preconditioning for training stability

Usage:
    model = EDM(d=4, cond_input_dim=36)
    loss = model.training_loss(x_0, measurement)
    x_0_pred = model.sample(measurement, n_steps=35)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import numpy as np
from tqdm import tqdm

from .unet import CholeskyUNet
from .conditioning import ConditioningNetwork
from src.representation.hermitian import vec_to_dm_torch as _herm_vec_to_dm_torch
from src.representation.bloch import vec_to_dm_torch as _bloch_vec_to_dm_torch
from .vdm import _vector_to_dm_pytorch, _purity


def _vec_to_dm_for_rep(x, d, rep):
    """Decode a model output vector to a density matrix in the training
    representation: "hermitian"/"bloch" build rho directly from raw params;
    anything else (cholesky) uses the Cholesky map."""
    if rep == "hermitian":
        return _herm_vec_to_dm_torch(x, d)
    if rep == "bloch":
        return _bloch_vec_to_dm_torch(x, d)
    return _vector_to_dm_pytorch(x, d)


def _fidelity_pytorch(
    rho1: torch.Tensor,
    rho2: torch.Tensor,
    eps: float = 1e-10,
) -> torch.Tensor:
    """
    Differentiable quantum fidelity F(rho1, rho2) = [Tr(sqrt(sqrt(rho1) rho2 sqrt(rho1)))]^2.

    Both inputs are batch Hermitian density matrices of shape (B, d, d) (complex).
    Uses eigendecomposition of the 4x4 (or d x d) matrix, stable and vectorized.

    Returns (B,) fidelities in [0, 1].
    """
    def _sqrtm_psd(r: torch.Tensor) -> torch.Tensor:
        # Symmetrize to kill tiny numerical anti-Hermitian residue.
        r = 0.5 * (r + r.transpose(-2, -1).conj())
        evals, evecs = torch.linalg.eigh(r)          # (B, d), (B, d, d) complex evecs
        evals = evals.clamp(min=0.0)                  # PSD projection
        sqrt_r = (evecs * evals.sqrt().unsqueeze(-2)) @ evecs.transpose(-2, -1).conj()
        return sqrt_r

    s1 = _sqrtm_psd(rho1)                             # sqrt(rho1)
    inner = s1 @ rho2 @ s1                            # sqrt(rho1) rho2 sqrt(rho1)
    inner = 0.5 * (inner + inner.transpose(-2, -1).conj())
    evals = torch.linalg.eigvalsh(inner)              # (B, d), real
    F = evals.clamp(min=0.0).sqrt().sum(dim=-1) ** 2  # (B,)
    return F.clamp(0.0, 1.0)


# ============================================================================
# EDM Preconditioning
# ============================================================================

class EDMPreconditioner(nn.Module):
    """
    EDM-style network preconditioning.

    D_theta(x; sigma) = c_skip(sigma) * x + c_out(sigma) * F_theta(c_in(sigma) * x; c_noise(sigma))

    For QST, we use GROUPED sigma_data:
        - Diagonal elements: sigma_data_diag
        - Off-diagonal elements: sigma_data_off

    This matters because Cholesky diagonal elements are positive-valued and
    have different statistics than the zero-mean off-diagonal elements.
    """

    def __init__(
        self,
        cholesky_dim: int,
        d: int,                        # Hilbert space dimension
        sigma_data_diag: float = 0.3,
        sigma_data_off: float = 0.2,
        use_preconditioning: bool = True,
        sigma_data_per_dim: Optional[np.ndarray] = None,
    ):
        """
        Args:
            cholesky_dim: Total dimension of Cholesky vector (d*d).
            d: Hilbert space dimension (2^n).
            sigma_data_diag: Empirical std of diagonal Cholesky elements.
            sigma_data_off: Empirical std of off-diagonal Cholesky elements.
            use_preconditioning: If False, use identity mapping (c_skip=0, c_out=1, c_in=1).
            sigma_data_per_dim: Optional per-coordinate sigma_data (Fix B),
                shape (cholesky_dim,). When provided, overrides the grouped
                diag/off values in the preconditioning coefficients and makes
                the loss weighting per-coordinate, matching the preconditioner
                to the anisotropic rho-loss geometry (J^T J diagonal varies
                ~9.4x across coordinates).
        """
        super().__init__()
        self.cholesky_dim = cholesky_dim
        self.d = d
        self.sigma_data_diag = sigma_data_diag
        self.sigma_data_off = sigma_data_off
        self.use_preconditioning = use_preconditioning
        self.sigma_data_per_dim: Optional[np.ndarray] = None
        if sigma_data_per_dim is not None:
            arr = np.asarray(sigma_data_per_dim, dtype=np.float32)
            if arr.ndim != 1 or arr.shape[0] != cholesky_dim:
                raise ValueError(
                    f"sigma_data_per_dim must have shape ({cholesky_dim},), got {arr.shape}"
                )
            self.sigma_data_per_dim = arr
            # Keep the grouped values consistent with the per-dim vector so
            # the two code paths agree wherever either source is read.
            self.sigma_data_diag = float(arr[:d].mean())
            self.sigma_data_off = float(arr[d:].mean())
            self.sigma_data_global = float(np.sqrt(float(np.mean(arr ** 2))))
        else:
            self.sigma_data_global = float(np.sqrt(
                (d * sigma_data_diag ** 2 + (d * d - d) * sigma_data_off ** 2) / (d * d)
            ))

    @property
    def sigma_data_vector(self) -> Optional[np.ndarray]:
        """Per-coordinate sigma_data (D,) or None in grouped mode."""
        return self.sigma_data_per_dim

    def forward(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        net: nn.Module,
        c_noise: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply EDM preconditioning and call the network.

        Args:
            x: Noisy Cholesky vector, shape (B, D).
            sigma: Noise level, shape (B,) or scalar.
            net: Denoising network taking (x, t_emb, cond_emb).
            c_noise: Continuous noise conditioning (log-space), shape (B,).
            cond_emb: Conditioning embedding.

        Returns:
            Denoised Cholesky vector, shape (B, D).
            Note: No positivity projection is applied here. The denoiser output
            is a raw Cholesky vector that may have negative diagonal elements.
            Positivity is enforced via softplus on the final sampling output.
        """
        B = x.shape[0]
        D = self.cholesky_dim
        device = x.device

        # Broadcast sigma
        if sigma.dim() == 0:
            sigma = sigma.expand(B)
        sigma = sigma.view(B, 1)  # (B, 1)

        # ----- Preconditioning coefficients -----
        if self.use_preconditioning:
            if self.sigma_data_per_dim is not None:
                # Fix B: per-coordinate sigma_data (B, D) from the vector.
                s_data = torch.from_numpy(self.sigma_data_per_dim).to(device)  # (D,)
                sigma_data = s_data.unsqueeze(0).expand(B, D)                  # (B, D)
            else:
                sigma_diag = torch.full((B, self.d), self.sigma_data_diag, device=device)
                sigma_off = torch.full((B, D - self.d), self.sigma_data_off, device=device)
                sigma_data = torch.cat([sigma_diag, sigma_off], dim=1)  # (B, D)

            # c_skip = sigma_data^2 / (sigma^2 + sigma_data^2)
            c_skip = sigma_data ** 2 / (sigma ** 2 + sigma_data ** 2)  # (B, D)

            # c_out = sigma * sigma_data / sqrt(sigma^2 + sigma_data^2)
            c_out = sigma * sigma_data / torch.sqrt(sigma ** 2 + sigma_data ** 2)  # (B, D)

            # c_in = 1 / sqrt(sigma^2 + sigma_data^2)
            c_in = 1.0 / torch.sqrt(sigma ** 2 + sigma_data ** 2)  # (B, D)
        else:
            # No preconditioning: identity mapping
            c_skip = torch.zeros(B, D, device=device)
            c_out = torch.ones(B, D, device=device)
            c_in = torch.ones(B, D, device=device)

        # c_noise (continuous) — passed through directly.
        # UNet's TimeEmbedding uses sinusoidal encoding, so it handles
        # continuous values naturally (no discretization needed).

        # ----- Network forward pass -----
        x_in = c_in * x  # (B, D)

        if hasattr(net, "cross_attn"):
            # Cross-attention conditioning: condition embedding as K/V
            # (B, 1, cond_dim); Q comes from the UNet features.
            kv = (cond_emb.unsqueeze(1), cond_emb.unsqueeze(1))
            net_out = net(x_in, c_noise, cond_emb, cross_attn_kv=kv)  # (B, D)
        else:
            net_out = net(x_in, c_noise, cond_emb)  # (B, D)

        # ----- Final preconditioned output -----
        # Note: No softplus positivity protection here.
        # Physical constraints (Hermiticity, PSD, unit trace) are guaranteed by
        # the Cholesky -> density matrix conversion (rho = L L^dag / Tr(L L^dag))
        # which is applied AFTER sampling, not inside the denoiser.
        # Keeping the ODE integration free of nonlinearities preserves
        # Heun 2nd-order accuracy (see Karras et al. Algorithm 1).
        D_theta = c_skip * x + c_out * net_out

        return D_theta


# ============================================================================
# Noise Schedule and Sampling Utilities
# ============================================================================

def edm_sigmas(
    n_steps: int,
    sigma_min: float,
    sigma_max: float,
    rho: float = 7.0,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    EDM rho-schedule for time discretization.

    sigma_i = (sigma_max^{1/rho} + i/(N-1) * (sigma_min^{1/rho} - sigma_max^{1/rho}))^rho
    """
    i = torch.arange(n_steps, device=device, dtype=torch.float32)
    sigma_min_rho = sigma_min ** (1.0 / rho)
    sigma_max_rho = sigma_max ** (1.0 / rho)
    sigmas = (sigma_max_rho + i / (n_steps - 1) * (sigma_min_rho - sigma_max_rho)) ** rho
    return sigmas


def lognormal_sigma_distribution(
    P_mean: float = -1.2,
    P_std: float = 1.2,
    sigma_min: float = 0.001,
    sigma_max: float = 80.0,
) -> callable:
    """
    Returns a function that samples sigma from log-normal distribution,
    truncated to [sigma_min, sigma_max].
    """
    def sample(batch_size: int, device: torch.device) -> torch.Tensor:
        rnd = torch.randn(batch_size, device=device)
        sigma = torch.exp(P_mean + P_std * rnd)
        sigma = torch.clamp(sigma, sigma_min, sigma_max)
        return sigma
    return sample


# ============================================================================
# QST-EDM Model
# ============================================================================

class EDM(nn.Module):
    """
    QST-EDM: EDM-adapted diffusion model for Quantum State Tomography.

    Complete design:
        - Forward: x_sigma = x_0 + sigma * epsilon
        - Preconditioning: grouped sigma_data (diag vs off-diag)
        - Loss: lambda(sigma) * f(sigma) * MSE with lambda = 1/c_out^2
          where f(sigma) is the lognormal PDF (ERDM-style reweighting)
        - Noise distribution: LogNormal(P_mean, P_std)
        - Sampling: Heun 2nd-order ODE + rho-scheduling
        - Physics: diagonal softplus (post-sampling) + low-rank auxiliary loss

    ERDM-style loss reweighting:
        Following Rühling Cachay et al. (NeurIPS 2025), we optionally
        reweight the loss by the lognormal PDF f(sigma) to focus model
        capacity on the most informative intermediate noise levels.
        This is controlled by ``use_loss_reweighting`` (default: True).
    """

    def __init__(
        self,
        d: int,
        cond_input_dim: int = 36,
        base_channels: int = 64,
        dim_mults: Tuple[int, ...] = (1, 2, 4),
        num_res_blocks: int = 2,
        cond_dim: int = 128,
        cond_dropout_prob: float = 0.1,
        # Condition injection style:
        #   "film"       - FiLM modulation in every ResBlock (default)
        #   "cross_attn" - cross-attention on the condition at the UNet middle
        #                  (stronger injection; CholeskyUNetWithCrossAttn)
        #   "both"       - FiLM + cross-attention
        cond_inject: str = "film",
        # Main-loss space: "cholesky" (default, weighted Cholesky MSE) or "rho"
        # (HS/Frobenius distance to the true state in density-matrix space).
        loss_space: str = "cholesky",
        loss_type: str = "l2",
        # EDM noise schedule
        sigma_min: float = 0.001,
        sigma_max: float = 0.8,
        sigma_data_diag: float = 0.3,
        sigma_data_off: float = 0.2,
        # Fix B: per-coordinate sigma_data (overrides grouped diag/off).
        sigma_data_per_dim: Optional[list] = None,
        # Training noise distribution
        P_mean: float = -1.2,
        P_std: float = 1.2,
        # Sampling
        rho: float = 7.0,
        # Physics loss
        lambda_rank: float = 0.0,
        lambda_rank_warmup: int = 1000,
        lambda_meas: float = 0.0,
        lambda_meas_warmup: int = 1000,
        # Measurement-consistency loss form.
        #   "l2"  -> masked-L2 (current default): mean (p_j - m_j)^2 over
        #           OBSERVED outcomes only (freq > 0).
        #   "nll" -> clipped negative log-likelihood: -sum_j m_j log clip(p_j, p_min).
        #           Statistically the exact multinomial likelihood, but the
        #           clipped floor (p_min, default 1e-3) caps the penalty on rare
        #           observed events -- at low shots those are shot noise, and an
        #           unclipped NLL overfits them. lambda_meas must be re-scaled
        #           for NLL (log scale != squared scale).
        meas_loss_type: str = "l2",
        meas_loss_p_min: float = 1e-3,
        use_preconditioning: bool = True,
        # ERDM-style loss reweighting
        use_loss_reweighting: bool = True,
        # Fidelity-aware auxiliary loss
        lambda_fid: float = 0.0,
        lambda_fid_warmup: int = 3000,
        # Representation: "cholesky" (default) or "hermitian" (direct rho params)
        representation: str = "cholesky",
        # Output vector dimension override: d*d-1 for the isometric
        # representations (bloch / hermitian_fix_trace); default d*d keeps
        # the classic Cholesky / Hermitian path byte-identical.
        out_dim: Optional[int] = None,
    ):
        """
        Args:
            d: Hilbert space dimension (2^n).
            cond_input_dim: Measurement vector dimension (6^n for Pauli).
            base_channels, dim_mults, num_res_blocks: UNet architecture.
            cond_dim: Conditioning embedding dimension.
            cond_dropout_prob: Conditioning dropout for CFG.
            loss_type: "l2", "l1", or "huber".
            sigma_min: Minimum noise level.
            sigma_max: Maximum noise level.
            sigma_data_diag: Empirical std of diagonal Cholesky elements.
            sigma_data_off: Empirical std of off-diagonal Cholesky elements.
            P_mean: Mean of log-normal noise distribution.
            P_std: Std of log-normal noise distribution.
            rho: Exponent for rho-scheduling (Karras default: 7).
            lambda_rank: Low-rank auxiliary loss weight.
            lambda_rank_warmup: Steps to linearly warm up lambda_rank.
            lambda_meas: Measurement consistency loss weight.
            lambda_meas_warmup: Steps to linearly warm up lambda_meas.
            use_loss_reweighting: If True, apply ERDM-style lognormal PDF
                reweighting to focus on intermediate noise levels.
        """
        super().__init__()
        self.d = d
        self.cholesky_dim = out_dim if out_dim is not None else d * d
        self.loss_type = loss_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.P_mean = P_mean
        self.P_std = P_std
        self.rho = rho
        self.lambda_rank = lambda_rank
        self.lambda_rank_warmup = lambda_rank_warmup
        self.lambda_meas = lambda_meas
        self.lambda_meas_warmup = lambda_meas_warmup
        self.meas_loss_type = meas_loss_type
        self.meas_loss_p_min = meas_loss_p_min
        self.use_loss_reweighting = use_loss_reweighting
        self.lambda_fid = lambda_fid
        self.lambda_fid_warmup = lambda_fid_warmup
        self.representation = representation
        self.register_buffer("global_step", torch.tensor(0))
        # Optional fixed sigma override used for stable validation loss
        # (set by the trainer during validation; None = sample from log-normal).
        self.fixed_sigma: Optional[float] = None

        # Born-rule projector cache for the measurement-consistency loss:
        # 6^n projectors {P_j} with probs_j = Tr(rho P_j) compared against the
        # observed Pauli frequencies. Built lazily (needs device), cached as a
        # (6^n, d, d) complex tensor.
        self._born_projectors: Optional[torch.Tensor] = None

        # --- Denoising network (same UNet as DDPM/VDM/FlowMatching) ---
        # Stronger condition injection (cross-attention at UNet middle) is
        # available via CholeskyUNetWithCrossAttn when cond_inject != "film".
        if cond_inject != "film":
            from .qst_mad import CholeskyUNetWithCrossAttn

            unet_cls = CholeskyUNetWithCrossAttn
        else:
            unet_cls = CholeskyUNet
        self.cond_inject = cond_inject
        self.loss_space = loss_space
        self.denoise_fn = unet_cls(
            d=d,
            base_channels=base_channels,
            dim_mults=dim_mults,
            num_res_blocks=num_res_blocks,
            in_channels=2,
            time_emb_dim=256,
            cond_dim=cond_dim,
            out_dim=self.cholesky_dim,
        )

        # --- Conditioning network ---
        self.conditioning = ConditioningNetwork(
            input_dim=cond_input_dim,
            hidden_dim=256,
            cond_dim=cond_dim,
            num_resolutions=self.denoise_fn.num_resolutions,
            base_channels=base_channels,
            dim_mults=dim_mults,
            cond_dropout_prob=cond_dropout_prob,
        )

        # --- EDM Preconditioner ---
        self.preconditioner = EDMPreconditioner(
            cholesky_dim=self.cholesky_dim,
            d=d,
            sigma_data_diag=sigma_data_diag,
            sigma_data_off=sigma_data_off,
            use_preconditioning=use_preconditioning,
            sigma_data_per_dim=sigma_data_per_dim,
        )

    # ------------------------------------------------------------------
    # Forward Process
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_0: torch.Tensor,
        sigma: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EDM forward: x_sigma = x_0 + sigma * epsilon.

        Args:
            x_0: Clean Cholesky vector (B, D).
            sigma: Noise level (B,) or scalar.
            noise: Optional pre-sampled noise.

        Returns:
            Noisy sample (B, D).
        """
        if noise is None:
            noise = torch.randn_like(x_0)
        if sigma.dim() == 0:
            sigma = sigma.expand(x_0.shape[0])
        sigma = sigma.view(-1, *([1] * (x_0.dim() - 1)))
        return x_0 + sigma * noise

    # ------------------------------------------------------------------
    # Born-rule measurement consistency (auxiliary loss support)
    # ------------------------------------------------------------------

    def _get_born_projectors(self) -> torch.Tensor:
        """Return the 6^n Pauli-measurement projectors on the model device."""
        if self._born_projectors is None:
            import itertools

            from src.data.measurements import _PAULI_BASES, get_measurement_projector

            n = int(round(np.log2(self.d)))
            proj_list = []
            for basis in itertools.product(_PAULI_BASES, repeat=n):
                for outcome in itertools.product([1, -1], repeat=n):
                    proj_list.append(get_measurement_projector(basis, outcome))
            proj = np.stack(proj_list)  # (6^n, d, d) complex128
            # Match the model's dtype: float32 weights -> complex64 projectors,
            # otherwise torch.einsum mismatches (ComplexFloat vs ComplexDouble).
            ref = next(self.parameters())
            complex_dtype = torch.complex64 if ref.dtype == torch.float32 else torch.complex128
            self._born_projectors = torch.from_numpy(proj).to(
                device=ref.device, dtype=complex_dtype
            )
        return self._born_projectors

    def _born_probs(self, rho: torch.Tensor) -> torch.Tensor:
        """Born probabilities of density matrices under all Pauli projectors.

        Args:
            rho: Density matrices of shape (B, d, d) (complex).
        Returns:
            probs: (B, 6^n) real tensor in [0, 1].
        probs[b, m] = Tr(rho[b] @ P[m]) = sum_ab rho[b, a, d] P[m, d, a].
        """
        P = self._get_born_projectors()  # (M, d, d) complex
        probs = torch.einsum("bad,mda->bm", rho, P).real
        return probs.clamp(0.0, 1.0)

    # ------------------------------------------------------------------
    # Denoiser (with preconditioning)
    # ------------------------------------------------------------------

    def denoiser(
        self,
        x: torch.Tensor,
        sigma: torch.Tensor,
        cond_emb: torch.Tensor,
    ) -> torch.Tensor:
        """
        Denoising function D_theta(x; sigma) with EDM preconditioning.

        Returns predicted clean Cholesky vector, with diagonal positivity
        applied after sampling (not during ODE integration).

        Args:
            x: Noisy Cholesky vector (B, D).
            sigma: Noise level (B,) or scalar.
            cond_emb: Conditioning embedding (B, cond_dim).

        Returns:
            Predicted clean Cholesky vector (B, D).
        """
        B = x.shape[0]
        device = x.device

        if sigma.dim() == 0:
            sigma = sigma.expand(B)

        # ---- Use continuous c_noise as time input to UNet ----
        # Standard EDM: c_noise(sigma) = 1/4 * ln(sigma / sigma_data_global)
        # This is a continuous value that the UNet's sinusoidal embedding
        # can naturally handle (no discretization needed).
        sigma_g = self.preconditioner.sigma_data_global
        c_noise = torch.log(sigma / sigma_g) / 4.0   # (B,)

        # ---- Call preconditioner with continuous c_noise ----
        return self.preconditioner(x, sigma, self.denoise_fn, c_noise, cond_emb)

    # ------------------------------------------------------------------
    # Training Loss
    # ------------------------------------------------------------------

    def training_loss(
        self,
        x_0: torch.Tensor,
        condition: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        measurement_target: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        EDM training loss with optional ERDM-style reweighting.

        L = E_{sigma ~ LogNormal} [ lambda(sigma) * f(sigma) * ||D_theta(x_0+sigma*eps, sigma) - x_0||^2 ]
        + lambda_rank * (1 - Purity)
        + lambda_meas * ||Tr(rho_pred * O) - m_obs||^2

        where:
            - lambda(sigma) = 1 / c_out(sigma)^2 (EDM preconditioning)
            - f(sigma) = lognormal PDF (ERDM reweighting, optional)

        The ERDM reweighting f(sigma) focuses model capacity on the most
        informative intermediate noise levels, where the lognormal PDF peaks
        around exp(P_mean). This is critical for low-shot QST where the
        transition from measurement-dominated to prior-dominated regimes
        determines reconstruction quality.

        Reference: Rühling Cachay et al., ERDM (NeurIPS 2025), Eq. (4)-(6).

        Args:
            x_0: Clean Cholesky vector (B, D).
            condition: Measurement vector (B, cond_input_dim).
            noise: Optional pre-sampled noise.

        Returns:
            Scalar loss.
        """
        B = x_0.shape[0]
        device = x_0.device

        # --- Sample sigma from log-normal ---
        sigma = self._sample_training_sigma(B, device)

        # --- Sample noise ---
        if noise is None:
            noise = torch.randn_like(x_0)

        # --- Forward diffusion ---
        x_sigma = self.q_sample(x_0, sigma, noise=noise)

        # --- Encode condition ---
        cond_emb, _ = self.conditioning(condition)

        # --- Denoise ---
        x_0_pred = self.denoiser(x_sigma, sigma, cond_emb)

        # --- EDM loss with weighting ---
        # lambda(sigma) = 1/c_out(sigma)^2 is automatically handled by
        # the preconditioner: training D_theta with MSE loss directly on the
        # preconditioned output gives effective lambda = 1/c_out^2 weighting
        # (see Karras et al. 2022, Section 5.2, Eq. 126)
        if self.loss_type == "l2":
            diff = (x_0_pred - x_0).pow(2).mean(dim=-1)  # (B,)
        elif self.loss_type == "l1":
            diff = (x_0_pred - x_0).abs().mean(dim=-1)  # (B,)
        elif self.loss_type == "huber":
            d = (x_0_pred - x_0).abs()
            diff = torch.where(d < 1.0, 0.5 * d.pow(2), d - 0.5).mean(dim=-1)
        else:
            diff = (x_0_pred - x_0).pow(2).mean(dim=-1)

        # Weight by inverse c_out^2 (effective lambda weighting).
        # Use per-coordinate sigma_data (Fix B) when available, else grouped.
        sigma_sq = sigma ** 2  # (B,) or (B, 1)
        if sigma_sq.dim() == 1:
            sigma_sq = sigma_sq.unsqueeze(1)  # (B, 1) for broadcasting
        if self.preconditioner.sigma_data_vector is not None:
            s_data = torch.from_numpy(self.preconditioner.sigma_data_vector).to(x_0.device)  # (D,)
            s_sq = s_data ** 2
            # Per-dimension weight: lambda_i = (sigma^2 + sigma_data_i^2) / (sigma^2 * sigma_data_i^2)
            # sigma_sq (B,1) * s_sq (D,) -> broadcast (B, D)
            weight = (sigma_sq + s_sq) / (sigma_sq * s_sq + 1e-8)   # (B, D)
        else:
            sigma_d_sq = self.preconditioner.sigma_data_diag ** 2
            sigma_o_sq = self.preconditioner.sigma_data_off ** 2

            # Per-dimension weight: lambda_i = (sigma^2 + sigma_data_i^2) / (sigma^2 * sigma_data_i^2)
            weight_diag = (sigma_sq + sigma_d_sq) / (sigma_sq * sigma_d_sq + 1e-8)   # (B, 1)
            weight_off  = (sigma_sq + sigma_o_sq) / (sigma_sq * sigma_o_sq + 1e-8)   # (B, 1)

            # Build per-sample weight vector matching Cholesky layout (d diag + (D-d) off-diag)
            B_ = x_0.shape[0]
            w_diag = weight_diag.expand(B_, self.preconditioner.d)                    # (B, d)
            w_off  = weight_off.expand(B_, self.cholesky_dim - self.preconditioner.d) # (B, D-d)
            weight = torch.cat([w_diag, w_off], dim=1)                                 # (B, D)

        # Normalize so that mean weight = 1 (loss scale independent of sigma)
        weight = weight / (weight.detach().mean() + 1e-8)

        # --- ERDM-style lognormal PDF reweighting ---
        # Focus model capacity on the most informative intermediate noise
        # levels where the lognormal PDF peaks around exp(P_mean).
        if self.use_loss_reweighting:
            f_sigma = self.lognormal_pdf(sigma, self.P_mean, self.P_std)  # (B,)
            # Normalize to mean=1 to preserve loss scale
            f_sigma = f_sigma / (f_sigma.detach().mean() + 1e-8)
            weight = weight * f_sigma.unsqueeze(1)  # (B, D) * (B, 1)

        # Apply per-dimension weighted MSE
        if self.loss_space == "rho":
            # rho-space main loss: the denoiser still predicts Cholesky vectors
            # and diffusion still runs in Cholesky coordinates, but the TARGET
            # is evaluated in density-matrix space (HS/Frobenius distance to the
            # true state). This removes the Cholesky-vs-fidelity metric mismatch:
            # two Cholesky errors of equal size can have very different rho-space
            # consequences. Gradients flow back through the differentiable
            # cholesky -> rho map.
            if self.representation in ("hermitian", "bloch"):
                rho_pred = _vec_to_dm_for_rep(x_0_pred, self.d, self.representation)
                rho_true = _vec_to_dm_for_rep(x_0, self.d, self.representation)
            else:
                rho_pred = _vector_to_dm_pytorch(x_0_pred, self.d)
                rho_true = _vector_to_dm_pytorch(x_0, self.d)
            # ||rho_pred - rho_true||_F^2 = sum_ij |Δ_ij|^2 (real)
            rho_diff = (rho_pred - rho_true).abs().pow(2).sum(dim=(-2, -1))  # (B,)
            if self.use_loss_reweighting:
                f_sigma = self.lognormal_pdf(sigma, self.P_mean, self.P_std)
                f_sigma = f_sigma / (f_sigma.detach().mean() + 1e-8)
                rho_diff = rho_diff * f_sigma
            base_loss = rho_diff.mean()
        elif self.loss_type == "l2":
            diff = (x_0_pred - x_0).pow(2)  # (B, D)
        elif self.loss_type == "l1":
            diff = (x_0_pred - x_0).abs()   # (B, D)
        elif self.loss_type == "huber":
            d = (x_0_pred - x_0).abs()
            diff = torch.where(d < 1.0, 0.5 * d.pow(2), d - 0.5)  # (B, D)
        else:
            diff = (x_0_pred - x_0).pow(2)
        if self.loss_space != "rho":
            base_loss = (weight * diff).mean()

        total_loss = base_loss

        # --- Low-rank auxiliary loss ---
        # NOTE (2026-08-02): keep lambda_rank = 0.0 in configs. Penalizing
        # (1 - purity) pushes predictions toward pure states, conflicting with
        # the training distribution (60% mixed states). Branch retained only
        # for ablation/tests; do not enable for main training runs.
        if self.lambda_rank > 0 and self.training:
            step = self.global_step.item()
            current_lambda = self.lambda_rank * min(1.0, step / max(1, self.lambda_rank_warmup))

            rho_pred = _vector_to_dm_pytorch(x_0_pred, self.d)
            pur = _purity(rho_pred)  # (B,)
            rank_loss = (1.0 - pur).mean()
            total_loss = total_loss + current_lambda * rank_loss

        # --- Fidelity-aware auxiliary loss ---
        # The Cholesky-space L2 target does not directly maximize the
        # evaluation metric (fidelity). Ramp in a differentiable fidelity
        # term so predictions are pushed toward states that score well on
        # the downstream metric, not merely low Cholesky error.
        # Note: rho_pred = L L^dag / Tr(L L^dag) is always a valid density
        # matrix even if predicted diagonal elements are negative, so the
        # fidelity is well-defined and the gradient is stable.
        if self.lambda_fid > 0 and self.training:
            step = self.global_step.item()
            current_lambda = self.lambda_fid * min(1.0, step / max(1, self.lambda_fid_warmup))
            rho_pred = _vector_to_dm_pytorch(x_0_pred, self.d)
            rho_true = _vector_to_dm_pytorch(x_0, self.d)
            fid = _fidelity_pytorch(rho_pred, rho_true)  # (B,)
            fid_loss = (1.0 - fid).mean()
            total_loss = total_loss + current_lambda * fid_loss

        # --- Measurement consistency loss (masked-L2) ---
        # The predicted state must explain the observed measurement: its
        # Born-rule probabilities Tr(rho_pred P_j) should match the observed
        # (possibly shot-resampled) frequencies. In the low-shot regime this
        # explicitly prevents the model from "cheating" by returning the
        # prior-mean state while ignoring the measurement — the same
        # information MLE uses is made explicit here during training.
        # measurement_target comes from the dataset (observed frequencies).
        #
        # Masked-L2: only the outcomes that were actually OBSERVED (freq > 0)
        # contribute to the loss. Full-L2 penalizes all 6^n outputs including
        # the freq=0 ones, which in the sparse low-shot regime (most outputs
        # unseen) forces the model to spread probability mass evenly instead
        # of fitting the observed outcomes — the exact inefficiency MLE avoids
        # by maximizing likelihood (where freq=0 terms contribute 0).
        # This is the L2 analogue of negative-log-likelihood: same scale
        # behaviour (lambda_meas=0.1 stays calibrated), likelihood-aligned.
        if self.lambda_meas > 0 and self.training and measurement_target is not None:
            step = self.global_step.item()
            current_lambda = self.lambda_meas * min(1.0, step / max(1, self.lambda_meas_warmup))
            if self.representation in ("hermitian", "bloch"):
                rho_pred = _vec_to_dm_for_rep(x_0_pred, self.d, self.representation)
            else:
                rho_pred = _vector_to_dm_pytorch(x_0_pred, self.d)
            probs_pred = self._born_probs(rho_pred)          # (B, 6^n)
            target = measurement_target.float()              # (B, 6^n)
            if self.meas_loss_type == "nll":
                # Clipped negative log-likelihood: -sum_j m_j log clip(p_j, p_min).
                # freq=0 terms contribute 0 (0*log p = 0) natively, so no mask is
                # needed. p_min caps the penalty on rare OBSERVED events, which at
                # low shot counts are mostly shot noise (unclipped NLL overfits them).
                p_clip = torch.clamp(probs_pred, self.meas_loss_p_min, 1.0)
                meas_loss = -(target * torch.log(p_clip)).sum(dim=-1).mean()
            else:
                # Masked-L2: only the outcomes that were actually OBSERVED
                # (freq > 0) contribute to the loss (likelihood-aligned).
                obs_mask = (target > 0).float()              # (B, 6^n): observed outcomes
                n_obs = obs_mask.sum()
                if n_obs > 0:
                    meas_loss = (((probs_pred - target) ** 2) * obs_mask).sum() / n_obs
                else:
                    # Degenerate: no observed outcome (shouldn't happen); fall back.
                    meas_loss = ((probs_pred - target) ** 2).mean()
            total_loss = total_loss + current_lambda * meas_loss

        # Only advance the internal step counter during training. During
        # validation (trainer calls training_loss with model.eval()) we must
        # not let the warmup counters be advanced by validation batches.
        if self.training:
            self.global_step += 1
        return total_loss

    def _sample_training_sigma(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample sigma from log-normal distribution."""
        if self.fixed_sigma is not None:
            # Validation mode: pin sigma to a representative value so the
            # validation loss is not dominated by random sigma sampling.
            return torch.full((batch_size,), self.fixed_sigma, device=device)
        rnd = torch.randn(batch_size, device=device)
        sigma = torch.exp(self.P_mean + self.P_std * rnd)
        sigma = torch.clamp(sigma, self.sigma_min, self.sigma_max)
        return sigma

    @staticmethod
    def lognormal_pdf(
        sigma: torch.Tensor, P_mean: float, P_std: float
    ) -> torch.Tensor:
        """
        Lognormal probability density function f(sigma).

        This is the ERDM-style reweighting term that focuses model capacity
        on the most informative intermediate noise levels (where the PDF
        peaks around exp(P_mean)).

        f(sigma) = 1/(sigma * P_std * sqrt(2*pi)) *
                   exp(-(ln(sigma) - P_mean)^2 / (2 * P_std^2))

        Reference: Rühling Cachay et al., ERDM (NeurIPS 2025), Eq. (4).

        Args:
            sigma: Noise levels, shape (B,).
            P_mean: Mean of the log-normal distribution.
            P_std: Std of the log-normal distribution.

        Returns:
            PDF values, shape (B,).
        """
        return (1.0 / (sigma * P_std * np.sqrt(2.0 * np.pi))) * torch.exp(
            -0.5 * ((torch.log(sigma) - P_mean) / P_std) ** 2
        )

    # ------------------------------------------------------------------
    # Heun 2nd-order ODE Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample(
        self,
        condition: torch.Tensor,
        n_steps: int = 35,
        rho: Optional[float] = None,
        batch_size: Optional[int] = None,
        progress: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Generate x_0 via Heun 2nd-order ODE integration (EDM Algorithm 1).

        ODE: dx/dsigma = (x - D_theta(x; sigma)) / sigma

        Using rho-scheduling with Heun's 2nd-order trapezoidal rule.

        Args:
            condition: Measurement vector (B, cond_input_dim).
            n_steps: Number of integration steps (35-50 recommended).
            rho: Override rho value (default: self.rho).
            batch_size: Override batch size.
            progress: Show progress bar.
            temperature: Sampling temperature (>1 amplifies the initial noise,
                widening the sampled posterior; 1.0 = default ODE sampling).

        Returns:
            Predicted clean Cholesky vector (B, D).
        """
        rho = rho if rho is not None else self.rho

        if batch_size is not None:
            B = batch_size
            if condition.shape[0] != B:
                condition = condition[:B]
        else:
            B = condition.shape[0]

        device = next(self.parameters()).device
        condition = condition.to(device)

        # Encode condition once
        cond_emb, _ = self.conditioning(condition)

        # ---- Time discretization (rho-schedule) ----
        sigmas = edm_sigmas(n_steps, self.sigma_min, self.sigma_max, rho, device)
        # Append sigma=0 at the end for the final clean prediction
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        # ---- Start from pure noise at sigma_max ----
        x = sigmas[0] * temperature * torch.randn(B, self.cholesky_dim, device=device)

        iterator = range(n_steps)
        if progress:
            iterator = tqdm(iterator, desc=f"EDM Heun Sampling ({n_steps} steps)")

        for i in iterator:
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]

            # ---- Heun 2nd-order step (EDM Algorithm 1) ----
            # Step 1: Compute slope at current point
            sigma_cur_tensor = sigma_cur.expand(B)
            d_cur = (x - self.denoiser(x, sigma_cur_tensor, cond_emb)) / sigma_cur

            # Step 2: Euler step to tentative point
            dt = sigma_next - sigma_cur  # negative (going to smaller sigma)
            x_euler = x + dt * d_cur

            # Step 3: Compute slope at tentative point (if not at sigma=0)
            if sigma_next > 1e-8:
                sigma_next_tensor = sigma_next.expand(B)
                d_next = (
                    x_euler - self.denoiser(x_euler, sigma_next_tensor, cond_emb)
                ) / sigma_next
            else:
                d_next = torch.zeros_like(d_cur)

            # Step 4: Trapezoidal correction
            x = x + dt * (d_cur + d_next) / 2.0

        # ---- Positivity projection on final output ----
        # Cholesky-only: softplus the diagonal so rho = L L^dag / Tr is valid.
        # Hermitian representations (direct/fix_trace) are NOT softplus'd --
        # their coordinates are raw rho params and softplus would distort the
        # isometric geometry the training loss was defined in. Applies once
        # after the ODE, preserving Heun 2nd-order accuracy.
        if self.representation == "cholesky":
            x[:, :self.d] = F.softplus(x[:, :self.d])

        return x

    # ------------------------------------------------------------------
    # CFG (Classifier-Free Guidance) Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_with_cfg(
        self,
        condition: torch.Tensor,
        cfg_weight: float = 2.0,
        n_steps: int = 35,
        batch_size: Optional[int] = None,
        progress: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Sample with classifier-free guidance.

        D_cfg = D_uncond + w * (D_cond - D_uncond)

        Args:
            condition: Measurement vector.
            cfg_weight: Guidance strength (1.0-2.0 recommended for QST).
            n_steps: Number of Heun steps.
            batch_size: Override batch size.
            progress: Show progress bar.
            temperature: Sampling temperature (>1 amplifies the initial noise,
                widening the sampled posterior; 1.0 = default ODE sampling).

        Returns:
            Predicted clean Cholesky vector.
        """
        if batch_size is not None:
            B = batch_size
            if condition.shape[0] != B:
                condition = condition[:B]
        else:
            B = condition.shape[0]

        device = next(self.parameters()).device
        condition = condition.to(device)

        # Encode conditions
        cond_emb, _ = self.conditioning(condition)
        uncond_emb, _ = self.conditioning(condition, force_dropout=True)

        # Time discretization
        sigmas = edm_sigmas(n_steps, self.sigma_min, self.sigma_max, self.rho, device)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        # Start from noise
        x = sigmas[0] * temperature * torch.randn(B, self.cholesky_dim, device=device)

        iterator = range(n_steps)
        if progress:
            iterator = tqdm(iterator, desc=f"EDM Heun+CFG ({n_steps} steps)")

        for i in iterator:
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_cur_tensor = sigma_cur.expand(B)

            # Get conditional and unconditional denoisers
            D_cond = self.denoiser(x, sigma_cur_tensor, cond_emb)
            D_uncond = self.denoiser(x, sigma_cur_tensor, uncond_emb)

            # CFG: extrapolate beyond conditional
            D_cfg = D_uncond + cfg_weight * (D_cond - D_uncond)

            # Heun step with CFG
            d_cur = (x - D_cfg) / sigma_cur
            dt = sigma_next - sigma_cur
            x_euler = x + dt * d_cur

            if sigma_next > 1e-8:
                sigma_next_tensor = sigma_next.expand(B)
                D_cfg_next = (
                    self.denoiser(x_euler, sigma_next_tensor, uncond_emb)
                    + cfg_weight * (
                        self.denoiser(x_euler, sigma_next_tensor, cond_emb)
                        - self.denoiser(x_euler, sigma_next_tensor, uncond_emb)
                    )
                )
                d_next = (x_euler - D_cfg_next) / sigma_next
            else:
                d_next = torch.zeros_like(d_cur)

            x = x + dt * (d_cur + d_next) / 2.0

        # ---- Positivity projection on final output (Cholesky-only) ----
        if self.representation == "cholesky":
            x[:, :self.d] = F.softplus(x[:, :self.d])

        return x

    # ------------------------------------------------------------------
    # DPS Sampling (Diffusion Posterior Sampling with Born-rule guidance)
    # ------------------------------------------------------------------

    def sample_dps(
        self,
        condition: torch.Tensor,
        measurement_freqs: torch.Tensor,
        n_steps: int = 35,
        dps_scale: float = 0.1,
        rho: Optional[float] = None,
        progress: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """
        Heun ODE sampling with diffusion-posterior-sampling (DPS) guidance.

        At every step, the denoiser's prediction x_hat0 = D_theta(x, sigma) is
        converted to a density matrix and its Born-rule log-likelihood under
        the observed Pauli frequencies is computed:
            log p(m | x_hat0) = sum_j m_j log p_j,   p_j = Tr(rho_hat0 P_j).
        The gradient of this log-likelihood w.r.t. the ODE point is added to
        the update with strength ``dps_scale``, pulling the trajectory toward
        states consistent with the measurement --- an explicit, sampling-time
        use of the likelihood that MLE also uses, complementing the implicit
        prior learned by the network.

        Stage 4 of the likelihood-utilization ladder (see configs/n2_lowshot.yaml).
        Disabled by setting dps_scale = 0.0 (evaluate.py then uses the plain
        Heun or CFG path).

        Args:
            condition: Measurement condition (B, cond_input_dim).
            measurement_freqs: Observed Pauli frequencies (6^n,) or (B, 6^n).
            n_steps: Number of Heun steps.
            dps_scale: Guidance strength for the likelihood gradient.
            rho: rho-scheduling exponent (default: self.rho).
            progress: Show progress bar.

        Returns:
            Predicted clean Cholesky vector (B, D) with positivity projection.
        """
        rho_sched = rho if rho is not None else self.rho
        B = condition.shape[0]
        device = next(self.parameters()).device
        condition = condition.to(device)
        m = measurement_freqs.float().to(device)
        if m.dim() == 1:
            m = m.unsqueeze(0).expand(B, -1)

        cond_emb, _ = self.conditioning(condition)
        sigmas = edm_sigmas(n_steps, self.sigma_min, self.sigma_max, rho_sched, device)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        x = sigmas[0] * temperature * torch.randn(B, self.cholesky_dim, device=device)

        iterator = range(n_steps)
        if progress:
            iterator = tqdm(iterator, desc=f"EDM DPS Sampling ({n_steps} steps)")

        for i in iterator:
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_cur_tensor = sigma_cur.expand(B)

            # --- ODE slope at current point (no grad needed) ---
            with torch.no_grad():
                d_cur = (x - self.denoiser(x, sigma_cur_tensor, cond_emb)) / sigma_cur

            dt = sigma_next - sigma_cur
            x_euler = x + dt * d_cur

            # --- DPS guidance: Born-rule likelihood gradient at the predicted
            # point (explicit sampling-time likelihood) ---
            # NOTE: needs grad even when called inside an outer no_grad scope
            # (evaluate.py wraps sampling in torch.no_grad()).
            if dps_scale > 0 and sigma_next > 1e-8:
                with torch.enable_grad():
                    x_euler = x_euler.detach().requires_grad_(True)
                    x_hat0 = self.denoiser(x_euler, sigma_next.expand(B), cond_emb)
                    # Decode in the representation the model was trained in
                    # (Hermitian outputs are raw rho params, not Cholesky vectors).
                    if self.representation in ("hermitian", "bloch"):
                        rho_hat = _vec_to_dm_for_rep(x_hat0, self.d, self.representation)
                    else:
                        rho_hat = _vector_to_dm_pytorch(x_hat0, self.d)
                    p = self._born_probs(rho_hat)          # (B, 6^n)
                    ll = torch.sum(m * torch.log(p.clamp(min=1e-8)))  # scalar
                    grad = torch.autograd.grad(ll, x_euler)[0]
                x_euler = x_euler + dps_scale * grad
                x_euler = x_euler.detach()

            # --- Heun corrector (second order) ---
            if sigma_next > 1e-8:
                sigma_next_tensor = sigma_next.expand(B)
                with torch.no_grad():
                    d_next = (
                        x_euler - self.denoiser(x_euler, sigma_next_tensor, cond_emb)
                    ) / sigma_next
            else:
                d_next = torch.zeros_like(d_cur)

            x = x + dt * (d_cur + d_next) / 2.0

        # ---- Positivity projection on final output (Cholesky-only) ----
        if self.representation == "cholesky":
            x[:, :self.d] = F.softplus(x[:, :self.d])
        return x

    # ------------------------------------------------------------------
    # Unconditional Sampling
    # ------------------------------------------------------------------

    @torch.no_grad()
    def unconditional_sample(
        self,
        batch_size: int = 1,
        n_steps: int = 35,
        progress: bool = True,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Generate states unconditionally (no measurement conditioning)."""
        device = next(self.parameters()).device
        dummy_cond = torch.zeros(
            batch_size, self.conditioning.encoder[0].in_features, device=device
        )
        cond_emb, _ = self.conditioning(dummy_cond, force_dropout=True)

        sigmas = edm_sigmas(n_steps, self.sigma_min, self.sigma_max, self.rho, device)
        sigmas = torch.cat([sigmas, torch.zeros(1, device=device)])

        x = sigmas[0] * temperature * torch.randn(batch_size, self.cholesky_dim, device=device)

        iterator = range(n_steps)
        if progress:
            iterator = tqdm(iterator, desc=f"EDM Uncond ({n_steps} steps)")

        for i in iterator:
            sigma_cur = sigmas[i]
            sigma_next = sigmas[i + 1]
            sigma_cur_tensor = sigma_cur.expand(batch_size)

            d_cur = (x - self.denoiser(x, sigma_cur_tensor, cond_emb)) / sigma_cur
            dt = sigma_next - sigma_cur
            x_euler = x + dt * d_cur

            if sigma_next > 1e-8:
                sigma_next_tensor = sigma_next.expand(batch_size)
                d_next = (
                    x_euler - self.denoiser(x_euler, sigma_next_tensor, cond_emb)
                ) / sigma_next
            else:
                d_next = torch.zeros_like(d_cur)

            x = x + dt * (d_cur + d_next) / 2.0

        # ---- Positivity projection on final output (Cholesky-only) ----
        if self.representation == "cholesky":
            x[:, :self.d] = F.softplus(x[:, :self.d])

        return x