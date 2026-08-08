"""Tests for the VDM (Variational Diffusion Model) module."""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.vdm_snr import MonotonicSNR
from src.models.vdm import VDM, _vector_to_dm_pytorch, _purity
from src.representation.cholesky import cholesky_to_dm
from src.training.vdm_losses import rank_penalty, vector_to_dm


def test_snr_monotonicity():
    """Test that gamma(t) is monotonically decreasing."""
    snr = MonotonicSNR(gamma_min=-10, gamma_max=10)
    t = torch.linspace(0, 1, 100)
    gamma, _ = snr.forward(t)

    # Check monotonic decreasing
    for i in range(len(gamma) - 1):
        assert gamma[i] >= gamma[i+1], \
            f"Not monotonic: gamma({t[i].item():.2f})={gamma[i].item():.4f} < gamma({t[i+1].item():.2f})={gamma[i+1].item():.4f}"


def test_snr_range():
    """Test that gamma stays within [gamma_min, gamma_max]."""
    snr = MonotonicSNR(gamma_min=-8, gamma_max=12)
    t = torch.rand(1000)
    gamma, _ = snr.forward(t)

    assert gamma.min() >= snr.gamma_min - 0.5, f"Below min: {gamma.min()}"
    assert gamma.max() <= snr.gamma_max + 0.5, f"Above max: {gamma.max()}"


def test_snr_derivative():
    """Test that gamma'(t) is non-positive (negative or zero)."""
    snr = MonotonicSNR(gamma_min=-10, gamma_max=10)
    t = torch.linspace(0.01, 0.99, 50, requires_grad=True)
    gamma, gamma_prime = snr.forward(t)

    # gamma'(t) should be <= 0 (gamma is decreasing)
    assert torch.all(gamma_prime <= 1e-5), f"Found positive derivative: {gamma_prime.max()}"


def test_snr_coefficients():
    """Test that alpha^2 + sigma^2 = 1."""
    snr = MonotonicSNR()
    t = torch.rand(100)
    alpha, sigma = snr.get_coefficients(t)

    total = alpha.pow(2) + sigma.pow(2)
    assert torch.allclose(total, torch.ones_like(total), atol=1e-5), \
        f"alpha^2 + sigma^2 != 1: max error = {(total - 1).abs().max()}"


def test_vdm_forward():
    """Test VDM forward diffusion."""
    d = 2  # 1 qubit
    model = VDM(
        d=d, cond_input_dim=6,
        base_channels=32, dim_mults=(1, 2), num_res_blocks=1,
        cond_dim=32,
    )

    x_0 = torch.randn(4, d * d)
    t = torch.rand(4)
    z_t = model.q_sample(x_0, t)

    assert z_t.shape == x_0.shape
    # At t ≈ 0, should be close to x_0
    z_0 = model.q_sample(x_0, torch.zeros(4))
    assert torch.allclose(z_0, x_0, atol=0.01)
    # At t ≈ 1, should be very noisy
    z_1 = model.q_sample(x_0, torch.ones(4))
    assert not torch.allclose(z_1, x_0, atol=0.5)


def test_vdm_training_loss():
    """Test VDM training loss computation."""
    d = 2
    model = VDM(
        d=d, cond_input_dim=6,
        base_channels=32, dim_mults=(1, 2), num_res_blocks=1,
        cond_dim=32, lambda_rank=0.0,
    )

    x_0 = torch.randn(4, d * d)
    condition = torch.randn(4, 6)

    loss = model.training_loss(x_0, condition)
    assert loss.ndim == 0
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_vdm_training_loss_with_rank():
    """Test VDM loss with low-rank constraint."""
    d = 2
    model = VDM(
        d=d, cond_input_dim=6,
        base_channels=32, dim_mults=(1, 2), num_res_blocks=1,
        cond_dim=32, lambda_rank=0.1, lambda_rank_warmup=0,
    )
    model.train()

    x_0 = torch.randn(4, d * d)
    condition = torch.randn(4, 6)

    loss = model.training_loss(x_0, condition)
    assert loss.ndim == 0
    assert loss.item() > 0
    assert not torch.isnan(loss)


def test_vdm_sampling():
    """Test VDM sampling (few steps for speed)."""
    d = 2
    model = VDM(
        d=d, cond_input_dim=6,
        base_channels=32, dim_mults=(1, 2), num_res_blocks=1,
        cond_dim=32,
    )
    model.eval()

    condition = torch.randn(2, 6)

    with torch.no_grad():
        # Test Euler sampling
        x = model.sample(condition, n_steps=10, method="euler", progress=False)
        assert x.shape == (2, d * d)
        assert not torch.isnan(x).any()

        # Test midpoint
        x_mid = model.sample(condition, n_steps=10, method="midpoint", progress=False)
        assert x_mid.shape == (2, d * d)


def test_vector_to_dm_pytorch():
    """Test differentiable vector→DM conversion."""
    d = 4  # 2 qubits
    B = 3

    # Create random Cholesky vectors and convert
    from src.data.states import haar_random_pure
    from src.representation.cholesky import dm_to_cholesky

    rho_np = np.array([haar_random_pure(2, seed=i) for i in range(B)])
    x_np = dm_to_cholesky(rho_np, eps=1e-6)
    x = torch.from_numpy(x_np.astype(np.float32))

    rho_torch = _vector_to_dm_pytorch(x, d)
    assert rho_torch.shape == (B, d, d)

    # Check Hermitian
    diff = rho_torch - rho_torch.transpose(-2, -1).conj()
    assert diff.abs().max() < 1e-5

    # Check trace ≈ 1
    trace = torch.real(torch.diagonal(rho_torch, dim1=-2, dim2=-1).sum(-1))
    assert torch.allclose(trace, torch.ones(B), atol=1e-5)


def test_rank_penalty():
    """Test low-rank penalty on pure vs mixed states."""
    d = 4
    from src.data.states import haar_random_pure, hilbert_schmidt_random
    from src.representation.cholesky import dm_to_cholesky

    # Pure state: should have low penalty
    rho_pure = np.array([haar_random_pure(2, seed=i) for i in range(5)])
    x_pure = torch.from_numpy(dm_to_cholesky(rho_pure).astype(np.float32))
    penalty_pure = rank_penalty(x_pure, d, method="purity")

    # Mixed state: should have higher penalty
    rho_mixed = np.array([hilbert_schmidt_random(2, seed=i) for i in range(5)])
    x_mixed = torch.from_numpy(dm_to_cholesky(rho_mixed).astype(np.float32))
    penalty_mixed = rank_penalty(x_mixed, d, method="purity")

    assert penalty_pure < penalty_mixed, \
        f"Pure penalty {penalty_pure:.4f} should be < mixed penalty {penalty_mixed:.4f}"


if __name__ == "__main__":
    test_snr_monotonicity()
    test_snr_range()
    test_snr_derivative()
    test_snr_coefficients()
    test_vdm_forward()
    test_vdm_training_loss()
    test_vdm_training_loss_with_rank()
    test_vdm_sampling()
    test_vector_to_dm_pytorch()
    test_rank_penalty()
    print("All VDM tests passed!")
