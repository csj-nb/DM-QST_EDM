"""Tests for the diffusion model module."""

import numpy as np
import torch
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.noise_schedules import make_beta_schedule, compute_diffusion_parameters
from src.models.conditioning import ConditioningNetwork, TimeEmbedding
from src.models.unet import CholeskyUNet
from src.models.diffusion import DDPM


def test_beta_schedule_linear():
    """Test linear beta schedule."""
    betas = make_beta_schedule(1000, "linear", 1e-4, 0.02)
    assert betas.shape == (1000,)
    assert torch.allclose(betas[0], torch.tensor(1e-4), atol=1e-6)
    assert torch.allclose(betas[-1], torch.tensor(0.02), atol=1e-3)
    assert torch.all(betas > 0) and torch.all(betas < 1)


def test_beta_schedule_cosine():
    """Test cosine beta schedule."""
    betas = make_beta_schedule(1000, "cosine")
    assert betas.shape == (1000,)
    assert torch.all(betas > 0) and torch.all(betas < 1)
    # Cosine schedule should clip at 0.999
    assert torch.all(betas <= 0.999)


def test_diffusion_parameters():
    """Test derived diffusion parameters."""
    betas = make_beta_schedule(100, "linear")
    params = compute_diffusion_parameters(betas)

    # Check shapes
    for key in ["betas", "alphas", "alphas_cumprod", "sqrt_alphas_cumprod"]:
        assert params[key].shape == (100,), f"Wrong shape for {key}"

    # alphas = 1 - betas
    assert torch.allclose(params["alphas"], 1.0 - betas)

    # alphas_cumprod[0] = alphas[0]
    assert torch.allclose(params["alphas_cumprod"][0], params["alphas"][0])

    # sqrt_alphas_cumprod = sqrt(alphas_cumprod)
    assert torch.allclose(
        params["sqrt_alphas_cumprod"],
        torch.sqrt(params["alphas_cumprod"]),
    )


def test_time_embedding():
    """Test time embedding layer."""
    emb = TimeEmbedding(dim=128)
    t = torch.randint(0, 1000, (16,))
    out = emb(t)
    assert out.shape == (16, 128), f"Wrong shape: {out.shape}"
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_conditioning_network():
    """Test conditioning network."""
    cond_dim = 36  # 6^2 (2 qubits)
    net = ConditioningNetwork(
        input_dim=cond_dim,
        hidden_dim=128,
        cond_dim=64,
        num_resolutions=2,
        base_channels=32,
        dim_mults=(1, 2),
        cond_dropout_prob=0.1,
    )

    x = torch.randn(8, cond_dim)
    cond_emb, film_params = net(x)

    assert cond_emb.shape == (8, 64), f"Wrong cond_emb shape: {cond_emb.shape}"
    assert len(film_params) == 2, f"Wrong number of FiLM params: {len(film_params)}"

    for i, (scale, shift) in enumerate(film_params):
        ch = 32 * (2 ** i)
        assert scale.shape == (8, ch), f"Wrong scale shape at level {i}: {scale.shape}"
        assert shift.shape == (8, ch), f"Wrong shift shape at level {i}: {shift.shape}"


def test_unet_forward():
    """Test UNet forward pass."""
    d = 4  # 2 qubits
    model = CholeskyUNet(
        d=d,
        base_channels=32,
        dim_mults=(1, 2),
        time_emb_dim=64,
        cond_dim=64,
    )

    B = 4
    x = torch.randn(B, d * d)  # Cholesky vector
    t = torch.randint(0, 1000, (B,))
    cond_emb = torch.randn(B, 64)

    out = model(x, t, cond_emb)
    assert out.shape == (B, d * d), f"Wrong output shape: {out.shape}"
    assert not torch.isnan(out).any()


def test_unet_vector_matrix_conversion():
    """Test vector <-> matrix conversion in UNet."""
    d = 4
    B = 8

    # Random Cholesky vector
    x = torch.randn(B, d * d)

    # Convert to matrix and back
    m = CholeskyUNet.vector_to_matrix(x, d)
    assert m.shape == (B, 2, d, d), f"Wrong matrix shape: {m.shape}"

    x_recon = CholeskyUNet.matrix_to_vector(m)
    assert torch.allclose(x, x_recon, atol=1e-6), "Vector-matrix round-trip failed"


def test_ddpm_training_loss():
    """Test DDPM training loss computation."""
    d = 2  # 1 qubit
    cond_dim = 6  # 6^1

    model = DDPM(
        d=d,
        timesteps=100,
        cond_input_dim=cond_dim,
        base_channels=32,
        dim_mults=(1, 2),
        cond_dim=32,
    )

    B = 4
    x_0 = torch.randn(B, d * d)
    condition = torch.randn(B, cond_dim)

    loss = model.training_loss(x_0, condition)
    assert loss.ndim == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(loss)


def test_ddpm_q_sample():
    """Test DDPM forward diffusion."""
    d = 2
    cond_dim = 6

    model = DDPM(
        d=d,
        timesteps=100,
        cond_input_dim=cond_dim,
        base_channels=32,
        dim_mults=(1, 2),
        cond_dim=32,
    )

    x_0 = torch.ones(4, d * d)
    t = torch.tensor([50, 50, 50, 50])
    x_t = model.q_sample(x_0, t)

    assert x_t.shape == x_0.shape
    # With high t, x_t should be noisy (different from x_0)
    assert not torch.allclose(x_t, x_0, atol=0.1)


def test_ddpm_sampling():
    """Test DDPM sampling (with few steps for speed)."""
    d = 2
    cond_dim = 6

    model = DDPM(
        d=d,
        timesteps=10,  # Very few steps for fast test
        cond_input_dim=cond_dim,
        base_channels=32,
        dim_mults=(1, 2),
        cond_dim=32,
    )

    model.eval()
    condition = torch.randn(2, cond_dim)

    with torch.no_grad():
        x_samples = model.sample(condition, progress=False)
        assert x_samples.shape == (2, d * d), f"Wrong sample shape: {x_samples.shape}"

        # DDIM sampling
        x_ddim = model.ddim_sample(condition, ddim_steps=5, progress=False)
        assert x_ddim.shape == (2, d * d), f"Wrong DDIM shape: {x_ddim.shape}"


if __name__ == "__main__":
    test_beta_schedule_linear()
    test_beta_schedule_cosine()
    test_diffusion_parameters()
    test_time_embedding()
    test_conditioning_network()
    test_unet_forward()
    test_unet_vector_matrix_conversion()
    test_ddpm_training_loss()
    test_ddpm_q_sample()
    test_ddpm_sampling()
    print("All diffusion model tests passed!")
