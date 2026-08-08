"""Tests for the Cholesky representation module."""

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.representation.cholesky import (
    dm_to_cholesky,
    cholesky_to_dm,
    cholesky_dim,
)
from src.representation.constraints import is_valid_dm, check_dm_constraints
from src.data.states import haar_random_pure, hilbert_schmidt_random


def test_cholesky_roundtrip_pure():
    """Test round-trip for a pure state."""
    rho = haar_random_pure(2, seed=42)  # 2 qubits, d=4
    x = dm_to_cholesky(rho, eps=1e-6)
    rho_recon = cholesky_to_dm(x)

    assert is_valid_dm(rho_recon), "Reconstructed state is not a valid density matrix"
    assert np.allclose(rho, rho_recon, atol=1e-8), "Round-trip failed for pure state"

    # Check dimension
    d = 2 ** 2
    assert len(x) == d * d, f"Cholesky vector has wrong dimension: {len(x)} != {d*d}"


def test_cholesky_roundtrip_mixed():
    """Test round-trip for a mixed state."""
    rho = hilbert_schmidt_random(2, seed=123)
    x = dm_to_cholesky(rho, eps=1e-6)
    rho_recon = cholesky_to_dm(x)

    assert is_valid_dm(rho_recon), "Reconstructed state is not a valid density matrix"
    assert np.allclose(rho, rho_recon, atol=1e-8), "Round-trip failed for mixed state"


def test_cholesky_roundtrip_1qubit():
    """Test round-trip for 1 qubit."""
    rho = haar_random_pure(1, seed=7)
    x = dm_to_cholesky(rho, eps=1e-6)
    rho_recon = cholesky_to_dm(x)

    assert is_valid_dm(rho_recon)
    assert np.allclose(rho, rho_recon, atol=1e-8)
    assert len(x) == 4  # d=2, d*d=4


def test_cholesky_roundtrip_3qubit():
    """Test round-trip for 3 qubits."""
    rho = haar_random_pure(3, seed=99)
    x = dm_to_cholesky(rho, eps=1e-6)
    rho_recon = cholesky_to_dm(x)

    assert is_valid_dm(rho_recon)
    assert np.allclose(rho, rho_recon, atol=1e-8)
    assert len(x) == 64  # d=8, d*d=64


def test_batch_cholesky():
    """Test batch conversion."""
    rhos = np.array([
        haar_random_pure(1, seed=i) for i in range(10)
    ])
    x_batch = dm_to_cholesky(rhos, eps=1e-6)
    rho_batch = cholesky_to_dm(x_batch)

    for i in range(10):
        assert is_valid_dm(rho_batch[i])
        assert np.allclose(rhos[i], rho_batch[i], atol=1e-6)


def test_cholesky_validity():
    """Test that Cholesky reconstruction always produces valid density matrices."""
    for n in [1, 2, 3]:
        for _ in range(20):
            rho = hilbert_schmidt_random(n, seed=None)
            x = dm_to_cholesky(rho, eps=1e-6)
            rho_recon = cholesky_to_dm(x)
            info = check_dm_constraints(rho_recon)
            assert info["is_valid"], f"Invalid DM for n={n}: {info}"


def test_cholesky_dimension():
    """Test the Cholesky vector dimension matches d*d."""
    for n in range(1, 5):
        d = 2 ** n
        assert cholesky_dim(n) == d * d


def test_pure_state_purity():
    """Test that pure states maintain high purity after round-trip."""
    rho = haar_random_pure(2, seed=42)
    x = dm_to_cholesky(rho, eps=1e-6)
    rho_recon = cholesky_to_dm(x)

    purity_orig = np.real(np.trace(rho @ rho))
    purity_recon = np.real(np.trace(rho_recon @ rho_recon))

    assert np.isclose(purity_orig, 1.0, atol=1e-10)
    assert np.isclose(purity_recon, 1.0, atol=1e-10)


def test_layout_consistent_with_unet():
    """The Cholesky vector layout must match what CholeskyUNet expects.

    Regression: cholesky.py used np.tril_indices (row-major) while
    unet.py/vdm.py use column-major loops; for d >= 4 (2+ qubits) the
    two orders differ, so the UNet's 2D image was scrambled. This test
    cross-checks the two modules directly.
    """
    try:
        import torch
        from src.models.unet import CholeskyUNet
    except ImportError:
        print("  (torch not available, skipping layout test)")
        return

    for n in [1, 2, 3]:
        d = 2 ** n
        rho = haar_random_pure(n, seed=100 + n)
        x = dm_to_cholesky(rho, eps=1e-6)

        # CholeskyUNet's vector_to_matrix should reconstruct the same
        # lower-triangular matrix (real/imag) that dm_to_cholesky encoded.
        x_t = torch.from_numpy(x).float().unsqueeze(0)
        m = CholeskyUNet.vector_to_matrix(x_t, d)  # (1, 2, d, d)
        L_real = m[0, 0].numpy()
        L_imag = m[0, 1].numpy()

        # Rebuild L directly from the numpy vector (column-major).
        L = np.zeros((d, d), dtype=np.complex128)
        np.fill_diagonal(L, x[:d])
        idx = d
        for col in range(d):
            for row in range(col + 1, d):
                L[row, col] = x[idx] + 1j * x[idx + 1]
                idx += 2

        assert np.allclose(L_real, np.real(L)), f"n={n}: real layout mismatch"
        assert np.allclose(L_imag, np.imag(L)), f"n={n}: imag layout mismatch"

        # And the round trip through the UNet's matrix_to_vector must be
        # the identity on the vector.
        x_back = CholeskyUNet.matrix_to_vector(m)[0].numpy()
        assert np.allclose(x, x_back), f"n={n}: vector round-trip mismatch"


if __name__ == "__main__":
    test_cholesky_roundtrip_pure()
    test_cholesky_roundtrip_mixed()
    test_cholesky_roundtrip_1qubit()
    test_cholesky_roundtrip_3qubit()
    test_batch_cholesky()
    test_cholesky_validity()
    test_cholesky_dimension()
    test_pure_state_purity()
    test_layout_consistent_with_unet()
    print("All Cholesky tests passed!")
