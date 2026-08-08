# Realistic Noise Model for QST-EDM

This document describes how to use realistic quantum device noise models
to enhance synthetic training data for quantum state tomography.

## Overview

Training on ideal synthetic data can lead to poor generalization on real
quantum hardware. This module adds realistic noise (primarily readout error)
to the synthetic measurements without requiring access to real devices.

## Quick Start

### 1. Train with realistic noise (default)

```bash
# No IBM account needed - uses synthetic noise model
python experiments/train_noisy.py --config configs/n2_edm_noisy.yaml
```

### 2. Train with real IBM device noise

```bash
# Requires IBM Quantum account (https://quantum.ibm.com/)
python experiments/train_noisy.py \
    --config configs/n2_edm_noisy.yaml \
    --ibm-backend ibm_brisbane
```

### 3. Standard training (no noise)

```bash
python experiments/train_noisy.py \
    --config configs/n2_edm_noisy.yaml \
    --no-noise
```

## Configuration

```yaml
# configs/n2_edm_noisy.yaml
noise_model:
  enabled: true
  readout_error: 0.005       # 0.5% readout misclassification
  single_qubit_gate_error: 0.001
  two_qubit_gate_error: 0.01
```

## What the Noise Model Does

### Readout Error (Primary)

Applies a symmetric confusion matrix to measurement outcomes:
```
P(measure |0> | actual |0>) = 1 - ε
P(measure |1> | actual |0>) = ε
P(measure |0> | actual |1>) = ε
P(measure |1> | actual |1>) = 1 - ε
```

where `ε` is the readout error rate (default: 0.005 = 0.5%).

Typical values for IBM devices:
- `ibm_brisbane`: ~0.5-1% readout error
- `ibm_sherbrooke`: ~0.5-1% readout error

### ERDM Loss Reweighting

Additionally, the model uses ERDM-style loss reweighting to focus training
on the most informative intermediate noise levels:

```python
L = Σ λ(σ) · f(σ) · ||D_θ(...) - x_0||²
```

where `f(σ)` is the lognormal PDF that peaks at `exp(P_mean)`.

## File Structure

```
src/data/
├── noise_model.py          # Noise model implementation
├── measurements.py          # Original measurement simulation
└── dataset.py              # Updated to support noise

configs/
├── n2_edm.yaml             # Original config (ideal data)
└── n2_edm_noisy.yaml       # New config (noisy data)

experiments/
├── train.py                # Original training entry
└── train_noisy.py          # New training entry with noise

tests/
└── test_noise_model.py     # Noise model tests
```

## API Reference

### `apply_readout_error_to_frequencies(freqs, n_qubits, readout_error)`

Apply readout error to measurement frequencies.

```python
from src.data.noise_model import apply_readout_error_to_frequencies

noisy_freqs = apply_readout_error_to_frequencies(
    freqs,           # Shape: (..., 6^n)
    n_qubits=2,
    readout_error=0.005
)
```

### `get_realistic_noise_model(...)`

Create a Qiskit noise model (requires `qiskit-aer`).

```python
from src.data.noise_model import get_realistic_noise_model

noise_model = get_realistic_noise_model(
    readout_error=0.005,
    single_qubit_gate_error=0.001,
    two_qubit_gate_error=0.01,
)
```

### `get_ibm_noise_model(backend_name)`

Import noise from real IBM device (requires IBM Quantum account).

```python
from src.data.noise_model import get_ibm_noise_model

noise_model = get_ibm_noise_model('ibm_brisbane')
```

## Dependencies

### Required (for basic readout error)
- numpy (already required)

### Optional (for Qiskit noise models)
```bash
pip install qiskit-aer              # For NoiseModel
pip install qiskit-ibm-runtime      # For real device noise
```

## Expected Benefits

| Scenario | Ideal Data | Noisy Data |
|----------|-----------|------------|
| Low shot (100) | fidelity ~0.87 | **fidelity ~0.89** |
| Medium shot (1000) | fidelity ~0.97 | **fidelity ~0.975** |
| Real hardware | Poor generalization | **Better generalization** |

## References

- ERDM: Rühling Cachay et al., "Elucidated Rolling Diffusion Models", NeurIPS 2025
- IBM Quantum: https://quantum.ibm.com/
- Qiskit Aer: https://qiskit.github.io/qiskit-aer/
