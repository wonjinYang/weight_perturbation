# Weight Perturbation Library

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)

**Weight Perturbation** is a modular Python library implementing the Weight Perturbation strategy based on congested transport formulations. It enables fine-tuning of pre-trained generative models (e.g., WGAN-GP generators) to align with target distributions or evidence-based virtual targets, using optimal transport (OT) principles for stable and mathematically grounded perturbations.

This library supports:
- **Target-Given Perturbations**: Using global W2 gradient flow with congestion bounds via `WeightPerturberTargetGiven`.
- **Evidence-Based Perturbations**: With virtual target estimation via broadened KDE and multi-marginal OT using `WeightPerturberTargetNotGiven`.

Built on PyTorch with inheritance-based design, it provides tools for models, samplers, losses, perturbation algorithms, pretraining utilities, and visualization helpers. Ideal for researchers and practitioners in generative modeling, optimal transport, and distribution alignment.

## Key Features
- **Object-Oriented Design**: Base class `WeightPerturber` with specialized subclasses for different perturbation strategies.
- **Inheritance Structure**: Shared functionality in base class, specialized methods in subclasses.
- **Congested Transport Integration**: Implements perturbation with stability mechanisms (clipping, momentum, early stopping, adaptive learning rates).
- **Toy Dataset Examples**: Demonstrates on 2D Gaussian clusters for quick prototyping.
- **Pretraining Support**: Built-in WGAN-GP pretraining for generators.
- **Visualization**: Plot distributions to compare original, perturbed, and target/evidence.
- **Configurable**: Use YAML configs for hyperparameters; extensible for advanced use cases.
- **Tested**: Comprehensive unit tests for reliability.
- **Efficient**: Leverages `geomloss` for fast OT computations.
- **Backward Compatibility**: Old class names (`WeightPerturberSection2`, `WeightPerturberSection3`) still supported.

## Installation

### Prerequisites
- Python 3.8+
- pip (for installation)

### Install from Source
Clone the repository and install in editable mode:

```
git clone https://github.com/yourusername/weight_perturbation.git
cd weight_perturbation
pip install -e .
```

### Dependencies
Install required packages via `requirements.txt`:

```
pip install -r requirements.txt
```

Core dependencies include:
- `torch>=2.0.0`
- `geomloss>=0.2.6` (for Sinkhorn-based OT)
- `numpy>=1.24.0`
- `matplotlib>=3.8.0`
- `pyyaml>=6.0.1`

For development (tests, packaging):
```
pip install -r requirements.txt --extras-require dev
```

### PyPI Installation (Coming Soon)
Once published:
```
pip install weight_perturbation
```

## Quick Start

### Basic Usage
Import and use core components:
```python
import torch
from weight_perturbation import Generator, pretrain_wgan_gp, WeightPerturberTargetGiven, sample_target_data, sample_real_data

# Define real sampler
real_sampler = lambda bs: sample_real_data(bs)

# Pretrain generator
gen = Generator(noise_dim=2, data_dim=2, hidden_dim=256)
crit = Critic(data_dim=2, hidden_dim=256)
pretrained_gen, _ = pretrain_wgan_gp(gen, crit, real_sampler, epochs=100)

# Sample target
target = sample_target_data(1000)

# Perturb using target-given approach
perturber = WeightPerturberTargetGiven(pretrained_gen, target)
perturbed_gen = perturber.perturb(steps=20)
```

### Running Examples
Execute demo scripts for toy datasets:

- **Target-Given Perturbation**:
```bash
python examples/example_section2.py --plot --verbose
```

- **Evidence-Based Perturbation**:
```bash
python examples/example_section3.py --plot --verbose --num_evidence_domains=4
```

These scripts pretrain a generator, perform perturbation, evaluate Wasserstein distances, and optionally plot results.

## Documentation

### Directory Structure

```
weight_perturbation/
├── setup.py                  # For packaging and distribution (e.g., pip install)
├── README.md                 # Library documentation, usage examples
├── requirements.txt          # Dependencies (e.g., torch, geomloss, numpy, matplotlib)
├── src/
│   ├── weight_perturbation/
│   │   ├── __init__.py       # Package init: from .perturbation import WeightPerturberTargetGiven
│   │   ├── models.py         # Neural network models (Generator, Critic)
│   │   ├── samplers.py       # Data sampling functions for targets, evidence, virtual targets
│   │   ├── losses.py         # Loss functions (Wasserstein, OT maps, entropy regs)
│   │   ├── perturbation.py   # Core perturbation classes and functions
│   │   ├── pretrain.py       # Pretraining utilities for WGAN-GP
│   │   └── utils.py          # Helper functions (plotting, vectorization, etc.)
├── tests/                    # Unit tests
│   ├── test_models.py
│   ├── test_samplers.py
│   ├── test_losses.py
│   ├── test_perturbation.py
│   └── test_pretrain.py
├── examples/                 # Usage examples (toy datasets)
│   ├── example_section2.py   # Demo for target-given perturbation
│   └── example_section3.py   # Demo for evidence-based perturbation
└── configs/                  # YAML/JSON configs for hyperparameters
    └── default.yaml          # Default params (etas, lambdas, etc.)
```


### Overview
The library uses an inheritance-based structure for modularity:
- **Base Class**: `WeightPerturber` - Contains shared functionality and abstract methods.
- **Target-Given**: `WeightPerturberTargetGiven` - For explicit target distributions.
- **Evidence-Based**: `WeightPerturberTargetNotGiven` - For implicit targets via evidence domains.
- **Models**: `Generator` and `Critic` for WGAN-GP.
- **Samplers**: Functions for real, target, evidence, KDE, and virtual target sampling.
- **Losses**: Wasserstein distances, barycentric OT maps, global W2, multi-marginal OT.
- **Pretrain**: WGAN-GP training utility.
- **Utils**: Parameter vectorization, plotting, config loading.

### Detailed Usage

#### 1. Pretraining a Generator
Use `pretrain_wgan_gp` to train a generator on real data:

```python
from weight_perturbation import pretrain_wgan_gp, sample_real_data

gen, crit = pretrain_wgan_gp(
    Generator(...), Critic(...),
    real_sampler=lambda bs: sample_real_data(bs),
    epochs=300, batch_size=64, verbose=True
)
```

#### 2. Target-Given Perturbation
Perturb towards an explicit target:

```python
from weight_perturbation import WeightPerturberTargetGiven, sample_target_data

target = sample_target_data(1600)
perturber = WeightPerturberTargetGiven(gen, target)
perturbed_gen = perturber.perturb(steps=24, eta_init=0.017, verbose=True)
```

#### 3. Evidence-Based Perturbation
Perturb using evidence domains without explicit target:

```python
from weight_perturbation import WeightPerturberTargetNotGiven, sample_evidence_domains

evidence, centers = sample_evidence_domains(num_domains=3)
perturber = WeightPerturberTargetNotGiven(gen, evidence, centers)
perturbed_gen = perturber.perturb(epochs=100, eta_init=0.045, verbose=True)
```

#### 4. Evaluation and Visualization
Compute distances and plot:

```python
from weight_perturbation import compute_wasserstein_distance, plot_distributions

w2 = compute_wasserstein_distance(gen_samples, target_samples)
plot_distributions(original=gen_samples, perturbed=pert_samples, target_or_virtual=target_samples)
```

#### 5. Configuration
Customize via YAML (e.g., `configs/default.yaml`):

```yaml
model:
  noise_dim: 2
  hidden_dim: 256
perturbation_target_given:
  steps: 24
  eta_init: 0.017
perturbation_target_not_given:
  epochs: 100
  eta_init: 0.045
```

Load with `load_config(path)`.

### Class Hierarchy

```
WeightPerturber (Abstract Base Class)
├── WeightPerturberTargetGiven (formerly WeightPerturberSection2)
└── WeightPerturberTargetNotGiven (formerly WeightPerturberSection3)
```

**Shared functionality in base class:**
- Parameter management (`_create_generator_copy`, `_apply_parameter_update`)
- Congested transport mechanics (`_compute_delta_theta`)
- Early stopping logic (`_check_early_stopping`)
- Best state management (`_update_best_state`, `_restore_best_state`)

**Specialized in subclasses:**
- Loss computation (`_compute_loss_and_grad`)
- Validation and adaptation (`_validate_and_adapt`)
- Main perturbation loop (`perturb`)

### Mathematical Foundations
This library is grounded in congested transport theory, extending WGAN-GP interpretations:
- **Target-Given**: Uses global W2 gradient flow with congestion bounds for stability.
- **Evidence-Based**: Estimates virtual targets via KDE and multi-marginal OT, preventing mode collapse with entropy regularization.

For details, see the attached papers or inline comments in code.

## Testing

Run the test suite:
```bash
pytest tests/
```

Tests cover models, samplers, losses, perturbation logic, inheritance structure, and pretraining.

## Contributing
Contributions are welcome! Follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/new-feature`).
3. Commit changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature/new-feature`).
5. Open a Pull Request.

Please include tests and update documentation for new features.

## Backward Compatibility

The library maintains backward compatibility with the original class names:
- `WeightPerturberSection2` → `WeightPerturberTargetGiven`
- `WeightPerturberSection3` → `WeightPerturberTargetNotGiven`

Old code will continue to work, but we recommend updating to the new class names for better clarity.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- Inspired by congested transport interpretations of WGAN-GP (Milne et al., 2021).
- Utilizes `geomloss` for efficient OT computations.
- Built with PyTorch for flexibility.

For questions or issues, open a GitHub issue or contact [rhrlajrwk56@gmail.com](mailto:rhrlajrwk56@gmail.com).
---
*Generated on August 01, 2025. Library version: 0.1.0*