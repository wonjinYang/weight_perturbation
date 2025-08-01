# src/weight_perturbation/__init__.py
# This is the package initialization file for the weight_perturbation library.
# It exposes the main classes, functions, and utilities for easy import.
# Users can import from the package level, e.g., from weight_perturbation import Generator

from .models import Generator, Critic
from .samplers import (
    sample_real_data,
    sample_target_data,
    sample_evidence_domains,
    kde_sampler,
    virtual_target_sampler
)
from .losses import (
    compute_wasserstein_distance,
    barycentric_ot_map,
    global_w2_loss_and_grad,
    multi_marginal_ot_loss
)
from .perturbation import (
    WeightPerturber,
    WeightPerturberTargetGiven,
    WeightPerturberTargetNotGiven,
    # Backward compatibility aliases
    WeightPerturberSection2,
    WeightPerturberSection3
)
from .pretrain import pretrain_wgan_gp
from .utils import (
    parameters_to_vector,
    vector_to_parameters,
    plot_distributions,
    load_config,
    compute_device,
    set_seed
)

# Package metadata
__version__ = '0.1.0'
__author__ = 'Your Name'
__description__ = 'A modular Python library for Weight Perturbation strategy based on congested transport.'
__license__ = 'MIT'

# Optional: Define __all__ to control what is imported with 'from weight_perturbation import *'
__all__ = [
    # Models
    'Generator', 'Critic',
    
    # Samplers
    'sample_real_data', 'sample_target_data', 'sample_evidence_domains',
    'kde_sampler', 'virtual_target_sampler',
    
    # Losses
    'compute_wasserstein_distance', 'barycentric_ot_map',
    'global_w2_loss_and_grad', 'multi_marginal_ot_loss',
    
    # Perturbation (new names)
    'WeightPerturber', 'WeightPerturberTargetGiven', 'WeightPerturberTargetNotGiven',
    
    # Perturbation (backward compatibility)
    'WeightPerturberSection2', 'WeightPerturberSection3',
    
    # Pretrain
    'pretrain_wgan_gp',
    
    # Utils
    'parameters_to_vector', 'vector_to_parameters', 'plot_distributions',
    'load_config', 'compute_device', 'set_seed'
]

# Optional: Add package-level docstring for better documentation
"""
Weight Perturbation Library

This library implements the Weight Perturbation strategy for neural networks,
focusing on congested transport formulations for distribution alignment.
It supports both target-given and evidence-based (target-not-given) perturbations.

Key Features:
- Modular design with separate modules for models, samplers, losses, perturbation, pretraining, and utilities.
- Inheritance-based perturbation classes with shared functionality.
- PyTorch-based implementation.
- Configurable via YAML files.
- Examples and tests included.

Usage Example:
    from weight_perturbation import Generator, pretrain_wgan_gp, WeightPerturberTargetGiven
    
    # Pretrain a generator
    generator, critic = pretrain_wgan_gp(...)
    
    # Perturb for target-given case
    perturber = WeightPerturberTargetGiven(generator, target_samples)
    perturbed_gen = perturber.perturb()

For more details, see README.md or the examples directory.
"""