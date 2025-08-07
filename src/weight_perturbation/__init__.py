"""
Weight Perturbation Library v0.2.0

This library implements the Weight Perturbation strategy for neural networks,
with full theoretical implementation of congested transport formulations for 
distribution alignment. It now includes spatial density estimation, traffic flow
computation, Sobolev regularization, and congestion tracking.

The library maintains backward compatibility while adding advanced theoretical components
as optional extensions.
"""

# Core models and original functionality
from .models import Generator, Critic
from .samplers import (
    sample_real_data,
    sample_target_data,
    sample_evidence_domains,
    kde_sampler,
    virtual_target_sampler
)

# Original loss functions
from .losses import (
    compute_wasserstein_distance,
    barycentric_ot_map,
    global_w2_loss_and_grad,
    multi_marginal_ot_loss
)

# Original perturbation classes (maintained for backward compatibility)
from .perturbation import (
    WeightPerturber,
    WeightPerturberTargetGiven,
    WeightPerturberTargetNotGiven,
    # Backward compatibility aliases
    WeightPerturberSection2,
    WeightPerturberSection3
)

# Pretraining and utilities
from .pretrain import pretrain_wgan_gp
from .utils import (
    parameters_to_vector,
    vector_to_parameters,
    plot_distributions,
    load_config,
    compute_device,
    set_seed
)

# New theoretical components (optional import)
try:
    # Congestion tracking and spatial density
    from .congestion import (
        CongestionTracker,
        compute_spatial_density,
        compute_traffic_flow,
        congestion_cost_function,
        verify_continuity_equation,
        QuadraticLinearCost,
        PowerLawCost,
        LogarithmicCost
    )
    
    # Sobolev regularization
    from .sobolev import (
        WeightedSobolevRegularizer,
        AdaptiveSobolevRegularizer,
        sobolev_regularization,
        SobolevConstrainedCritic,
        compute_sobolev_gradient_penalty,
        SobolevWGANLoss,
        apply_spectral_norm
    )
    
    # loss functions with congestion tracking
    from .ct_losses import (
        global_w2_loss_and_grad_with_congestion,
        multi_marginal_ot_loss_with_congestion,
        CongestionAwareLossFunction,
        compute_convergence_metrics
    )
    
    # perturbation classes with congestion tracking
    from .ct_perturbation import (
        CTWeightPerturber,
        CTWeightPerturberTargetGiven,
        CTWeightPerturberTargetNotGiven,
        # Convenience aliases
        CongestionAwareWeightPerturberTargetGiven,
        CongestionAwareWeightPerturberTargetNotGiven
    )
    
    _THEORETICAL_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    _THEORETICAL_COMPONENTS_AVAILABLE = False
    print(f"Note: Advanced theoretical components not available: {e}")
    print("Only basic weight perturbation functionality is enabled.")

# Package metadata
__version__ = '0.2.0'
__author__ = 'Weight Perturbation Team'
__description__ = 'A modular Python library for Weight Perturbation strategy based on congested transport.'
__license__ = 'MIT'

# Core functionality (always available)
__all__ = [
    # Models
    'Generator', 'Critic',
    
    # Samplers
    'sample_real_data', 'sample_target_data', 'sample_evidence_domains',
    'kde_sampler', 'virtual_target_sampler',
    
    # Basic losses
    'compute_wasserstein_distance', 'barycentric_ot_map',
    'global_w2_loss_and_grad', 'multi_marginal_ot_loss',
    
    # Basic perturbation classes
    'WeightPerturber', 'WeightPerturberTargetGiven', 'WeightPerturberTargetNotGiven',
    
    # Backward compatibility
    'WeightPerturberSection2', 'WeightPerturberSection3',
    
    # Pretraining
    'pretrain_wgan_gp',
    
    # Utils
    'parameters_to_vector', 'vector_to_parameters', 'plot_distributions',
    'load_config', 'compute_device', 'set_seed'
]

# Add theoretical components to __all__ if available
if _THEORETICAL_COMPONENTS_AVAILABLE:
    __all__.extend([
        # Congestion components
        'CongestionTracker', 'compute_spatial_density', 'compute_traffic_flow',
        'congestion_cost_function', 'verify_continuity_equation',
        'QuadraticLinearCost', 'PowerLawCost', 'LogarithmicCost',
        
        # Sobolev components
        'WeightedSobolevRegularizer', 'AdaptiveSobolevRegularizer',
        'sobolev_regularization', 'SobolevConstrainedCritic',
        'compute_sobolev_gradient_penalty', 'SobolevWGANLoss',
        'apply_spectral_norm',
        
        # Advanced losses
        'global_w2_loss_and_grad_with_congestion',
        'multi_marginal_ot_loss_with_congestion',
        'CongestionAwareLossFunction', 'compute_convergence_metrics',
        
        # Advanced perturbation
        'CTWeightPerturber',
        'CTWeightPerturberTargetGiven',
        'CTWeightPerturberTargetNotGiven',
        'CongestionAwareWeightPerturberTargetGiven',
        'CongestionAwareWeightPerturberTargetNotGiven'
    ])


def get_version_info():
    """Get detailed version and capability information."""
    info = {
        'version': __version__,
        'basic_functionality': True,
        'theoretical_components': _THEORETICAL_COMPONENTS_AVAILABLE,
        'capabilities': {
            'basic_perturbation': True,
            'congestion_tracking': _THEORETICAL_COMPONENTS_AVAILABLE,
            'sobolev_regularization': _THEORETICAL_COMPONENTS_AVAILABLE,
            'spatial_density_estimation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'traffic_flow_computation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'multi_marginal_congestion': _THEORETICAL_COMPONENTS_AVAILABLE
        }
    }
    return info


def check_theoretical_support():
    """Check if theoretical components are available and working."""
    if not _THEORETICAL_COMPONENTS_AVAILABLE:
        print("Theoretical components are not available.")
        return False
    
    try:
        # Test basic imports
        import torch
        from .congestion import CongestionTracker
        from .sobolev import WeightedSobolevRegularizer
        
        # Test basic functionality
        tracker = CongestionTracker()
        regularizer = WeightedSobolevRegularizer()
        
        print("All theoretical components are available and functional.")
        return True
        
    except Exception as e:
        print(f"Theoretical components have issues: {e}")
        return False


# Usage examples and documentation
"""
Basic Usage (always available):
    from weight_perturbation import Generator, WeightPerturberTargetGiven
    
    gen = Generator(noise_dim=2, data_dim=2, hidden_dim=256)
    perturber = WeightPerturberTargetGiven(gen, target_samples)
    perturbed_gen = perturber.perturb()

Advanced Usage (if theoretical components are available):
    from weight_perturbation import (
        CTWeightPerturberTargetGiven,
        SobolevConstrainedCritic,
        CongestionTracker
    )
    
    # Create Sobolev-constrained critic for congestion tracking
    critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=256)
    
    # Use congestion-aware perturbation
    perturber = CTWeightPerturberTargetGiven(
        gen, target_samples, critic=critic, enable_congestion_tracking=True
    )
    perturbed_gen = perturber.perturb()

Check capabilities:
    from weight_perturbation import get_version_info, check_theoretical_support
    
    print(get_version_info())
    check_theoretical_support()
"""