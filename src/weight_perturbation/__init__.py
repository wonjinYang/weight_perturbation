"""
Weight Perturbation Library v0.2.1

This library implements the Weight Perturbation strategy for neural networks,
with enhanced theoretical implementation of congested transport formulations for 
distribution alignment. Now includes improved mass conservation enforcement,
theoretical validation, and reduced over-conservative constraints.

The library maintains backward compatibility while adding enhanced theoretical components
with better theoretical integration and practical stability.
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
    set_seed,
    safe_density_resize,
)

# Enhanced theoretical components (optional import)
try:
    # Enhanced congestion tracking and spatial density
    from .congestion import (
        CongestionTracker,
        compute_spatial_density,
        compute_traffic_flow,
        congestion_cost_function,
        get_congestion_second_derivative,
        verify_continuity_equation,
        enforce_mass_conservation,
        validate_theoretical_consistency,
        QuadraticLinearCost,
        PowerLawCost,
        LogarithmicCost
    )
    
    # Enhanced Sobolev regularization
    from .sobolev import (
        WeightedSobolevRegularizer,
        AdaptiveSobolevRegularizer,
        MassConservationSobolevRegularizer,
        sobolev_regularization,
        SobolevConstrainedCritic,
        compute_sobolev_gradient_penalty,
        SobolevWGANLoss,
    )
    
    # Enhanced loss functions with congestion tracking
    from .ct_losses import (
        global_w2_loss_and_grad_with_congestion,
        multi_marginal_ot_loss_with_congestion,
        CongestionAwareLossFunction,
        compute_convergence_metrics
    )
    
    # Enhanced perturbation classes with theoretical integration
    from .ct_perturbation import (
        CTWeightPerturber,
        CTWeightPerturberTargetGiven,
        CTWeightPerturberTargetNotGiven,
        # Convenience aliases
        CongestionAwareWeightPerturberTargetGiven,
        CongestionAwareWeightPerturberTargetNotGiven
    )

    # Visualization of congestion tracking
    from .visualizer import (
        CongestedTransportVisualizer,
        MultiMarginalCongestedTransportVisualizer
    )
    
    _THEORETICAL_COMPONENTS_AVAILABLE = True
    
except ImportError as e:
    print(f"Enhanced theoretical components not available: {e}")
    _THEORETICAL_COMPONENTS_AVAILABLE = False

# Package metadata
__version__ = '0.2.1'
__author__ = 'Weight Perturbation Team'
__description__ = 'Enhanced modular Python library for Weight Perturbation strategy based on congested transport with improved theoretical integration.'
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
    'load_config', 'compute_device', 'set_seed', 'safe_density_resize'
]

# Add enhanced theoretical components to __all__ if available
if _THEORETICAL_COMPONENTS_AVAILABLE:
    __all__.extend([
        # Enhanced congestion components
        'CongestionTracker', 'compute_spatial_density', 'compute_traffic_flow',
        'congestion_cost_function', 'get_congestion_second_derivative', 
        'verify_continuity_equation', 'enforce_mass_conservation', 
        'validate_theoretical_consistency',
        'QuadraticLinearCost', 'PowerLawCost', 'LogarithmicCost',
        
        # Enhanced Sobolev components
        'WeightedSobolevRegularizer', 'AdaptiveSobolevRegularizer',
        'MassConservationSobolevRegularizer', 'sobolev_regularization', 
        'SobolevConstrainedCritic', 'compute_sobolev_gradient_penalty', 
        'SobolevWGANLoss',
        
        # Enhanced losses
        'global_w2_loss_and_grad_with_congestion',
        'multi_marginal_ot_loss_with_congestion',
        'CongestionAwareLossFunction', 'compute_convergence_metrics',
        
        # Enhanced perturbation
        'CTWeightPerturber',
        'CTWeightPerturberTargetGiven',
        'CTWeightPerturberTargetNotGiven',
        'CongestionAwareWeightPerturberTargetGiven',
        'CongestionAwareWeightPerturberTargetNotGiven',

        # Enhanced visualizer
        'CongestedTransportVisualizer',
        'MultiMarginalCongestedTransportVisualizer',
    ])


def get_version_info():
    """Get detailed version and capability information."""
    info = {
        'version': __version__,
        'basic_functionality': True,
        'enhanced_theoretical_components': _THEORETICAL_COMPONENTS_AVAILABLE,
        'capabilities': {
            'basic_perturbation': True,
            'enhanced_congestion_tracking': _THEORETICAL_COMPONENTS_AVAILABLE,
            'mass_conservation_enforcement': _THEORETICAL_COMPONENTS_AVAILABLE,
            'theoretical_validation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'adaptive_sobolev_regularization': _THEORETICAL_COMPONENTS_AVAILABLE,
            'spatial_density_estimation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'traffic_flow_computation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'multi_marginal_congestion': _THEORETICAL_COMPONENTS_AVAILABLE,
            'enhanced_theoretical_integration': _THEORETICAL_COMPONENTS_AVAILABLE
        },
        'improvements': {
            'reduced_over_conservatism': _THEORETICAL_COMPONENTS_AVAILABLE,
            'better_mass_conservation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'enhanced_validation': _THEORETICAL_COMPONENTS_AVAILABLE,
            'improved_congestion_scaling': _THEORETICAL_COMPONENTS_AVAILABLE
        }
    }
    return info


def check_theoretical_support():
    """Check if enhanced theoretical components are available and working."""
    if not _THEORETICAL_COMPONENTS_AVAILABLE:
        print("Enhanced theoretical components are not available.")
        return False
    
    try:
        # Test enhanced imports
        import torch
        from .congestion import CongestionTracker, enforce_mass_conservation, validate_theoretical_consistency
        from .sobolev import AdaptiveSobolevRegularizer, MassConservationSobolevRegularizer
        from .ct_losses import CongestionAwareLossFunction
        
        # Test enhanced functionality
        tracker = CongestionTracker()
        regularizer = AdaptiveSobolevRegularizer()
        mass_regularizer = MassConservationSobolevRegularizer()
        loss_function = CongestionAwareLossFunction()
        
        print("All enhanced theoretical components are available and functional.")
        print("New features include:")
        print("  - Mass conservation enforcement")
        print("  - Theoretical validation")
        print("  - Enhanced adaptive Sobolev regularization")
        print("  - Reduced over-conservative constraints")
        print("  - Improved congestion scaling")
        return True
        
    except Exception as e:
        print(f"Enhanced theoretical components have issues: {e}")
        return False


def get_enhancement_summary():
    """Get summary of enhancements made in this version."""
    if not _THEORETICAL_COMPONENTS_AVAILABLE:
        return "Enhanced theoretical components not available."
    
    return """
Enhanced Weight Perturbation Library v0.2.1 Improvements:

Theoretical Enhancements:
  ✓ Mass conservation enforcement with Lagrangian multipliers
  ✓ Theoretical validation and consistency checking
  ✓ Enhanced congestion scaling with H''(x,i) second derivatives
  ✓ Multi-domain mass conservation for evidence-based perturbation

Implementation Improvements:
  ✓ Reduced over-conservative clamping and bounds
  ✓ More responsive adaptive learning rate adjustment
  ✓ Enhanced Sobolev regularization with mass conservation integration
  ✓ Better theoretical justification for weight space perturbation

Stability and Performance:
  ✓ Improved numerical stability without sacrificing theoretical purity
  ✓ More aggressive but theoretically justified parameter adaptation
  ✓ Enhanced rollback mechanisms with theoretical consistency checks
  ✓ Better integration between different theoretical components

New Components:
  ✓ MassConservationSobolevRegularizer for integrated constraints
  ✓ Enhanced CongestionAwareLossFunction with full theoretical integration
  ✓ Theoretical validation functions for runtime consistency checking
  ✓ Improved multi-marginal congestion tracking with better domain coordination
"""


# Usage examples and documentation
"""
Basic Usage (always available):
    from weight_perturbation import Generator, WeightPerturberTargetGiven
    
    gen = Generator(noise_dim=2, data_dim=2, hidden_dim=256)
    perturber = WeightPerturberTargetGiven(gen, target_samples)
    perturbed_gen = perturber.perturb()

Enhanced Usage (if theoretical components are available):
    from weight_perturbation import (
        CTWeightPerturberTargetGiven,
        SobolevConstrainedCritic,
        MassConservationSobolevRegularizer,
        CongestionAwareLossFunction
    )
    
    # Create enhanced Sobolev-constrained critic with mass conservation
    critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=256)
    
    # Use enhanced congestion-aware perturbation with theoretical validation
    perturber = CTWeightPerturberTargetGiven(
        gen, target_samples, critic=critic, 
        enable_congestion_tracking=True,
        config={'theoretical_validation': True, 'mass_conservation_weight': 0.1}
    )
    perturbed_gen = perturber.perturb()
    
    # Check theoretical consistency
    loss_function = CongestionAwareLossFunction(
        enable_mass_conservation=True,
        enable_theoretical_validation=True
    )
    stats = loss_function.get_congestion_statistics()
    print(f"Theoretical consistency: {stats.get('recent_theoretical_consistency', 'N/A')}")

Check capabilities and enhancements:
    from weight_perturbation import get_version_info, check_theoretical_support, get_enhancement_summary
    
    print(get_version_info())
    check_theoretical_support()
    print(get_enhancement_summary())
"""