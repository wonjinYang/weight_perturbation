"""
Enhanced Weight Perturbation Library Usage Examples

This script demonstrates the improvements made to the weight perturbation library,
showcasing the enhanced theoretical integration and reduced over-conservatism.
"""

import torch
import numpy as np
from weight_perturbation import (
    # Enhanced components
    CTWeightPerturberTargetGiven,
    CTWeightPerturberTargetNotGiven,
    SobolevConstrainedCritic,
    MassConservationSobolevRegularizer,
    CongestionAwareLossFunction,
    enforce_mass_conservation,
    validate_theoretical_consistency,
    get_enhancement_summary,
    check_theoretical_support,
    
    # Basic components
    Generator,
    sample_target_data,
    sample_evidence_domains
)


def demonstrate_enhanced_target_given_perturbation():
    """
    Demonstrate enhanced target-given perturbation with theoretical validation.
    """
    print("=== Enhanced Target-Given Perturbation ===")
    
    # Create models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=128).to(device)
    
    # Create enhanced Sobolev-constrained critic
    critic = SobolevConstrainedCritic(
        data_dim=2, 
        hidden_dim=128, 
        lambda_sobolev=0.1,
        sobolev_bound=5.0  # Increased from previous conservative bound of 2.0
    ).to(device)
    
    # Generate target samples
    target_samples = sample_target_data(batch_size=200, device=device)
    
    # Enhanced configuration with theoretical validation
    enhanced_config = {
        'eta_init': 0.08,                    # Increased from 0.045
        'clip_norm': 0.6,                    # Increased from 0.4
        'momentum': 0.88,                    # Optimized value
        'patience': 10,                      # Reduced from 12
        'lambda_congestion': 1.0,
        'lambda_sobolev': 0.1,
        'mass_conservation_weight': 0.1,     # NEW: Mass conservation enforcement
        'theoretical_validation': True,      # NEW: Theoretical validation
        'congestion_threshold': 0.2,         # Less conservative threshold
    }
    
    # Create enhanced perturber
    perturber = CTWeightPerturberTargetGiven(
        generator=generator,
        target_samples=target_samples,
        critic=critic,
        config=enhanced_config,
        enable_congestion_tracking=True
    )
    
    print("Starting enhanced perturbation with:")
    print(f"  - Mass conservation enforcement: {enhanced_config['mass_conservation_weight']}")
    print(f"  - Theoretical validation: {enhanced_config['theoretical_validation']}")
    print(f"  - Enhanced learning rate: {enhanced_config['eta_init']}")
    print(f"  - Improved clipping norm: {enhanced_config['clip_norm']}")
    
    # Perform perturbation
    perturbed_generator = perturber.perturb(steps=30, verbose=True)
    
    # Get theoretical statistics
    if hasattr(perturber, 'congestion_tracker'):
        stats = perturber.congestion_tracker.get_statistics()
        print("\nTheoretical Statistics:")
        for key, value in stats.items():
            if 'latest' in key or 'mean' in key:
                print(f"  {key}: {value:.4f}")
    
    print("Enhanced target-given perturbation completed!\n")
    return perturbed_generator


def demonstrate_enhanced_evidence_based_perturbation():
    """
    Demonstrate enhanced evidence-based perturbation with multi-marginal integration.
    """
    print("=== Enhanced Evidence-Based Perturbation ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=128).to(device)
    
    # Create multiple critics for multi-marginal setup
    critics = [
        SobolevConstrainedCritic(data_dim=2, hidden_dim=128).to(device)
        for _ in range(3)
    ]
    
    # Generate evidence domains
    evidence_list, centers = sample_evidence_domains(
        num_domains=3, 
        samples_per_domain=50,
        device=device
    )
    
    # Enhanced configuration for evidence-based perturbation
    enhanced_config = {
        'eta_init': 0.08,                    # Increased from 0.045
        'clip_norm': 0.6,                    # Increased from 0.4
        'momentum': 0.88,
        'patience': 10,                      # Reduced patience for faster convergence
        'lambda_congestion': 1.0,
        'lambda_sobolev': 0.1,
        'lambda_entropy': 0.012,
        'mass_conservation_weight': 0.1,     # NEW: Mass conservation
        'theoretical_validation': True,      # NEW: Validation
    }
    
    # Create enhanced multi-marginal perturber
    perturber = CTWeightPerturberTargetNotGiven(
        generator=generator,
        evidence_list=evidence_list,
        centers=centers,
        critics=critics,
        config=enhanced_config,
        enable_congestion_tracking=True
    )
    
    print("Starting enhanced multi-marginal perturbation with:")
    print(f"  - Number of evidence domains: {len(evidence_list)}")
    print(f"  - Multi-domain mass conservation: {enhanced_config['mass_conservation_weight']}")
    print(f"  - Domain-specific congestion tracking: Enabled")
    print(f"  - Enhanced theoretical validation: {enhanced_config['theoretical_validation']}")
    
    # Perform perturbation
    perturbed_generator = perturber.perturb(epochs=25, verbose=True)
    
    # Get multi-domain statistics
    if hasattr(perturber, 'multi_congestion_trackers'):
        print("\nMulti-Domain Congestion Statistics:")
        for i, tracker in enumerate(perturber.multi_congestion_trackers):
            if tracker.history['congestion_cost']:
                avg_congestion = tracker.get_average_congestion()
                print(f"  Domain {i+1} avg congestion: {avg_congestion:.4f}")
    
    print("Enhanced evidence-based perturbation completed!\n")
    return perturbed_generator


def demonstrate_theoretical_validation():
    """
    Demonstrate the new theoretical validation capabilities.
    """
    print("=== Theoretical Validation Demonstration ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Generate sample data
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
    target_samples = sample_target_data(batch_size=100, device=device)
    
    # Generate samples from current generator
    noise = torch.randn(100, 2, device=device)
    with torch.no_grad():
        gen_samples = generator(noise)
    
    # Compute spatial density
    from weight_perturbation.congestion import compute_spatial_density, compute_traffic_flow
    density_info = compute_spatial_density(gen_samples)
    sigma = density_info['density_at_samples']
    
    # Create a simple critic for flow computation
    critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=64).to(device)
    
    # Compute traffic flow
    flow_info = compute_traffic_flow(critic, generator, noise, sigma, lambda_param=1.0)
    
    # Perform theoretical validation
    validation_results = validate_theoretical_consistency(
        flow_info, density_info, gen_samples, target_samples
    )
    
    print("Theoretical Validation Results:")
    for key, value in validation_results.items():
        print(f"  {key}: {value:.4f}")
    
    # Demonstrate mass conservation enforcement
    print("\nMass Conservation Demonstration:")
    target_density_info = compute_spatial_density(target_samples)
    target_density = target_density_info['density_at_samples']
    
    conservation_result = enforce_mass_conservation(
        flow_info['traffic_flow'],
        target_density[:gen_samples.shape[0]],  # Match sizes
        sigma,
        gen_samples,
        lagrange_multiplier=0.1
    )
    
    print(f"  Mass conservation error: {conservation_result['mass_conservation_error'].item():.6f}")
    print(f"  Flow divergence norm: {conservation_result['divergence'].norm().item():.6f}")
    
    print("Theoretical validation demonstration completed!\n")


def demonstrate_enhanced_loss_function():
    """
    Demonstrate the enhanced congestion-aware loss function.
    """
    print("=== Enhanced Loss Function Demonstration ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create enhanced loss function with all new features
    loss_function = CongestionAwareLossFunction(
        lambda_congestion=1.0,
        lambda_sobolev=0.1,
        lambda_entropy=0.012,
        enable_mass_conservation=True,      # NEW: Mass conservation
        enable_theoretical_validation=True  # NEW: Theoretical validation
    )
    
    # Create test models and data
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
    critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=64).to(device)
    target_samples = sample_target_data(batch_size=100, device=device)
    noise_samples = torch.randn(100, 2, device=device)
    
    # Compute enhanced loss with all features
    loss, grads, congestion_info = loss_function.compute_target_given_loss(
        generator, target_samples, noise_samples, critic
    )
    
    print("Enhanced Loss Function Results:")
    print(f"  Total loss: {loss.item():.4f}")
    print(f"  Gradient norm: {grads.norm().item():.4f}")
    
    if congestion_info:
        print("  Congestion Information:")
        for key, value in congestion_info.items():
            if isinstance(value, torch.Tensor) and value.dim() == 0:
                print(f"    {key}: {value.item():.4f}")
            elif isinstance(value, (int, float)):
                print(f"    {key}: {value:.4f}")
    
    # Update and get statistics
    loss_function.update_congestion_history(congestion_info)
    stats = loss_function.get_congestion_statistics()
    
    print("  Enhanced Statistics:")
    for key, value in stats.items():
        if 'recent' in key or 'avg' in key:
            print(f"    {key}: {value:.4f}")
    
    print("Enhanced loss function demonstration completed!\n")


def main():
    """
    Main demonstration function showing all enhancements.
    """
    print("Weight Perturbation Library Enhanced Features Demonstration")
    print("=" * 60)
    
    # Check if enhanced features are available
    if not check_theoretical_support():
        print("Enhanced theoretical components are not available.")
        return
    
    print(get_enhancement_summary())
    print("\n" + "=" * 60)
    
    # Demonstrate all enhanced features
    try:
        demonstrate_theoretical_validation()
        demonstrate_enhanced_loss_function()
        demonstrate_enhanced_target_given_perturbation()
        demonstrate_enhanced_evidence_based_perturbation()
        
        print("=" * 60)
        print("All enhanced features demonstrated successfully!")
        print("\nKey Improvements:")
        print("  ✓ Mass conservation enforcement with Lagrangian multipliers")
        print("  ✓ Theoretical validation and consistency checking")
        print("  ✓ Enhanced congestion scaling with H''(x,i) derivatives")
        print("  ✓ Reduced over-conservative constraints and bounds")
        print("  ✓ More responsive adaptive mechanisms")
        print("  ✓ Better integration between theoretical components")
        
    except Exception as e:
        print(f"Demonstration failed with error: {e}")
        print("This might be due to missing dependencies or computational issues.")


if __name__ == "__main__":
    main()