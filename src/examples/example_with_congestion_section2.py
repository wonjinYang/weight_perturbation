"""
Section 2 example: Target-given perturbation with congestion tracking and traffic flow visualization.

This example demonstrates the theoretical integration:
- Spatial density estimation σ(x) with mass conservation
- Traffic flow computation w_Q with theoretical validation
- Traffic intensity i_Q visualization with H''(x,i) scaling  
- Congestion tracking with theoretical consistency checks
- Traffic flow vector field visualization with mass conservation
- Sobolev regularization with adaptive weights
- Theoretical validation at each step
"""

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Try to import theoretical components
try:
    from weight_perturbation import (
        Generator,
        sample_real_data,
        sample_target_data,
        pretrain_wgan_gp,
        # Theoretical components
        CTWeightPerturberTargetGiven,
        SobolevConstrainedCritic,
        CongestionTracker,
        compute_spatial_density,
        compute_traffic_flow,
        compute_convergence_metrics,
        parameters_to_vector,
        set_seed,
        compute_device,
        check_theoretical_support,
        # Components
        validate_theoretical_consistency,
        enforce_mass_conservation,
        get_congestion_second_derivative,
        CongestionAwareLossFunction,
        MassConservationSobolevRegularizer
    )
    THEORETICAL_AVAILABLE = True
except ImportError as e:
    print(f"Theoretical components not available: {e}")
    print("This example requires the theoretical components to run.")
    exit(1)

class TrafficFlowVisualizer:
    """
    Traffic flow visualization class with theoretical validation for monitoring.
    
    Now includes:
    - Mass conservation visualization
    - Theoretical consistency tracking
    - H''(x,i) second derivative analysis
    - Sobolev regularization effects
    """
    
    def __init__(self, figsize=(16, 12), save_dir="test_results/plots/traffic_flow_plots"):
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.step_data = []
        
        # Initialize theoretical validation tracker
        self.theoretical_consistency_history = []
        self.mass_conservation_history = []
        
    def visualize_traffic_flow_step(self, step, generator, critic, target_samples, 
                                   noise_samples, congestion_info=None, save=True):
        """
        Visualize traffic flow for a single perturbation step with theoretical validation.
        
        Args:
            step (int): Current perturbation step.
            generator: Current generator model.
            critic: Sobolev-constrained critic model.
            target_samples: Target distribution samples.
            noise_samples: Noise input samples.
            congestion_info: Congestion information with theoretical metrics.
            save (bool): Whether to save the plot.
        """
        device = next(generator.parameters()).device
        
        # Generate samples from current generator
        with torch.no_grad():
            gen_samples = generator(noise_samples)
        
        # Compute spatial density with better bandwidth
        density_info = compute_spatial_density(gen_samples, bandwidth=0.15)
        sigma = density_info['density_at_samples']
        
        # Compute traffic flow with theoretical validation
        flow_info = compute_traffic_flow(
            critic, generator, noise_samples, sigma, lambda_param=1.0
        )
        
        # Perform theoretical validation
        validation_results = validate_theoretical_consistency(
            flow_info, density_info, gen_samples, target_samples
        )
        
        # Enforce mass conservation if target samples available
        target_density_info = compute_spatial_density(target_samples, bandwidth=0.15)
        target_density = target_density_info['density_at_samples']
        
        # Resample target density to match generator output size
        if target_density.shape[0] != gen_samples.shape[0]:
            indices = torch.randperm(target_density.shape[0])[:gen_samples.shape[0]]
            target_density = target_density[indices]
        
        mass_conservation_result = enforce_mass_conservation(
            flow_info['traffic_flow'],
            target_density,
            sigma,
            gen_samples,
            lagrange_multiplier=0.1
        )
        
        # Compute H''(x,i) second derivatives for theoretical scaling
        h_second_derivatives = get_congestion_second_derivative(
            flow_info['traffic_intensity'], 
            sigma, 
            lambda_param=1.0
        )
        
        # Store step data with theoretical metrics
        step_data = {
            'step': step,
            'samples': gen_samples.detach().cpu().numpy(),
            'target': target_samples.detach().cpu().numpy(),
            'flow': flow_info['traffic_flow'].detach().cpu().numpy(),
            'intensity': flow_info['traffic_intensity'].detach().cpu().numpy(),
            'density': sigma.detach().cpu().numpy(),
            'gradient_norm': flow_info['gradient_norm'].detach().cpu().numpy(),
            # Theoretical metrics
            'theoretical_consistency': validation_results.get('overall_consistency', 0.0),
            'mass_conservation_error': mass_conservation_result['mass_conservation_error'].item(),
            'h_second_derivatives': h_second_derivatives.detach().cpu().numpy(),
            'validation_results': validation_results,
            'corrected_flow': mass_conservation_result['corrected_flow'].detach().cpu().numpy()
        }
        self.step_data.append(step_data)
        
        # Track theoretical metrics over time
        self.theoretical_consistency_history.append(validation_results.get('overall_consistency', 0.0))
        self.mass_conservation_history.append(mass_conservation_result['mass_conservation_error'].item())
        
        # Create visualization
        fig, axes = plt.subplots(3, 3, figsize=self.figsize)
        fig.suptitle(f'Traffic Flow Analysis with Theoretical Validation - Step {step}', 
                    fontsize=16, fontweight='bold')
        
        # Plot 1: Generated samples with target overlay
        ax1 = axes[0, 0]
        ax1.scatter(gen_samples[:, 0].cpu(), gen_samples[:, 1].cpu(), 
                   c='blue', alpha=0.6, s=30, label='Generated')
        ax1.scatter(target_samples[:, 0].cpu(), target_samples[:, 1].cpu(),
                   c='red', alpha=0.6, s=30, label='Target')
        ax1.set_title('Generated vs Target Samples')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_xlabel('X₁')
        ax1.set_ylabel('X₂')
        
        # Plot 2: Traffic flow vector field
        ax2 = axes[0, 1]
        samples_np = gen_samples.detach().cpu().numpy()
        flow_np = flow_info['traffic_flow'].detach().cpu().numpy()
        intensity_np = flow_info['traffic_intensity'].detach().cpu().numpy()
        
        # Create quiver plot for traffic flow vectors
        quiver = ax2.quiver(
            samples_np[:, 0], samples_np[:, 1],
            flow_np[:, 0], flow_np[:, 1],
            intensity_np,
            cmap='viridis', scale=1.0, scale_units='xy', angles='xy',
            width=0.003, alpha=0.8
        )
        
        # Add colorbar for intensity
        cbar = plt.colorbar(quiver, ax=ax2, shrink=0.8)
        cbar.set_label('Traffic Intensity |w_Q|')
        
        ax2.set_title('Traffic Flow Vector Field w_Q(x)')
        ax2.set_xlabel('X₁')
        ax2.set_ylabel('X₂')
        ax2.grid(True, alpha=0.3)
        
        # Add target samples as reference
        ax2.scatter(target_samples[:, 0].cpu(), target_samples[:, 1].cpu(), 
                   c='red', alpha=0.3, s=20, marker='x', label='Target')
        ax2.legend()
        
        # Plot 3: Mass-conserved flow field
        ax3 = axes[0, 2]
        corrected_flow_np = mass_conservation_result['corrected_flow'].detach().cpu().numpy()
        
        quiver3 = ax3.quiver(
            samples_np[:, 0], samples_np[:, 1],
            corrected_flow_np[:, 0], corrected_flow_np[:, 1],
            np.linalg.norm(corrected_flow_np, axis=1),
            cmap='plasma', scale=1.0, scale_units='xy', angles='xy',
            width=0.003, alpha=0.8
        )
        
        cbar3 = plt.colorbar(quiver3, ax=ax3, shrink=0.8)
        cbar3.set_label('Corrected Flow Magnitude')
        
        ax3.set_title('Mass-Conserved Flow Field')
        ax3.set_xlabel('X₁')
        ax3.set_ylabel('X₂')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Traffic intensity heatmap with H''(x,i) overlay
        ax4 = axes[1, 0]
        
        scatter = ax4.scatter(samples_np[:, 0], samples_np[:, 1], 
                            c=intensity_np, cmap='plasma', s=50, alpha=0.7)
        
        cbar4 = plt.colorbar(scatter, ax=ax4, shrink=0.8)
        cbar4.set_label('Traffic Intensity i_Q(x)')
        
        # Overlay H''(x,i) as contours
        h_second_np = h_second_derivatives.detach().cpu().numpy()
        ax4.scatter(samples_np[:, 0], samples_np[:, 1], 
                   s=h_second_np*100, facecolors='none', edgecolors='white', alpha=0.5)
        
        ax4.set_title('Traffic Intensity + H\'\'(x,i) Scaling')
        ax4.set_xlabel('X₁')
        ax4.set_ylabel('X₂')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Spatial density comparison
        ax5 = axes[1, 1]
        
        # Current density
        density_scatter = ax5.scatter(samples_np[:, 0], samples_np[:, 1], 
                                    c=sigma.detach().cpu().numpy(), cmap='coolwarm', 
                                    s=50, alpha=0.7, label='Current σ(x)')
        
        # Target density overlay
        ax5.scatter(target_samples[:, 0].cpu(), target_samples[:, 1].cpu(),
                   c=target_density.detach().cpu().numpy(), cmap='Reds',
                   s=30, alpha=0.5, marker='^', label='Target σ_target')
        
        cbar5 = plt.colorbar(density_scatter, ax=ax5, shrink=0.8)
        cbar5.set_label('Spatial Density σ(x)')
        ax5.set_title('Spatial Density: Current vs Target')
        ax5.set_xlabel('X₁')
        ax5.set_ylabel('X₂')
        ax5.legend()
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Theoretical validation metrics
        ax6 = axes[1, 2]
        
        # Create bar plot of validation metrics
        validation_metrics = [
            validation_results.get('intensity_consistency', 0),
            validation_results.get('density_positive', 0),
            validation_results.get('gradient_close_to_one', 0),
            validation_results.get('overall_consistency', 0)
        ]
        
        metric_names = ['Intensity\nConsistency', 'Density\nPositive', 
                       'Gradient\nBounds', 'Overall\nConsistency']
        
        bars = ax6.bar(metric_names, validation_metrics, 
                      color=['skyblue', 'lightgreen', 'orange', 'coral'], alpha=0.7)
        
        ax6.set_title('Theoretical Validation Metrics')
        ax6.set_ylabel('Validation Score')
        ax6.set_ylim(0, 1.1)
        ax6.grid(True, alpha=0.3)
        
        # Add values on bars
        for bar, value in zip(bars, validation_metrics):
            height = bar.get_height()
            ax6.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom', fontsize=9)
        
        # Plot 7: Mass conservation analysis
        ax7 = axes[2, 0]
        
        # Plot divergence field
        divergence = mass_conservation_result['divergence'].detach().cpu().numpy()
        div_scatter = ax7.scatter(samples_np[:, 0], samples_np[:, 1],
                                 c=divergence, cmap='RdBu_r', s=50, alpha=0.7)
        
        cbar7 = plt.colorbar(div_scatter, ax=ax7, shrink=0.8)
        cbar7.set_label('Flow Divergence ∇·w')
        
        ax7.set_title('Mass Conservation: Flow Divergence')
        ax7.set_xlabel('X₁')
        ax7.set_ylabel('X₂')
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Theoretical consistency over time
        ax8 = axes[2, 1]
        
        if len(self.theoretical_consistency_history) > 1:
            ax8.plot(range(len(self.theoretical_consistency_history)), 
                    self.theoretical_consistency_history, 'b-', linewidth=2, label='Consistency')
            ax8.plot(range(len(self.mass_conservation_history)), 
                    self.mass_conservation_history, 'r--', linewidth=2, label='Mass Error')
        
        ax8.set_title('Theoretical Metrics Evolution')
        ax8.set_xlabel('Step')
        ax8.set_ylabel('Metric Value')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # Plot 9: Statistics text
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Statistics
        stats_text = f"""Traffic Flow Statistics:

Flow Characteristics:
• Mean Intensity: {intensity_np.mean():.6f}
• Max Intensity: {intensity_np.max():.6f}
• Flow Magnitude: {np.linalg.norm(flow_np, axis=1).mean():.6f}

Theoretical Validation:
• Overall Consistency: {validation_results.get('overall_consistency', 0):.4f}
• Mass Conservation Error: {mass_conservation_result['mass_conservation_error'].item():.6f}
• Mean H''(x,i): {h_second_np.mean():.6f}

Density Properties:
• Mean Density: {sigma.mean().item():.6f}
• Density Variance: {sigma.var().item():.6f}
• Min Density: {validation_results.get('min_density', 0):.6f}

Gradient Properties:
• Mean Gradient Norm: {validation_results.get('mean_gradient_norm', 0):.6f}
• Coverage Error: {validation_results.get('coverage_error', 0):.6f}"""
        
        ax9.text(0.02, 0.98, stats_text, transform=ax9.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9, fontfamily='monospace')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"traffic_flow_step_{step:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved traffic flow visualization: {save_path}")
        
        plt.close()
        
        return step_data
    
    def create_traffic_flow_summary(self, save=True):
        """
        Create summary visualization with theoretical validation evolution.
        """
        if not self.step_data:
            print("No step data available for summary.")
            return
        
        fig, axes = plt.subplots(3, 3, figsize=(20, 15))
        fig.suptitle('Traffic Flow Evolution Summary with Theoretical Validation', 
                    fontsize=16, fontweight='bold')
        
        steps = [data['step'] for data in self.step_data]
        
        # Extract metrics over time
        mean_intensities = [data['intensity'].mean() for data in self.step_data]
        max_intensities = [data['intensity'].max() for data in self.step_data]
        mean_densities = [data['density'].mean() for data in self.step_data]
        theoretical_consistencies = [data['theoretical_consistency'] for data in self.step_data]
        mass_conservation_errors = [data['mass_conservation_error'] for data in self.step_data]
        h_second_means = [data['h_second_derivatives'].mean() for data in self.step_data]
        
        # Plot 1: Intensity evolution
        ax1 = axes[0, 0]
        ax1.plot(steps, mean_intensities, 'b-', linewidth=2, label='Mean Intensity')
        ax1.plot(steps, max_intensities, 'r--', linewidth=2, label='Max Intensity')
        ax1.set_title('Traffic Intensity Evolution')
        ax1.set_xlabel('Perturbation Step')
        ax1.set_ylabel('Traffic Intensity |w_Q|')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Theoretical consistency evolution
        ax2 = axes[0, 1]
        ax2.plot(steps, theoretical_consistencies, 'g-', linewidth=2, label='Consistency Score')
        ax2.axhline(y=0.8, color='orange', linestyle='--', alpha=0.7, label='Target (0.8)')
        ax2.set_title('Theoretical Consistency Evolution')
        ax2.set_xlabel('Perturbation Step')
        ax2.set_ylabel('Consistency Score')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 1.1)
        
        # Plot 3: Mass conservation evolution
        ax3 = axes[0, 2]
        ax3.plot(steps, mass_conservation_errors, 'm-', linewidth=2, label='Mass Error')
        ax3.axhline(y=0.05, color='red', linestyle='--', alpha=0.7, label='Tolerance (0.05)')
        ax3.set_title('Mass Conservation Error Evolution')
        ax3.set_xlabel('Perturbation Step')
        ax3.set_ylabel('Mass Conservation Error')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_yscale('log')
        
        # Plot 4: H''(x,i) second derivative evolution
        ax4 = axes[1, 0]
        ax4.plot(steps, h_second_means, 'purple', linewidth=2, label='Mean H\'\'(x,i)')
        ax4.set_title('Congestion Second Derivative H\'\'(x,i) Evolution')
        ax4.set_xlabel('Perturbation Step')
        ax4.set_ylabel('Mean H\'\'(x,i)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Validation metrics heatmap
        ax5 = axes[1, 1]
        
        # Create heatmap of validation metrics over time
        validation_matrix = []
        for data in self.step_data:
            vr = data['validation_results']
            validation_matrix.append([
                vr.get('intensity_consistency', 0),
                vr.get('density_positive', 0),
                vr.get('gradient_close_to_one', 0),
                vr.get('overall_consistency', 0)
            ])
        
        validation_matrix = np.array(validation_matrix).T
        
        im = ax5.imshow(validation_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax5.set_title('Validation Metrics Heatmap')
        ax5.set_xlabel('Perturbation Step')
        ax5.set_ylabel('Metric Type')
        ax5.set_yticks(range(4))
        ax5.set_yticklabels(['Intensity', 'Density+', 'Gradient', 'Overall'])
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax5, shrink=0.8)
        cbar.set_label('Validation Score')
        
        # Plot 6: Flow field comparison (initial vs final)
        ax6 = axes[1, 2]
        
        if len(self.step_data) >= 2:
            initial_data = self.step_data[0]
            final_data = self.step_data[-1]
            
            # Plot initial and final samples
            ax6.scatter(initial_data['samples'][:, 0], initial_data['samples'][:, 1],
                       c='lightblue', alpha=0.5, s=20, label=f'Initial (Step {initial_data["step"]})')
            ax6.scatter(final_data['samples'][:, 0], final_data['samples'][:, 1],
                       c='darkblue', alpha=0.7, s=20, label=f'Final (Step {final_data["step"]})')
            ax6.scatter(final_data['target'][:, 0], final_data['target'][:, 1],
                       c='red', alpha=0.5, s=20, label='Target')
            
            ax6.set_title('Initial vs Final Sample Distribution')
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
        # Plot 7: Density evolution
        ax7 = axes[2, 0]
        ax7.plot(steps, mean_densities, 'orange', linewidth=2, label='Mean Density')
        ax7.set_title('Spatial Density Evolution')
        ax7.set_xlabel('Perturbation Step')
        ax7.set_ylabel('Mean Spatial Density σ(x)')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # Plot 8: Convergence quality score
        ax8 = axes[2, 1]
        
        # Compute overall quality score
        quality_scores = []
        for i, data in enumerate(self.step_data):
            # Combine multiple metrics into quality score
            consistency = data['theoretical_consistency']
            mass_error = data['mass_conservation_error']
            intensity_stability = 1.0 / (1.0 + data['intensity'].std())
            
            quality = (consistency + (1.0 / (1.0 + mass_error)) + intensity_stability) / 3.0
            quality_scores.append(quality)
        
        ax8.plot(steps, quality_scores, 'green', linewidth=2, marker='o', label='Quality Score')
        ax8.set_title('Overall Convergence Quality')
        ax8.set_xlabel('Perturbation Step')
        ax8.set_ylabel('Quality Score')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        ax8.set_ylim(0, 1.1)
        
        # Plot 9: Final statistics summary
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        if self.step_data:
            final_data = self.step_data[-1]
            initial_data = self.step_data[0]
            
            summary_text = f"""Traffic Flow Summary:

Initial → Final Comparison:
• Theoretical Consistency: {initial_data['theoretical_consistency']:.3f} → {final_data['theoretical_consistency']:.3f}
• Mass Conservation Error: {initial_data['mass_conservation_error']:.6f} → {final_data['mass_conservation_error']:.6f}
• Mean Traffic Intensity: {initial_data['intensity'].mean():.6f} → {final_data['intensity'].mean():.6f}

Best Achieved Metrics:
• Best Consistency: {max(theoretical_consistencies):.4f}
• Min Mass Error: {min(mass_conservation_errors):.6f}
• Max Quality Score: {max(quality_scores):.4f}

Theoretical Validation:
• Steps with Good Consistency (>0.8): {sum(1 for x in theoretical_consistencies if x > 0.8)}/{len(steps)}
• Steps with Low Mass Error (<0.05): {sum(1 for x in mass_conservation_errors if x < 0.05)}/{len(steps)}

Total Steps Analyzed: {len(steps)}"""
            
            ax9.text(0.02, 0.98, summary_text, transform=ax9.transAxes,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8),
                    fontsize=10, fontfamily='monospace')
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "traffic_flow_summary.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved traffic flow summary: {save_path}")
        
        return {
            'steps': steps,
            'mean_intensities': mean_intensities,
            'max_intensities': max_intensities,
            'mean_densities': mean_densities,
            'theoretical_consistencies': theoretical_consistencies,
            'mass_conservation_errors': mass_conservation_errors,
            'h_second_means': h_second_means,
            'quality_scores': quality_scores
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Section 2: Target-Given Perturbation with Theoretical Integration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--pretrain_epochs", type=int, default=300, help="Pretraining epochs")
    parser.add_argument("--perturb_steps", type=int, default=50, help="Perturbation steps")
    parser.add_argument("--batch_size", type=int, default=96, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=600, help="Evaluation batch size")
    parser.add_argument("--eta_init", type=float, default=0.08, help="Initial learning rate")
    parser.add_argument("--enable_congestion", action="store_true", default=True, help="Enable congestion tracking")
    parser.add_argument("--use_sobolev_critic", action="store_true", default=True, help="Use Sobolev-constrained critic")
    parser.add_argument("--enable_mass_conservation", action="store_true", default=True, help="Enable mass conservation enforcement")
    parser.add_argument("--enable_theoretical_validation", action="store_true", default=True, help="Enable theoretical validation")
    parser.add_argument("--visualize_every", type=int, default=5, help="Visualize traffic flow every N steps")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    return parser.parse_args()


def run_section2_example():
    """Run Section 2 example with theoretical integration."""
    args = parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else compute_device()
    
    print("="*80)
    print("SECTION 2: TARGET-GIVEN PERTURBATION WITH THEORETICAL INTEGRATION")
    print("="*80)
    print(f"Device: {device}")
    print(f"Theoretical components available: {THEORETICAL_AVAILABLE}")
    
    if not THEORETICAL_AVAILABLE:
        print("This example requires theoretical components. Exiting.")
        return
    
    # Check theoretical support
    support_ok = check_theoretical_support()
    if not support_ok:
        print("Warning: Some theoretical components may not work correctly.")
    
    # Create models with architecture
    print("\nInitializing models...")
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)
    
    if args.use_sobolev_critic:
        # Use Sobolev-constrained critic with mass conservation
        critic = SobolevConstrainedCritic(
            data_dim=2, hidden_dim=256,
            use_spectral_norm=True, 
            lambda_sobolev=0.1,
            sobolev_bound=5.0
        ).to(device)
        print("Using Sobolev-constrained critic with mass conservation integration")
    else:
        from weight_perturbation import Critic
        critic = Critic(data_dim=2, hidden_dim=256).to(device)
        print("Using standard critic")
    
    # Pretrain models
    print(f"\nPretraining for {args.pretrain_epochs} epochs...")
    means = [torch.tensor([0.0, 0.0], device=device, dtype=torch.float32)]
    real_sampler = lambda bs: sample_real_data(
        bs, means=means, std=0.7, device=device 
    )
    
    pretrained_gen, pretrained_critic = pretrain_wgan_gp(
        generator, critic, real_sampler,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=2e-4,
        gp_lambda=0.,
        betas=(0., 0.95),
        device=device,
        verbose=args.verbose
    )
    
    # Create target data with clear shift for visualization
    print("\nCreating target distribution...")
    target_samples = sample_target_data(
        batch_size=args.eval_batch_size,
        shift=[2.5, 2.5],  # Clear shift for visualization
        device=device
    )
    
    # Initialize traffic flow visualizer
    visualizer = TrafficFlowVisualizer(
        figsize=(16, 12),
        save_dir=f"test_results/plots/section2_traffic_flow_seed_{args.seed}"
    )
    
    # Create perturber with theoretical integration
    print(f"\nInitializing perturber with theoretical validation...")
    print(f"  - Congestion tracking: {args.enable_congestion}")
    print(f"  - Mass conservation enforcement: {args.enable_mass_conservation}")
    print(f"  - Theoretical validation: {args.enable_theoretical_validation}")
    
    perturber = CTWeightPerturberTargetGiven(
        pretrained_gen, target_samples,
        critic=pretrained_critic,
        enable_congestion_tracking=args.enable_congestion
    )
    
    # Configuration with theoretical parameters
    perturber.config.update({
        'eta_init': args.eta_init,
        'eta_min': 1e-6,
        'eta_max': 0.8,
        'eta_decay_factor': 0.9,
        'eta_boost_factor': 1.05,
        'clip_norm': 0.6,
        'momentum': 0.85,
        'patience': 15,
        'rollback_patience': 10,
        'lambda_entropy': 0.012,
        'lambda_virtual': 0.8,
        'lambda_multi': 1.0,
        'lambda_congestion': 1.0,
        'lambda_sobolev': 0.1,
        'eval_batch_size': args.eval_batch_size,
        # Parameters
        'mass_conservation_weight': 0.1,
        'theoretical_validation': args.enable_theoretical_validation,
        'congestion_threshold': 0.15,
        'improvement_threshold': 1e-5
    })

    # Perturbation loop with theoretical validation
    print(f"\nStarting perturbation with theoretical analysis...")
    print(f"Will visualize every {args.visualize_every} steps")
    
    try:
        data_dim = target_samples.shape[1]
        pert_gen = perturber._create_generator_copy(data_dim)
        pretrained_critic.eval()

        # Initialize perturbation state
        theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
        delta_theta_prev = torch.zeros_like(theta_prev)
        eta = args.eta_init
        best_vec = None
        best_w2 = float('inf')
        loss_hist = []
        w2_hist = []
        no_improvement_count = 0
        consecutive_rollbacks = 0
        
        # Initialize loss function with theoretical integration
        loss_function = CongestionAwareLossFunction(
            lambda_congestion=perturber.config.get('lambda_congestion', 1.0),
            lambda_sobolev=perturber.config.get('lambda_sobolev', 0.1),
            lambda_entropy=perturber.config.get('lambda_entropy', 0.012),
            enable_mass_conservation=args.enable_mass_conservation,
            enable_theoretical_validation=args.enable_theoretical_validation
        )

        # Main perturbation loop with theoretical validation
        for step in range(args.perturb_steps):
            pert_gen.train()

            # Generate noise for this step
            noise_samples = torch.randn(args.eval_batch_size, 2, device=device)
            
            # Compute loss and gradients with theoretical integration
            if args.enable_congestion:
                loss, grads, congestion_info = loss_function.compute_target_given_loss(
                    pert_gen, target_samples, noise_samples, pretrained_critic
                )
                
                # Update congestion tracker
                if congestion_info:
                    loss_function.update_congestion_history(congestion_info)
                
                # Visualization with theoretical validation at specified intervals
                if step % args.visualize_every == 0:
                    print(f"\n--- Visualization with Theoretical Analysis at Step {step} ---")
                    step_data = visualizer.visualize_traffic_flow_step(
                        step, pert_gen, pretrained_critic, target_samples,
                        noise_samples, congestion_info, save=args.save_plots
                    )
                    
                    # Print step statistics
                    print(f"Step {step} Statistics:")
                    print(f"  Mean traffic intensity: {step_data['intensity'].mean():.6f}")
                    print(f"  Max traffic intensity: {step_data['intensity'].max():.6f}")
                    print(f"  Mean spatial density: {step_data['density'].mean():.6f}")
                    print(f"  Theoretical consistency: {step_data['theoretical_consistency']:.6f}")
                    print(f"  Mass conservation error: {step_data['mass_conservation_error']:.6f}")
                    print(f"  Mean H''(x,i): {step_data['h_second_derivatives'].mean():.6f}")
                
                # Delta_theta computation with theoretical justification
                with torch.no_grad():
                    gen_samples = pert_gen(noise_samples)
                
                delta_theta = perturber._compute_delta_theta_with_congestion(
                    grads, eta, perturber.config.get('clip_norm', 0.6), 
                    perturber.config.get('momentum', 0.85), delta_theta_prev, 
                    congestion_info, target_samples, gen_samples
                )
            else:
                loss, grads = perturber._compute_loss_and_grad(pert_gen)
                delta_theta = perturber._compute_delta_theta(
                    grads, eta, perturber.config.get('clip_norm', 0.6), 
                    perturber.config.get('momentum', 0.85), delta_theta_prev
                )
                
                # Basic visualization without features
                if step % args.visualize_every == 0:
                    step_data = visualizer.visualize_traffic_flow_step(
                        step, pert_gen, pretrained_critic, target_samples,
                        noise_samples, save=args.save_plots
                    )
            
            # Apply parameter update
            theta_prev = perturber._apply_parameter_update(pert_gen, theta_prev, delta_theta)
            delta_theta_prev = delta_theta.clone()
            
            # Validation and adaptation
            w2_pert, improvement = perturber._validate_and_adapt(
                pert_gen, eta, w2_hist, perturber.config.get('patience', 15), args.verbose, step
            )
            
            # Update best state
            best_w2, best_vec = perturber._update_best_state(w2_pert, pert_gen, best_w2, best_vec)
            
            # Learning rate adaptation
            eta, no_improvement_count = perturber._adapt_learning_rate(
                eta, improvement, step, no_improvement_count, loss_hist
            )

            # Rollback condition checking with theoretical validation
            if perturber._check_rollback_condition_with_congestion(w2_hist, no_improvement_count):
                if args.verbose:
                    print(f"Rollback triggered at step {step}")
                
                perturber._restore_best_state(pert_gen, best_vec)
                # Reset parameters after rollback
                eta = max(eta * 0.95, perturber.config.get('eta_min', 1e-6))
                no_improvement_count = 0
                consecutive_rollbacks += 1
                delta_theta_prev = torch.zeros_like(delta_theta_prev)
                
                # Rollback handling
                if consecutive_rollbacks >= 3:
                    if args.verbose:
                        print(f"Too many rollbacks, stopping early")
                    break
            else:
                consecutive_rollbacks = 0

            # Progress logging
            if args.verbose:
                log_msg = f"[{step:2d}] W2={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                if args.enable_congestion and 'congestion_info' in locals() and congestion_info:
                    log_msg += f" Congestion={congestion_info.get('congestion_cost', torch.tensor(0)).item():.4f}"
                    if 'theoretical_consistency' in congestion_info:
                        log_msg += f" Consistency={congestion_info['theoretical_consistency']:.3f}"
                print(log_msg)
            
            # Record loss for adaptation
            loss_hist.append(loss.item())
        
        # Create traffic flow summary
        print("\nCreating traffic flow evolution summary...")
        summary_data = visualizer.create_traffic_flow_summary(save=args.save_plots)
        
        # Final evaluation
        print("\n" + "="*60)
        print("FINAL EVALUATION")
        print("="*60)
        
        noise_eval = torch.randn(args.eval_batch_size, 2, device=device)
        with torch.no_grad():
            original_samples = pretrained_gen(noise_eval)
            final_samples = pert_gen(noise_eval)
        
        # Compute convergence metrics with theoretical validation
        final_metrics = compute_convergence_metrics(
            pert_gen, target_samples, noise_eval, 
            include_theoretical_metrics=True
        )
        print("Convergence Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # Congestion statistics
        if args.enable_congestion:
            final_congestion_stats = loss_function.get_congestion_statistics()
            print("\nCongestion Statistics:")
            for key, value in final_congestion_stats.items():
                print(f"  {key}: {value:.6f}")
        
        return {
            'perturbed_generator': pert_gen,
            'original_samples': original_samples,
            'final_samples': final_samples,
            'target_samples': target_samples,
            'convergence_metrics': final_metrics,
            'traffic_flow_data': visualizer.step_data,
            'summary_data': summary_data,
            'loss_function': loss_function
        }
        
    except Exception as e:
        print(f"Error during perturbation: {e}")
        return None


if __name__ == "__main__":
    results = run_section2_example()
    
    if results:
        print("\n" + "="*60)
        print("SECTION 2 DEMONSTRATION COMPLETED SUCCESSFULLY")
        print("="*60)
        print("Results:")
        print("  ✓ Traffic flow visualization generated")
        print("  ✓ Theoretical validation tracking completed")
        print("  ✓ Mass conservation enforcement applied")
        print("  ✓ Spatial density analysis performed")
        print("  ✓ H''(x,i) second derivative tracking completed")
        print("  ✓ Flow vector field evolution captured")
        print("  ✓ Summary plots created")
        
        # Print key insights
        traffic_data = results['traffic_flow_data']
        if traffic_data:
            initial_consistency = traffic_data[0]['theoretical_consistency']
            final_consistency = traffic_data[-1]['theoretical_consistency']
            initial_mass_error = traffic_data[0]['mass_conservation_error']
            final_mass_error = traffic_data[-1]['mass_conservation_error']
            
            print(f"\nKey Insights:")
            print(f"  Theoretical consistency: {initial_consistency:.6f} → {final_consistency:.6f}")
            print(f"  Mass conservation error: {initial_mass_error:.6f} → {final_mass_error:.6f}")
            print(f"  Total steps with theoretical validation: {len(traffic_data)}")
            
        print(f"\nPlots saved in: section2_traffic_flow_seed_{parse_args().seed}/")
    else:
        print("Section 2 demonstration failed.")
