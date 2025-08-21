"""
Section 2 example: Target-given perturbation with congestion tracking and traffic flow visualization.

This example demonstrates:
- Spatial density estimation σ(x)
- Traffic flow computation w_Q with vector directions
- Traffic intensity i_Q visualization
- Step-by-step congestion tracking
- Real-time traffic flow vector field visualization
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
        # Advanced components
        CTWeightPerturberTargetGiven,
        SobolevConstrainedCritic,
        CongestionTracker,
        compute_spatial_density,
        compute_traffic_flow,
        compute_convergence_metrics,
        parameters_to_vector,
        set_seed,
        compute_device,
        check_theoretical_support
    )
    THEORETICAL_AVAILABLE = True
except ImportError as e:
    print(f"Theoretical components not available: {e}")
    print("This example requires the theoretical components to run.")
    exit(1)


class TrafficFlowVisualizer:
    """
    Traffic flow visualization class for real-time monitoring during perturbation.
    """
    
    def __init__(self, figsize=(15, 10), save_dir="test_results/plots/traffic_flow_plots"):
        self.figsize = figsize
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
        self.step_data = []
        
    def visualize_traffic_flow_step(self, step, generator, critic, target_samples, 
                                   noise_samples, congestion_info=None, save=True):
        """
        Visualize traffic flow for a single perturbation step.
        
        Args:
            step (int): Current perturbation step.
            generator: Current generator model.
            critic: Critic model for flow computation.
            target_samples: Target distribution samples.
            noise_samples: Noise input samples.
            congestion_info: Optional congestion information.
            save (bool): Whether to save the plot.
        """
        device = next(generator.parameters()).device
        
        # Generate samples from current generator
        with torch.no_grad():
            gen_samples = generator(noise_samples)
        
        # Compute spatial density
        density_info = compute_spatial_density(gen_samples, bandwidth=0.12)
        sigma = density_info['density_at_samples']
        
        # Compute traffic flow
        flow_info = compute_traffic_flow(
            critic, generator, noise_samples, sigma, lambda_param=1.
        )
        
        # Store step data for analysis
        step_data = {
            'step': step,
            'samples': gen_samples.detach().cpu().numpy(),
            'target': target_samples.detach().cpu().numpy(),
            'flow': flow_info['traffic_flow'].detach().cpu().numpy(),
            'intensity': flow_info['traffic_intensity'].detach().cpu().numpy(),
            'density': sigma.detach().cpu().numpy(),
            'gradient_norm': flow_info['gradient_norm'].detach().cpu().numpy()
        }
        self.step_data.append(step_data)
        
        # Create visualization
        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle(f'Traffic Flow Analysis - Step {step}', fontsize=16, fontweight='bold')
        
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
        
        # Subsample for cleaner visualization
        # subsample_idx = np.random.choice(len(samples_np), min(100, len(samples_np)), replace=False)
        
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
        
        # Plot 3: Traffic intensity heatmap
        ax3 = axes[1, 0]
        
        # Create a grid for interpolation
        x_min, x_max = samples_np[:, 0].min() - 1, samples_np[:, 0].max() + 1
        y_min, y_max = samples_np[:, 1].min() - 1, samples_np[:, 1].max() + 1
        
        scatter = ax3.scatter(samples_np[:, 0], samples_np[:, 1], 
                            c=intensity_np, cmap='plasma', s=50, alpha=0.7)
        
        cbar3 = plt.colorbar(scatter, ax=ax3, shrink=0.8)
        cbar3.set_label('Traffic Intensity i_Q(x)')
        
        ax3.set_title('Traffic Intensity Distribution')
        ax3.set_xlabel('X₁')
        ax3.set_ylabel('X₂')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Spatial density and congestion metrics
        ax4 = axes[1, 1]
        
        # Density scatter plot
        density_scatter = ax4.scatter(samples_np[:, 0], samples_np[:, 1], 
                                    c=sigma.detach().cpu().numpy(), cmap='coolwarm', 
                                    s=50, alpha=0.7)
        
        cbar4 = plt.colorbar(density_scatter, ax=ax4, shrink=0.8)
        cbar4.set_label('Spatial Density σ(x)')
        
        ax4.set_title('Spatial Density Distribution')
        ax4.set_xlabel('X₁')
        ax4.set_ylabel('X₂')
        ax4.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Traffic Flow Statistics:
Mean Intensity: {intensity_np.mean():.4f}
Max Intensity: {intensity_np.max():.4f}
Mean Density: {sigma.mean().item():.4f}
Flow Magnitude: {np.linalg.norm(flow_np, axis=1).mean():.4f}"""
        
        if congestion_info:
            stats_text += f"\nCongestion Cost: {congestion_info.get('congestion_cost', 0):.4f}"
        
        ax4.text(0.02, 0.98, stats_text, transform=ax4.transAxes, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                fontsize=9)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / f"traffic_flow_step_{step:03d}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved traffic flow visualization: {save_path}")
        
        # plt.show() if step % 5 == 0 else plt.close()  # Only show every 5th step
        plt.close()
        
        return step_data
    
    def _unit_quiver(self, ax, X, Y, U, V, color_by=None, arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.85):
        """
        Draws a quiver plot that shows only directions using arrows of equal length.
        - arrow_frac: Arrow length as a fraction of the axis range (e.g., 0.1 means 10% of the x-range)
        - color_by: Values to encode color (e.g., traffic_intensity); if None, use a single color
        """
        X = np.asarray(X); Y = np.asarray(Y); U = np.asarray(U); V = np.asarray(V)
        # Direction unit vectors
        mag = np.sqrt(U**2 + V**2) + 1e-12
        Uu = U / mag
        Vu = V / mag

        # Axis range extents
        x_min = np.nanmin(X); x_max = np.nanmax(X)
        y_min = np.nanmin(Y); y_max = np.nanmax(Y)
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # Determine on-screen uniform arrow length considering both x and y ranges
        # Quiver interprets U,V in data coordinates; compensate for x/y scale differences
        # Reference length: includes compensation for diagonal components
        target_len_x = arrow_frac * x_range
        target_len_y = arrow_frac * y_range
        U_plot = Uu * target_len_x
        V_plot = Vu * target_len_y

        quiv = ax.quiver(
            X, Y, U_plot, V_plot,
            color_by if color_by is not None else None,
            cmap=cmap, angles='xy', scale_units='xy', scale=1.0,
            width=width, alpha=alpha
        )
        # Set axis limits with a small margin for better visuals
        ax.set_xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        return quiv
    
    def create_traffic_flow_summary(self, save=True):
        """
        Create a summary visualization of traffic flow evolution across all steps.
        """
        if not self.step_data:
            print("No step data available for summary.")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Traffic Flow Evolution Summary', fontsize=16, fontweight='bold')
        
        steps = [data['step'] for data in self.step_data]
        
        # Extract metrics over time
        mean_intensities = [data['intensity'].mean() for data in self.step_data]
        max_intensities = [data['intensity'].max() for data in self.step_data]
        mean_densities = [data['density'].mean() for data in self.step_data]
        flow_magnitudes = [np.linalg.norm(data['flow'], axis=1).mean() for data in self.step_data]
        
        # Plot 1: Intensity evolution
        ax1 = axes[0, 0]
        ax1.plot(steps, mean_intensities, 'b-', linewidth=2, label='Mean Intensity')
        ax1.plot(steps, max_intensities, 'r--', linewidth=2, label='Max Intensity')
        ax1.set_title('Traffic Intensity Evolution')
        ax1.set_xlabel('Perturbation Step')
        ax1.set_ylabel('Traffic Intensity |w_Q|')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Density evolution
        ax2 = axes[0, 1]
        ax2.plot(steps, mean_densities, 'g-', linewidth=2, label='Mean Density')
        ax2.set_title('Spatial Density Evolution')
        ax2.set_xlabel('Perturbation Step')
        ax2.set_ylabel('Spatial Density σ(x)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Flow magnitude evolution
        ax3 = axes[0, 2]
        ax3.plot(steps, flow_magnitudes, 'm-', linewidth=2, label='Flow Magnitude')
        ax3.set_title('Traffic Flow Magnitude Evolution')
        ax3.set_xlabel('Perturbation Step')
        ax3.set_ylabel('Mean |w_Q|')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: First step flow field
        ax4 = axes[1, 0]
        first_data = self.step_data[0]
        samples = first_data['samples']
        flow = first_data['flow']
        intensity = first_data['intensity']
        
        subsample_idx = np.random.choice(len(samples), min(50, len(samples)), replace=False)
        
        quiver1 = self._unit_quiver(
            ax4,
            samples[subsample_idx, 0], samples[subsample_idx, 1],
            flow[subsample_idx, 0], flow[subsample_idx, 1],
            color_by=intensity[subsample_idx],
            arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.8
        )
        ax4.set_title(f'Initial Flow Field (Step {first_data["step"]})')
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Final step flow field
        ax5 = axes[1, 1]
        final_data = self.step_data[-1]
        samples = final_data['samples']
        flow = final_data['flow']
        intensity = final_data['intensity']
        
        subsample_idx = np.random.choice(len(samples), min(50, len(samples)), replace=False)
        quiver2 = self._unit_quiver(
            ax5,
            samples[subsample_idx, 0], samples[subsample_idx, 1],
            flow[subsample_idx, 0], flow[subsample_idx, 1],
            color_by=intensity[subsample_idx],
            arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.8
        )
        ax5.set_title(f'Final Flow Field (Step {final_data["step"]})')
        ax5.grid(True, alpha=0.3)
        
        # Plot 6: Flow direction statistics
        ax6 = axes[1, 2]
        
        # Compute flow direction changes
        direction_changes = []
        for i in range(1, len(self.step_data)):
            flow_prev = self.step_data[i-1]['flow']
            flow_curr = self.step_data[i]['flow']
            
            # Compute angle between flow vectors
            dot_products = np.sum(flow_prev * flow_curr, axis=1)
            norms_prev = np.linalg.norm(flow_prev, axis=1)
            norms_curr = np.linalg.norm(flow_curr, axis=1)
            
            # Avoid division by zero
            valid_mask = (norms_prev > 1e-6) & (norms_curr > 1e-6)
            if valid_mask.sum() > 0:
                cos_angles = dot_products[valid_mask] / (norms_prev[valid_mask] * norms_curr[valid_mask])
                cos_angles = np.clip(cos_angles, -1, 1)
                angles = np.arccos(cos_angles)
                direction_changes.append(np.mean(angles))
            else:
                direction_changes.append(0)
        
        ax6.plot(steps[1:], direction_changes, 'orange', linewidth=2, marker='o')
        ax6.set_title('Flow Direction Stability')
        ax6.set_xlabel('Perturbation Step')
        ax6.set_ylabel('Mean Direction Change (radians)')
        ax6.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            save_path = self.save_dir / "traffic_flow_summary.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved traffic flow summary: {save_path}")
        
        # plt.show()
        
        return {
            'steps': steps,
            'mean_intensities': mean_intensities,
            'max_intensities': max_intensities,
            'mean_densities': mean_densities,
            'flow_magnitudes': flow_magnitudes,
            'direction_changes': direction_changes
        }


def parse_args():
    parser = argparse.ArgumentParser(description="Section 2: Target-Given Perturbation with Traffic Flow Visualization")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--pretrain_epochs", type=int, default=300, help="Pretraining epochs")
    parser.add_argument("--perturb_steps", type=int, default=50, help="Perturbation steps")
    parser.add_argument("--batch_size", type=int, default=96, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=600, help="Evaluation batch size")
    parser.add_argument("--eta_init", type=float, default=0.045, help="Initial learning rate")
    parser.add_argument("--enable_congestion", action="store_true", default=True, help="Enable congestion tracking")
    parser.add_argument("--use_sobolev_critic", action="store_true", default=True, help="Use Sobolev-constrained critic")
    parser.add_argument("--visualize_every", type=int, default=5, help="Visualize traffic flow every N steps")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    return parser.parse_args()


def run_section2_example():
    """Run Section 2 example with comprehensive traffic flow visualization."""
    args = parse_args()
    
    # Set seed and device
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else compute_device()
    
    print("="*80)
    print("SECTION 2: TARGET-GIVEN PERTURBATION WITH TRAFFIC FLOW VISUALIZATION")
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
    
    # Create models
    print("\nInitializing models...")
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)
    
    if args.use_sobolev_critic:
        critic = SobolevConstrainedCritic(
            data_dim=2, hidden_dim=256,
            use_spectral_norm=True, 
            lambda_sobolev=0.1,
            sobolev_bound=50.0
        ).to(device)
        print("Using Sobolev-constrained critic with spectral normalization")
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
        lr=1e-4,
        gp_lambda=0.,
        betas=(0., 0.95),
        device=device,
        verbose=args.verbose
    )
    
    # Create target data (shifted clusters)
    print("\nCreating target distribution...")
    target_samples = sample_target_data(
        batch_size=args.eval_batch_size,
        shift=[2., 2.],  # Significant shift for clear traffic flow
        device=device
    )
    
    # Initialize traffic flow visualizer
    visualizer = TrafficFlowVisualizer(
        figsize=(15, 10),
        save_dir=f"test_results/plots/section2_traffic_flow_seed_{args.seed}"
    )
    
    # Create perturber with congestion tracking
    print(f"\nInitializing perturber with congestion tracking: {args.enable_congestion}")
    perturber = CTWeightPerturberTargetGiven(
        pretrained_gen, target_samples,
        critic=pretrained_critic,
        enable_congestion_tracking=args.enable_congestion
    )
    
    perturber.config.update({
        'eta_init': args.eta_init,           # Significantly reduced learning rate
        'eta_min': 1e-6,             # Lower minimum
        'eta_max': 0.5,             # Lower maximum
        'eta_decay_factor': 0.9,    # Less aggressive decay
        'eta_boost_factor': 1.05,    # Very conservative boost
        'clip_norm': 0.2,            # Strong clipping
        'momentum': 0.85,            # Reduced momentum
        'patience': 15,              # Increased patience
        'rollback_patience': 10,      # Increased rollback patience
        'lambda_entropy': 0.012,     # Reduced entropy
        'lambda_virtual': 0.8,       # Reduced virtual weight
        'lambda_multi': 1.0,         # Reduced multi weight
        'lambda_congestion': 10.0,   # Reduced congestion parameter
        'lambda_sobolev': 0.1,     # Reduced Sobolev parameter
        'eval_batch_size': args.eval_batch_size
    })

    # Custom perturbation loop with traffic flow visualization
    print(f"\nStarting perturbation with traffic flow visualization...")
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

        # Main perturbation loop with visualization
        for step in range(args.perturb_steps):
            pretrained_gen.train()

            # Generate noise for this step
            noise_samples = torch.randn(args.eval_batch_size, 2, device=device)
            
            # Compute loss and gradients with congestion tracking
            if args.enable_congestion:
                loss, grads, congestion_info = perturber._compute_loss_and_grad_with_congestion(pert_gen)
                
                # Update congestion tracker
                if congestion_info:
                    perturber.congestion_tracker.update(congestion_info)
                
                # Visualize traffic flow at specified intervals
                if step % args.visualize_every == 0:
                    print(f"\n--- Visualizing Traffic Flow at Step {step} ---")
                    step_data = visualizer.visualize_traffic_flow_step(
                        step, pert_gen, pretrained_critic, target_samples,
                        noise_samples, congestion_info, save=args.save_plots
                    )
                    
                    # Print step statistics
                    print(f"Step {step} Statistics:")
                    print(f"  Mean traffic intensity: {step_data['intensity'].mean():.6f}")
                    print(f"  Max traffic intensity: {step_data['intensity'].max():.6f}")
                    print(f"  Mean spatial density: {step_data['density'].mean():.6f}")
                    print(f"  Congestion cost: {congestion_info.get('congestion_cost', .0):.6f}")
                    print(f"  Flow magnitude: {np.linalg.norm(step_data['flow'], axis=1).mean():.6f}")
                
                # Compute delta_theta with congestion awareness
                delta_theta = perturber._compute_delta_theta_with_congestion(
                    grads, eta, perturber.config['clip_norm'], perturber.config['momentum'], delta_theta_prev, congestion_info
                )
            else:
                loss, grads = perturber._compute_loss_and_grad(pert_gen)
                delta_theta = perturber._compute_delta_theta(
                    grads, eta, perturber.config['clip_norm'], perturber.config['momentum'], delta_theta_prev
                )
                
                loss_hist.append(loss.item())

                # Visualize without congestion info
                if step % args.visualize_every == 0:
                    step_data = visualizer.visualize_traffic_flow_step(
                        step, pert_gen, pretrained_critic, target_samples,
                        noise_samples, save=args.save_plots
                    )
            
            # Apply parameter update
            theta_prev = perturber._apply_parameter_update(pert_gen, theta_prev, delta_theta)
            delta_theta_prev = delta_theta.clone()
            
            # Validate and adapt
            w2_pert, improvement = perturber._validate_and_adapt(pert_gen, eta, w2_hist, perturber.config['patience'], args.verbose, step)
            
            # Update best state
            best_w2, best_vec = perturber._update_best_state(w2_pert, pert_gen, best_w2, best_vec)
            
            # Adapt learning rate
            eta, no_improvement_count = perturber._adapt_learning_rate(eta, improvement, step, no_improvement_count, loss_hist)

            
            # Check for rollback condition
            if perturber._check_rollback_condition_with_congestion(w2_hist, no_improvement_count):
                if args.verbose:
                    print(f"Rollback triggered at step {step}")
                
                perturber._restore_best_state(pert_gen, best_vec)
                # Reset parameters after rollback
                eta = max(eta * 0.9, perturber.config.get('eta_min', 1e-6))
                no_improvement_count = 0
                consecutive_rollbacks += 1
                delta_theta_prev = torch.zeros_like(delta_theta_prev)
                
                # If too many rollbacks, break early
                if consecutive_rollbacks >= 3:  # Reduced from 5
                    if args.verbose:
                        print(f"Too many rollbacks, stopping early")
                    break
            else:
                consecutive_rollbacks = 0

            # Print progress
            if args.verbose:
                log_msg = f"[{step:2d}] W2={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                if args.enable_congestion and 'congestion_info' in locals():
                    log_msg += f" Congestion={congestion_info.get('congestion_cost', 0):.4f}"
                print(log_msg)
        
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
        
        # Compute final convergence metrics
        final_metrics = compute_convergence_metrics(pert_gen, target_samples, noise_eval)
        print("Final Convergence Metrics:")
        for key, value in final_metrics.items():
            print(f"  {key}: {value:.6f}")
        
        # Final traffic flow analysis
        if args.enable_congestion:
            final_congestion_stats = perturber.loss_function.get_congestion_statistics()
            print("\nFinal Congestion Statistics:")
            for key, value in final_congestion_stats.items():
                print(f"  {key}: {value:.6f}")
        
        return {
            'perturbed_generator': pert_gen,
            'original_samples': original_samples,
            'final_samples': final_samples,
            'target_samples': target_samples,
            'convergence_metrics': final_metrics,
            'traffic_flow_data': visualizer.step_data,
            'summary_data': summary_data
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
        print("  ✓ Congestion tracking completed")
        print("  ✓ Spatial density analysis performed")
        print("  ✓ Flow vector field evolution captured")
        print("  ✓ Summary plots created")
        
        # Print key insights
        traffic_data = results['traffic_flow_data']
        if traffic_data:
            initial_intensity = traffic_data[0]['intensity'].mean()
            final_intensity = traffic_data[-1]['intensity'].mean()
            print(f"\nKey Insights:")
            print(f"  Traffic intensity change: {initial_intensity:.6f} → {final_intensity:.6f}")
            print(f"  Total steps visualized: {len(traffic_data)}")
            
        print(f"\nPlots saved in: section2_traffic_flow_seed_{parse_args().seed}/")
    else:
        print("Section 2 demonstration failed.")