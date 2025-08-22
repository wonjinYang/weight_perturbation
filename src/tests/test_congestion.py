"""
Congestion Tracking and Traffic Flow Visualization Example

This example demonstrates:
1. Spatial density estimation σ(x)
2. Traffic flow computation w_Q with vector directions
3. Traffic intensity tracking i_Q  
4. Congestion tracking
5. Traffic flow vector field visualization
6. Sobolev regularization integration
7. Theoretical framework implementation
8. Theoretical validation and mass conservation
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import logging

from weight_perturbation import (
    _THEORETICAL_COMPONENTS_AVAILABLE,
    check_theoretical_support,
    set_seed,
    compute_device,
    sample_real_data,
    pretrain_wgan_gp,
    sample_evidence_domains,
    parameters_to_vector,
    Generator,
    # Theoretical components
    SobolevConstrainedCritic,
    CTWeightPerturberTargetNotGiven,
    MultiMarginalCongestedTransportVisualizer,
    # Validation
    validate_theoretical_consistency,
    enforce_mass_conservation,
)

# Set up plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set up logging
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser(description="Section 3: Evidence-Based Perturbation with Multi-Marginal Flow Visualization")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--pretrain_epochs", type=int, default=300, help="Pretraining epochs")
    parser.add_argument("--perturb_epochs", type=int, default=50, help="Perturbation epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=600, help="Evaluation batch size")
    parser.add_argument("--num_evidence_domains", type=int, default=3, help="Number of evidence domains")
    parser.add_argument("--samples_per_domain", type=int, default=20, help="Samples per evidence domain")
    parser.add_argument("--eta_init", type=float, default=0.08, help="Initial learning rate")
    parser.add_argument("--enable_congestion", action="store_true", default=True, help="Enable congestion tracking")
    parser.add_argument("--use_sobolev_critic", action="store_true", default=True, help="Use Sobolev-constrained critics")
    parser.add_argument("--visualize_every", type=int, default=5, help="Visualize traffic flow every N epochs")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    parser.add_argument("--enable_theoretical_validation", action="store_true", default=True, help="Enable theoretical validation")
    parser.add_argument("--enable_mass_conservation", action="store_true", default=True, help="Enable mass conservation")
    return parser.parse_args()

def run_section3_example():
    """Run Section 3 example with multi-marginal traffic flow visualization."""
    args = parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else compute_device()

    print("="*80)
    print("SECTION 3: EVIDENCE-BASED PERTURBATION WITH MULTI-MARGINAL TRAFFIC FLOW")
    print("="*80)
    print(f"Device: {device}")
    print(f"Theoretical components available: {_THEORETICAL_COMPONENTS_AVAILABLE}")
    print(f"Evidence domains: {args.num_evidence_domains}")

    if not _THEORETICAL_COMPONENTS_AVAILABLE:
        print("This example requires theoretical components. Exiting.")
        return

    # Check theoretical support
    support_ok = check_theoretical_support()
    if not support_ok:
        print("Warning: Some theoretical components may not work correctly.")

    # Create models with theoretical integration
    print("\nInitializing models...")
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)

    # Create multiple critics for evidence domains
    critics = []
    for i in range(args.num_evidence_domains):
        if args.use_sobolev_critic:
            critic = SobolevConstrainedCritic(
                data_dim=2, hidden_dim=256,
                use_spectral_norm=True,
                lambda_sobolev=0.1,
                sobolev_bound=5.0
            ).to(device)
        else:
            from weight_perturbation import Critic
            critic = Critic(data_dim=2, hidden_dim=256).to(device)
        critics.append(critic)
    print(f"Created {len(critics)} {'Sobolev-constrained' if args.use_sobolev_critic else 'standard'} critics")

    # Pretrain generator with first critic
    print(f"\nPretraining generator for {args.pretrain_epochs} epochs...")
    real_sampler = lambda bs: sample_real_data(
        bs, means=[torch.tensor([0.0, 0.0], device=device)], std=0.7, device=device
    )

    pretrained_gen, _ = pretrain_wgan_gp(
        generator, critics[0], real_sampler,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=2e-4,
        gp_lambda=0.5,
        device=device,
        verbose=args.verbose
    )

    # Create evidence domains with theoretical spread
    print(f"\nCreating {args.num_evidence_domains} evidence domains...")
    evidence_list, centers = sample_evidence_domains(
        num_domains=args.num_evidence_domains,
        samples_per_domain=args.samples_per_domain,
        random_shift=3.5,  # Spread for better theoretical analysis
        std=0.6,
        device=device
    )

    print("Evidence domain centers:")
    for i, center in enumerate(centers):
        print(f" Domain {i+1}: {center}")

    # Initialize multi-marginal traffic flow visualizer
    visualizer = MultiMarginalCongestedTransportVisualizer(
        num_domains=args.num_evidence_domains,
        figsize=(22, 16),
        save_dir=f"test_results/plots/congestion_analysis/"
    )

    # Create multi-marginal perturber with theoretical integration
    print(f"\nInitializing multi-marginal perturber...")
    
    # Configuration with theoretical parameters
    config = {
        'eta_init': args.eta_init,
        'eta_min': 5e-6,
        'eta_max': 0.8,
        'eta_decay_factor': 0.9,
        'eta_boost_factor': 1.05,
        'clip_norm': 0.6,
        'momentum': 0.88,
        'patience': 12,
        'rollback_patience': 8,
        'lambda_entropy': 0.012,
        'lambda_virtual': 0.8,
        'lambda_multi': 1.0,
        'lambda_congestion': 1.0,
        'lambda_sobolev': 0.1,
        'eval_batch_size': args.eval_batch_size,
        'theoretical_validation': args.enable_theoretical_validation,
        'mass_conservation_weight': 0.1 if args.enable_mass_conservation else 0.0,
    }
    
    perturber = CTWeightPerturberTargetNotGiven(
        pretrained_gen, evidence_list, centers,
        critics=critics,
        config=config,
        enable_congestion_tracking=args.enable_congestion
    )

    print(f"Multi-marginal congestion tracking: {args.enable_congestion}")
    print(f"Theoretical validation: {args.enable_theoretical_validation}")
    print(f"Mass conservation: {args.enable_mass_conservation}")
    if args.enable_congestion:
        print(f"Initialized {len(perturber.multi_congestion_trackers)} domain-specific congestion trackers")

    # Perturbation loop with theoretical validation
    print(f"\nStarting multi-marginal perturbation with theoretical validation...")
    print(f"Will visualize every {args.visualize_every} epochs")

    try:
        data_dim = evidence_list[0].shape[1]
        pert_gen = perturber._create_generator_copy(data_dim)
        
        theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
        delta_theta_prev = torch.zeros_like(theta_prev)
        eta = config['eta_init']
        best_vec = None
        best_ot = float('inf')
        ot_hist = []
        no_improvement_count = 0
        consecutive_rollbacks = 0

        # Main perturbation loop with theoretical validation
        for epoch in range(args.perturb_epochs):
            try:
                # Estimate virtual target with congestion awareness
                virtual_samples = perturber._estimate_virtual_target_with_congestion(
                    evidence_list, epoch
                )

                # Generate noise for this epoch
                noise_samples = torch.randn(perturber.eval_batch_size, 2, device=device)

                # Compute multi-marginal congestion if enabled
                multi_congestion_info = None
                if args.enable_congestion and critics:
                    multi_congestion_info = perturber._compute_multi_marginal_congestion(pert_gen, noise_samples)

                # Theoretical validation at each step
                if args.enable_theoretical_validation and epoch % 3 == 0:
                    with torch.no_grad():
                        gen_samples = pert_gen(noise_samples)
                    validation_results = perturber._validate_theoretical_step(
                        pert_gen, virtual_samples, multi_congestion_info
                    )
                    if validation_results:
                        consistency = validation_results.get('overall_consistency', 0.0)
                        if consistency < 0.3:
                            print(f"Warning: Low theoretical consistency at epoch {epoch}: {consistency:.3f}")

                # Visualize multi-marginal traffic flow at specified intervals
                if epoch % args.visualize_every == 0:
                    print(f"\n--- Multi-Marginal Traffic Flow Visualization at Epoch {epoch} ---")
                    epoch_data = visualizer.visualize_congested_transport_step(
                        epoch, pert_gen, critics, evidence_list, virtual_samples,
                        noise_samples, multi_congestion_info, save=args.save_plots
                    )

                    # Epoch statistics with theoretical metrics
                    if epoch_data['domain_flows']:
                        print(f"Epoch {epoch} Multi-Marginal Statistics:")
                        for df in epoch_data['domain_flows']:
                            domain_id = df['domain_id']
                            print(f" Domain {domain_id+1}:")
                            print(f" Mean intensity: {df['intensity'].mean():.6f}")
                            print(f" Max intensity: {df['intensity'].max():.6f}")
                            print(f" Flow magnitude: {np.linalg.norm(df['flow'], axis=1).mean():.6f}")
                            print(f" Spatial density: {df['density'].mean():.6f}")

                        if multi_congestion_info and 'domains' in multi_congestion_info:
                            total_congestion = sum(
                                d['congestion_cost'].item() if isinstance(d['congestion_cost'], torch.Tensor)
                                else float(d['congestion_cost'])
                                for d in multi_congestion_info['domains']
                            )
                            print(f" Total congestion cost: {total_congestion:.6f}")

                            # Theoretical consistency checking
                            if args.enable_theoretical_validation:
                                domain_consistency = []
                                for d in multi_congestion_info['domains']:
                                    if 'theoretical_consistency' in d:
                                        domain_consistency.append(d['theoretical_consistency'])
                                if domain_consistency:
                                    avg_consistency = np.mean(domain_consistency)
                                    print(f" Average theoretical consistency: {avg_consistency:.3f}")

                # Compute loss and gradients with theoretical integration
                loss, grads = perturber._compute_loss_and_grad(
                    pert_gen, virtual_samples, 
                    config['lambda_entropy'], 
                    config['lambda_virtual'], 
                    config['lambda_multi']
                )

                # Compute delta_theta with multi-marginal congestion awareness
                if multi_congestion_info and multi_congestion_info['domains']:
                    avg_congestion_info = perturber._average_multi_marginal_congestion(multi_congestion_info)
                    with torch.no_grad():
                        gen_samples = pert_gen(noise_samples)
                    delta_theta = perturber._compute_delta_theta_with_congestion(
                        grads, eta, config['clip_norm'],
                        config['momentum'], delta_theta_prev, 
                        avg_congestion_info, virtual_samples, gen_samples
                    )
                else:
                    delta_theta = perturber._compute_delta_theta(
                        grads, eta, config['clip_norm'],
                        config['momentum'], delta_theta_prev
                    )

                # Apply parameter update
                theta_prev = perturber._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                delta_theta_prev = delta_theta.clone()

                # Validation and adaptation
                ot_pert, improvement = perturber._validate_and_adapt(
                    pert_gen, virtual_samples, eta, ot_hist, config['patience'], args.verbose, epoch
                )

                # Update best state
                best_ot, best_vec = perturber._update_best_state(ot_pert, pert_gen, best_ot, best_vec)

                # Learning rate adaptation
                eta, no_improvement_count = perturber._adapt_learning_rate(
                    eta, improvement, epoch, no_improvement_count, ot_hist
                )

                # Rollback condition checking with theoretical validation
                if perturber._check_rollback_condition_with_congestion(ot_hist, no_improvement_count):
                    if args.verbose:
                        print(f"Rollback triggered at epoch {epoch}")
                    perturber._restore_best_state(pert_gen, best_vec)
                    eta = max(eta * 0.95, config.get('eta_min', 5e-6))
                    no_improvement_count = 0
                    consecutive_rollbacks += 1
                    delta_theta_prev = torch.zeros_like(delta_theta_prev)
                    
                    if consecutive_rollbacks >= 3:
                        if args.verbose:
                            print(f"Too many rollbacks, stopping early")
                        break
                else:
                    consecutive_rollbacks = 0

                if args.verbose:
                    log_msg = f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                    if multi_congestion_info and multi_congestion_info['domains']:
                        total_congestion = sum(
                            d['congestion_cost'].item() if isinstance(d['congestion_cost'], torch.Tensor)
                            else float(d['congestion_cost'])
                            for d in multi_congestion_info['domains']
                        )
                        log_msg += f" Congestion={total_congestion:.2f}"
                    print(log_msg)

                # Early stopping with theoretical criteria
                if no_improvement_count >= config['patience']:
                    if args.verbose:
                        print(f"Early stopping at epoch {epoch}")
                    break

                # Check for good theoretical convergence
                if ot_pert < 0.01 and args.enable_theoretical_validation:
                    if args.verbose:
                        print(f"Good convergence at epoch {epoch}")
                    break

            except Exception as e:
                print(f"Error in epoch {epoch}: {e}")
                if epoch == 0:
                    raise
                break

        # Final restore
        perturber._restore_best_state(pert_gen, best_vec)
        
        # Create summary visualization
        visualizer.create_multimarginal_summary(save=args.save_plots)

        print("\nPerturbation completed successfully!")
        
        # Final theoretical validation
        if args.enable_theoretical_validation:
            print("\n--- Final Theoretical Validation ---")
            final_noise = torch.randn(perturber.eval_batch_size, 2, device=device)
            final_validation = perturber._validate_theoretical_step(
                pert_gen, virtual_samples, multi_congestion_info
            )
            if final_validation:
                print(f"Final theoretical consistency: {final_validation.get('overall_consistency', 0.0):.3f}")
                print(f"Final mass conservation error: {final_validation.get('coverage_error', 0.0):.6f}")

    except Exception as e:
        print(f"Error in perturbation process: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_section3_example()
