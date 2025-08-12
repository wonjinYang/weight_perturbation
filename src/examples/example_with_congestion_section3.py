"""
Section 3 example: Evidence-based perturbation with multi-marginal congestion tracking
and traffic flow visualization.

This example demonstrates:
- Multi-marginal traffic flow computation across evidence domains
- Domain-specific congestion tracking
- Virtual target estimation with congestion awareness
- Multi-domain traffic flow vector field visualization
- Evidence-weighted spatial density analysis
"""

import torch
import argparse
import numpy as np

# Try to import theoretical components
try:
    from weight_perturbation import (
        Generator,
        sample_real_data,
        sample_evidence_domains,
        pretrain_wgan_gp,
        parameters_to_vector,
        # Advanced components
        CTWeightPerturberTargetNotGiven,
        SobolevConstrainedCritic,
        MultiMarginalCongestedTransportVisualizer,
        set_seed,
        compute_device,
        check_theoretical_support
    )
    THEORETICAL_AVAILABLE = True
except ImportError as e:
    print(f"Theoretical components not available: {e}")
    print("This example requires the theoretical components to run.")
    exit(1)

def parse_args():
    parser = argparse.ArgumentParser(description="Section 3: Evidence-Based Perturbation with Multi-Marginal Flow Visualization")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed")
    parser.add_argument("--device", type=str, default=None, help="Device to use")
    parser.add_argument("--pretrain_epochs", type=int, default=300, help="Pretraining epochs")
    parser.add_argument("--perturb_epochs", type=int, default=50, help="Perturbation epochs")
    parser.add_argument("--batch_size", type=int, default=96, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=300, help="Evaluation batch size")
    parser.add_argument("--num_evidence_domains", type=int, default=3, help="Number of evidence domains")
    parser.add_argument("--samples_per_domain", type=int, default=20, help="Samples per evidence domain")
    parser.add_argument("--eta_init", type=float, default=0.045, help="Initial learning rate")
    parser.add_argument("--enable_congestion", action="store_true", default=True, help="Enable congestion tracking")
    parser.add_argument("--use_sobolev_critic", action="store_true", default=True, help="Use Sobolev-constrained critics")
    parser.add_argument("--visualize_every", type=int, default=5, help="Visualize traffic flow every N epochs")
    parser.add_argument("--save_plots", action="store_true", default=True, help="Save plots")
    parser.add_argument("--verbose", action="store_true", default=True, help="Verbose output")
    return parser.parse_args()

def run_section3_example():
    """Run Section 3 example with comprehensive multi-marginal traffic flow visualization."""
    args = parse_args()

    # Set seed and device
    set_seed(args.seed)
    device = torch.device(args.device) if args.device else compute_device()

    print("="*80)
    print("SECTION 3: EVIDENCE-BASED PERTURBATION WITH MULTI-MARGINAL TRAFFIC FLOW")
    print("="*80)
    print(f"Device: {device}")
    print(f"Theoretical components available: {THEORETICAL_AVAILABLE}")
    print(f"Evidence domains: {args.num_evidence_domains}")

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

    # Create multiple critics for evidence domains
    critics = []
    for i in range(args.num_evidence_domains):
        if args.use_sobolev_critic:
            critic = SobolevConstrainedCritic(
                data_dim=2, hidden_dim=256,
                use_spectral_norm=True,
                lambda_sobolev=0.1,
                sobolev_bound=50.0
            ).to(device)
        else:
            from weight_perturbation import Critic
            critic = Critic(data_dim=2, hidden_dim=256).to(device)
        critics.append(critic)
    print(f"Created {len(critics)} {'Sobolev-constrained' if args.use_sobolev_critic else 'standard'} critics")

    # Pretrain generator with first critic
    print(f"\nPretraining generator for {args.pretrain_epochs} epochs...")
    real_sampler = lambda bs: sample_real_data(
        bs, means=[torch.tensor([0.0, 0.0], device=device)], std=0.5, device=device
    )

    pretrained_gen, _ = pretrain_wgan_gp(
        generator, critics[0], real_sampler,
        epochs=args.pretrain_epochs,
        batch_size=args.batch_size,
        lr=2e-4,
        gp_lambda=0.,
        betas=(0., 0.95),
        device=device,
        verbose=args.verbose
    )

    # Create evidence domains
    print(f"\nCreating {args.num_evidence_domains} evidence domains...")
    evidence_list, centers = sample_evidence_domains(
        num_domains=args.num_evidence_domains,
        samples_per_domain=args.samples_per_domain,
        random_shift=3.0,  # Spread out domains
        std=0.5,
        device=device
    )

    print("Evidence domain centers:")
    for i, center in enumerate(centers):
        print(f" Domain {i+1}: {center}")

    # Initialize multi-marginal traffic flow visualizer
    visualizer = MultiMarginalCongestedTransportVisualizer(
        num_domains=args.num_evidence_domains,
        figsize=(20, 12),
        save_dir=f"test_results/plots/section3_multimarginal_seed_{args.seed}"
    )

    # Create perturber with multi-marginal congestion tracking
    print(f"\nInitializing multi-marginal perturber...")
    perturber = CTWeightPerturberTargetNotGiven(
        pretrained_gen, evidence_list, centers,
        critics=critics,
        enable_congestion_tracking=args.enable_congestion
    )

    print(f"Multi-marginal congestion tracking: {args.enable_congestion}")
    if args.enable_congestion:
        print(f"Initialized {len(perturber.multi_congestion_trackers)} domain-specific congestion trackers")

    # Custom perturbation loop with multi-marginal visualization
    print(f"\nStarting multi-marginal perturbation with flow visualization...")
    print(f"Will visualize every {args.visualize_every} epochs")

    try:
        data_dim = evidence_list[0].shape[1]
        pert_gen = perturber._create_generator_copy(data_dim)

        # Initialize perturbation state
        theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
        delta_theta_prev = torch.zeros_like(theta_prev)
        eta = args.eta_init
        best_vec = None
        best_ot = float('inf')

        # Main perturbation loop with multi-marginal visualization
        for epoch in range(args.perturb_epochs):
            # Estimate virtual target with congestion awareness
            virtual_samples = perturber._estimate_virtual_target_with_congestion(
                evidence_list, epoch
            )

            # Generate noise for this epoch
            noise_samples = torch.randn(args.eval_batch_size, 2, device=device)

            # Compute multi-marginal congestion if enabled
            multi_congestion_info = None
            if args.enable_congestion and critics:
                multi_congestion_info = perturber._compute_multi_marginal_congestion(pert_gen, noise_samples)

            # Visualize multi-marginal traffic flow at specified intervals
            if epoch % args.visualize_every == 0:
                print(f"\n--- Visualizing Multi-Marginal Traffic Flow at Epoch {epoch} ---")
                epoch_data = visualizer.visualize_multimarginal_flow_epoch(
                    epoch, pert_gen, critics, evidence_list, virtual_samples,
                    noise_samples, multi_congestion_info, save=args.save_plots
                )

                # Print epoch statistics
                if epoch_data['domain_flows']:
                    print(f"Epoch {epoch} Multi-Marginal Statistics:")
                    for df in epoch_data['domain_flows']:
                        domain_id = df['domain_id']
                        print(f" Domain {domain_id+1}:")
                        print(f" Mean intensity: {df['intensity'].mean():.6f}")
                        print(f" Max intensity: {df['intensity'].max():.6f}")
                        print(f" Flow magnitude: {np.linalg.norm(df['flow'], axis=1).mean():.6f}")

                    if multi_congestion_info and 'domains' in multi_congestion_info:
                        total_congestion = sum(d['congestion_cost'].item() for d in multi_congestion_info['domains'])
                        print(f" Total congestion cost: {total_congestion:.6f}")

            # Compute loss and gradients
            loss, grads = perturber._compute_loss_and_grad(
                pert_gen, virtual_samples, 0.045, 0.8, 1.0
            )

            # Compute delta_theta with multi-marginal congestion awareness
            if multi_congestion_info and multi_congestion_info['domains']:
                avg_congestion_info = perturber._average_multi_marginal_congestion(multi_congestion_info)
                delta_theta = perturber._compute_delta_theta_with_congestion(
                    grads, eta, perturber.config.get('clip_norm', 0.4),
                    perturber.config.get('momentum', 0.975),
                    delta_theta_prev, avg_congestion_info
                )
            else:
                delta_theta = perturber._compute_delta_theta(
                    grads, eta, perturber.config.get('clip_norm', 0.4),
                    perturber.config.get('momentum', 0.975),
                    delta_theta_prev
                )

            # Apply update
            theta_prev = perturber._apply_parameter_update(pert_gen, theta_prev, delta_theta)
            delta_theta_prev = delta_theta.clone()

            # Validate and adapt
            ot_pert, improvement = perturber._validate_and_adapt(
                pert_gen, virtual_samples, eta, [], perturber.config.get('patience', 6), args.verbose, epoch
            )

            # Update best state
            best_ot, best_vec = perturber._update_best_state(ot_pert, pert_gen, best_ot, best_vec)

            # Adapt learning rate
            eta, no_improvement_count = perturber._adapt_learning_rate(
                eta, improvement, epoch, 0, []
            )

            # Check rollback
            if perturber._check_rollback_condition_with_congestion([], 0):
                perturber._restore_best_state(pert_gen, best_vec)
                eta *= 0.6
                delta_theta_prev = torch.zeros_like(delta_theta_prev)

            if args.verbose:
                print(f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}")

        # Final restore
        perturber._restore_best_state(pert_gen, best_vec)

        # Create summary visualization
        visualizer.create_multimarginal_summary(save=args.save_plots)

        print("\nPerturbation completed successfully!")

    except Exception as e:
        print(f"Error in perturbation process: {e}")

if __name__ == "__main__":
    run_section3_example()
