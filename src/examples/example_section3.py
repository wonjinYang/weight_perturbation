# This script demonstrates the Weight Perturbation strategy for Section 3 (evidence-based perturbation without explicit target).
# It pretrains a WGAN-GP generator on a toy real data distribution (e.g., Gaussian clusters),
# samples multiple evidence domains (circularly placed clusters), estimates a virtual target via broadened KDE,
# and applies perturbation using the WeightPerturberSection3 class to align the generator's output with the evidence.
# Finally, it evaluates the results by computing multi-marginal OT losses and plotting distributions.

import torch
import argparse
from typing import Optional, List

from weight_perturbation import (
    Generator,
    Critic,
    sample_real_data,
    sample_evidence_domains,
    pretrain_wgan_gp,
    WeightPerturberTargetNotGiven,
    multi_marginal_ot_loss,
    plot_distributions,
    load_config,
    set_seed,
    compute_device,
    virtual_target_sampler
)

# Parse command-line arguments for customization
def parse_args():
    parser = argparse.ArgumentParser(description="Demo for Section 3: Evidence-Based Weight Perturbation")
    parser.add_argument("--seed", type=int, default=2025, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--pretrain_epochs", type=int, default=300, help="Number of pretraining epochs")
    parser.add_argument("--perturb_epochs", type=int, default=100, help="Number of perturbation epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for pretraining")
    parser.add_argument("--eval_batch_size", type=int, default=600, help="Batch size for evaluation and virtual sampling")
    parser.add_argument("--noise_dim", type=int, default=2, help="Dimension of noise input")
    parser.add_argument("--data_dim", type=int, default=2, help="Dimension of data output")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for models")
    parser.add_argument("--num_evidence_domains", type=int, default=3, help="Number of evidence domains")
    parser.add_argument("--samples_per_domain", type=int, default=35, help="Samples per evidence domain")
    parser.add_argument("--random_shift", type=float, default=3.4, help="Radius for circular evidence placement")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of distributions")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration with fallback to defaults
    try:
        config = load_config(args.config)
        # Merge with args
        for key, value in vars(args).items():
            if value is not None and key != 'config':
                config[key] = value
    except Exception as e:
        print(f"Warning: Could not load config from {args.config}: {e}")
        print("Using default configuration...")
        config = {
            'seed': args.seed,
            'noise_dim': args.noise_dim,
            'data_dim': args.data_dim,
            'hidden_dim': args.hidden_dim,
            'pretrain_epochs': args.pretrain_epochs,
            'perturb_epochs': args.perturb_epochs,
            'batch_size': args.batch_size,
            'eval_batch_size': args.eval_batch_size,
            'num_evidence_domains': args.num_evidence_domains,
            'samples_per_domain': args.samples_per_domain,
            'random_shift': args.random_shift,
            'plot': args.plot,
            'verbose': args.verbose
        }
    
    # Set seed and device
    set_seed(config["seed"])
    device = torch.device(args.device) if args.device else compute_device()
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Initialize models
    generator = Generator(
        noise_dim=config["noise_dim"],
        data_dim=config["data_dim"],
        hidden_dim=config["hidden_dim"]
    ).to(device)
    
    critic = Critic(
        data_dim=config["data_dim"],
        hidden_dim=config["hidden_dim"]
    ).to(device)
    
    # Define real data sampler (closure for compatibility)
    def real_sampler(batch_size: int):
        return sample_real_data(
            batch_size=batch_size,
            means=None,  # Default 4 clusters
            std=0.7,
            device=device
        )
    
    # Pretrain using WGAN-GP
    print("Starting pretraining...")
    pretrained_gen, pretrained_crit = pretrain_wgan_gp(
        generator=generator,
        critic=critic,
        real_sampler=real_sampler,
        epochs=config["pretrain_epochs"],
        batch_size=config["batch_size"],
        lr=2e-4,
        betas=(0.5, 0.98),
        gp_lambda=0.06,
        critic_iters=2,
        noise_dim=config["noise_dim"],
        device=device,
        verbose=config["verbose"]
    )
    print("Pretraining completed.")
    
    # Sample evidence domains
    evidence_list, centers = sample_evidence_domains(
        num_domains=config["num_evidence_domains"],
        samples_per_domain=config["samples_per_domain"],
        random_shift=config["random_shift"],
        std=0.4,
        device=device
    )
    
    # Initialize perturber
    perturber = WeightPerturberTargetNotGiven(
        generator=pretrained_gen,
        evidence_list=evidence_list,
        centers=centers,
        config=config
    )
    
    # Perform perturbation
    print("Starting perturbation...")
    perturbed_gen = perturber.perturb(
        epochs=config["perturb_epochs"],
        eta_init=0.045,
        clip_norm=0.23,
        momentum=0.975,
        patience=6,
        lambda_entropy=0.012,
        verbose=config["verbose"]
    )
    print("Perturbation completed.")
    
    # Evaluation
    noise = torch.randn(config["eval_batch_size"], config["noise_dim"], device=device)
    
    with torch.no_grad():
        original_samples = pretrained_gen(noise)
        perturbed_samples = perturbed_gen(noise)
    
    # Compute multi-marginal OT losses
    ot_original = multi_marginal_ot_loss(
        original_samples,
        evidence_list,
        weights=None,
        blur=0.06,
        entropy_lambda=0.012
    ).item()
    
    ot_perturbed = multi_marginal_ot_loss(
        perturbed_samples,
        evidence_list,
        weights=None,
        blur=0.06,
        entropy_lambda=0.012
    ).item()
    
    print(f"Multi-Marginal OT (Original to Evidence): {ot_original:.4f}")
    print(f"Multi-Marginal OT (Perturbed to Evidence): {ot_perturbed:.4f}")
    print(f"Improvement: {((ot_original - ot_perturbed) / ot_original):.4f}")
    
    # Optional plotting (using a final virtual target for visualization)
    if args.plot:
        # Sample a final virtual target for plotting
        final_virtual = virtual_target_sampler(
            evidence_list,
            weights=None,
            bandwidth=0.19,
            num_samples=config["eval_batch_size"],
            temperature=1.0,
            device=device
        )
        
        plot_distributions(
            original=original_samples,
            perturbed=perturbed_samples,
            target_or_virtual=final_virtual,
            evidence=evidence_list,
            title="Section 3: Evidence-Based Perturbation Results",
            save_path="section3_results.png",
            show=True
        )

if __name__ == "__main__":
    main()