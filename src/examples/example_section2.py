# This script demonstrates the Weight Perturbation strategy for Section 2 (target-given perturbation).
# It pretrains a WGAN-GP generator on a toy real data distribution (e.g., Gaussian clusters),
# samples a target distribution (shifted clusters), and applies perturbation using the
# WeightPerturberSection2 class to align the generator's output with the target.
# Finally, it evaluates the results by computing Wasserstein-2 distances and plotting distributions.

import torch
import argparse
from typing import Optional

from weight_perturbation import (
    Generator,
    Critic,
    sample_real_data,
    sample_target_data,
    pretrain_wgan_gp,
    WeightPerturberTargetGiven,
    compute_wasserstein_distance,
    plot_distributions,
    load_config,
    set_seed,
    compute_device
)

# Parse command-line arguments for customization
def parse_args():
    parser = argparse.ArgumentParser(description="Demo for Section 2: Target-Given Weight Perturbation")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default=None, help="Device to use ('cpu' or 'cuda')")
    parser.add_argument("--config", type=str, default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--pretrain_epochs", type=int, default=300, help="Number of pretraining epochs")
    parser.add_argument("--perturb_steps", type=int, default=24, help="Number of perturbation steps")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for pretraining")
    parser.add_argument("--eval_batch_size", type=int, default=1600, help="Batch size for evaluation")
    parser.add_argument("--noise_dim", type=int, default=2, help="Dimension of noise input")
    parser.add_argument("--data_dim", type=int, default=2, help="Dimension of data output")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension for models")
    parser.add_argument("--plot", action="store_true", help="Enable plotting of distributions")
    parser.add_argument("--verbose", action="store_true", help="Print verbose output during training")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load configuration with fallback to defaults
    try:
        config = load_config(args.config)
    except:
        print(f"Warning: Could not load config from {args.config}, using defaults")
        config = {
            'seed': 42,
            'noise_dim': 2,
            'data_dim': 2,
            'hidden_dim': 256,
            'pretrain_epochs': 300,
            'perturb_steps': 24,
            'batch_size': 64,
            'eval_batch_size': 1600
        }
    
    # Override config with command-line args
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value
    
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
            std=0.4,
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
        betas=(0.0, 0.9),
        gp_lambda=10.0,
        critic_iters=5,
        noise_dim=config["noise_dim"],
        device=device,
        verbose=config["verbose"]
    )
    print("Pretraining completed.")
    
    # Sample target data
    target_samples = sample_target_data(
        batch_size=config["eval_batch_size"],
        shift=[1.8, 1.8],  # Default shift for toy example
        means=None,
        std=0.4,
        device=device
    )
    
    # Initialize perturber
    perturber = WeightPerturberTargetGiven(
        generator=pretrained_gen,
        target_samples=target_samples,
        config=config
    )
    
    # Perform perturbation
    print("Starting perturbation...")
    perturbed_gen = perturber.perturb(
        steps=config["perturb_steps"],
        eta_init=0.017,
        clip_norm=0.12,
        momentum=0.91,
        patience=7,
        verbose=config["verbose"]
    )
    print("Perturbation completed.")
    
    # Evaluation
    noise = torch.randn(config["eval_batch_size"], config["noise_dim"], device=device)
    
    with torch.no_grad():
        original_samples = pretrained_gen(noise)
        perturbed_samples = perturbed_gen(noise)
    
    w2_original = compute_wasserstein_distance(original_samples, target_samples, p=2, blur=0.07)
    w2_perturbed = compute_wasserstein_distance(perturbed_samples, target_samples, p=2, blur=0.07)
    
    print(f"W2 Distance (Original to Target): {w2_original.item():.4f}")
    print(f"W2 Distance (Perturbed to Target): {w2_perturbed.item():.4f}")
    print(f"Improvement: {((w2_original - w2_perturbed) / w2_original).item():.4f}")
    
    # Optional plotting
    if args.plot:
        plot_distributions(
            original=original_samples,
            perturbed=perturbed_samples,
            target_or_virtual=target_samples,
            evidence=None,
            title="Section 2: Target-Given Perturbation Results",
            save_path="section2_results.png",
            show=True
        )

if __name__ == "__main__":
    main()