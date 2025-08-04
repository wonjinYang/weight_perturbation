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
    parser.add_argument("--num_evidence_domains", type=int, default=4, help="Number of evidence domains")
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
    
    # Sample evidence domains
    evidence_list, centers = sample_evidence_domains(
        num_domains=config["num_evidence_domains"],
        samples_per_domain=config["samples_per_domain"],
        random_shift=config["random_shift"],
        std=0.7,
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
            bandwidth=0.22,
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
    main()# This script demonstrates the Weight Perturbation strategy for Section 3 (evidence-based perturbation without explicit target).
# It pretrains a WGAN-GP generator on a toy real data distribution (e.g., Gaussian clusters),
# samples multiple evidence domains (circularly placed clusters), estimates a virtual target via broadened KDE,
# and applies perturbation using the WeightPerturberTargetNotGiven class to align the generator's output with the evidence.
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
)=2, verbose=True):
    
    pert_gen = Generator().to(DEVICE)
    pert_gen.load_state_dict(generator.state_dict())
    params = list(pert_gen.parameters())
    theta_prev = torch.cat([p.data.flatten() for p in params])
    delta_theta_prev = torch.zeros_like(theta_prev)
    eta = eta_init
    z_eval = torch.randn(600, NOISE_DIM, device=DEVICE)
    sloss = SamplesLoss("sinkhorn", p=2, blur=0.06)
    w2_hist, best_vec, best_loss, stopped = [], None, float('inf'), False

    for ep in range(epochs):
        z = torch.randn(600, NOISE_DIM, device=DEVICE)
        virt_bw = 0.22 + 0.07 * np.exp(-ep/10)
        virtuals = virtual_target_samples(evidence_samples, bandwidth=virt_bw, N=600)
        
        pert_gen.train()
        pert_gen.zero_grad()
        x_gen = pert_gen(z)
        
        ot_loss_virtual = sloss(x_gen, virtuals)
        ot_loss_multi = 0.0
        for ev in evidence_samples:
            ot_loss_multi += sloss(x_gen, ev)
        ot_loss_multi /= len(evidence_samples)
        
        covar = torch.det(torch.cov(x_gen.T)+torch.eye(DATA_DIM, device=DEVICE)*0.025)
        entropy_reg = -lambda_entropy*torch.log(torch.clamp(covar, min=1e-5))
        total_loss = lambda_virtual*ot_loss_virtual + lambda_multi*ot_loss_multi + entropy_reg

        total_loss.backward()
        grads = torch.cat([p.grad.flatten() for p in pert_gen.parameters() if p.grad is not None])
        delta_theta = -eta * grads
        norm = delta_theta.norm()
        maxnorm = delta_theta_clip * 160
        if norm > maxnorm:
            delta_theta = delta_theta * (maxnorm / (norm+1e-8))
        delta_theta = momentum * delta_theta_prev + (1-momentum) * delta_theta
        new_vec = torch.cat([p.data.flatten() for p in pert_gen.parameters()]) + delta_theta
        ptr = 0
        for p in pert_gen.parameters():
            n = p.data.numel()
            p.data.copy_(new_vec[ptr:ptr+n].view_as(p))
            ptr += n
        delta_theta_prev = delta_theta.detach().clone()

        base_large = generator(z_eval).detach()
        pert_large = pert_gen(z_eval).detach()
        w2_base = sloss(base_large, virtuals).item()
        w2_pert = sloss(pert_large, virtuals).item()
        improvement = w2_base - w2_pert
        w2_hist.append(w2_pert)
        
        if verbose:
            print(f"[{ep:2d}] OT(Pert,Virtual)={ot_loss_virtual.item():.4f} OT(Pert,MultiEvi)={ot_loss_multi.item():.4f} Δθ={norm:.5f} eta={eta:.4f} improve={improvement:.4f}")
            
        if w2_pert < best_loss:
            best_loss, best_vec = w2_pert, torch.cat([p.data.flatten() for p in pert_gen.parameters()]).clone()
        if ep > 1 and improvement < 0.:
            eta *= 0.72
        if len(w2_hist) > early_patience:
            recent = w2_hist[-early_patience:]
            if all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
                stopped = True
        if ep % eval_every == 0 or ep == epochs-1 or stopped:
            plot_all(generator, pert_gen, virtuals, evidence_samples, centers, step_msg=ep)
        if stopped: break
        
    if best_vec is not None:
        ptr = 0
        for p in pert_gen.parameters():
            n = p.data.numel()
            p.data.copy_(best_vec[ptr:ptr+n].view_as(p))
            ptr += n
    return pert_gen

def pretrain_generator():
    generator = Generator().to(DEVICE)
    critic = Generator().to(DEVICE)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5,0.98))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5,0.98))
    
    for ep in range(300):
        for _ in range(2):
            real = torch.randn(BATCH_SIZE, DATA_DIM, device=DEVICE) * 2.8
            z = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
            fake = generator(z)
            loss_c = -critic(real).mean() + critic(fake).mean()
            optimizer_c.zero_grad(); loss_c.backward(); optimizer_c.step()
        z = torch.randn(BATCH_SIZE, NOISE_DIM, device=DEVICE)
        fake = generator(z)
        loss_g = -critic(fake).mean()
        optimizer_g.zero_grad(); loss_g.backward(); optimizer_g.step()
        if ep % 80 == 0: print(f"Pretrain {ep:3d}: G={loss_g.item():.3f}, C={loss_c.item():.3f}")
    return generator

def main():
    parser = argparse.ArgumentParser(description="Section 3: Evidence-Based Weight Perturbation (test69 style)")
    parser.add_argument("--plot", action="store_true", help="Enable plotting")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")
    parser.add_argument("--num_evidence_domains", type=int, default=3, help="Number of evidence domains")
    parser.add_argument("--samples_per_domain", type=int, default=35, help="Samples per domain")
    parser.add_argument("--random_shift", type=float, default=3.6, help="Random shift for evidence placement")
    parser.add_argument("--epochs", type=int, default=100, help="Perturbation epochs")
    args = parser.parse_args()
    
    print(f"Using device: {DEVICE}")
    
    evidence_samples, centers = sample_evidence_domains(
        num_domains=args.num_evidence_domains, 
        samples_per_domain=args.samples_per_domain, 
        random_shift=args.random_shift
    )
    
    print("Starting pretraining...")
    generator = pretrain_generator()
    print("Pretraining completed.")
    
    print("Starting perturbation...")
    perturbed_gen = barycentric_ot_perturb_improved(
        generator, evidence_samples, centers,
        epochs=args.epochs, eta_init=0.045, delta_theta_clip=0.23,
        lambda_entropy=0.012, lambda_virtual=0.8, lambda_multi=1.0, momentum=0.975,
        early_patience=6, eval_every=2, verbose=args.verbose
    )
    print("Perturbation completed.")
    
    # 최종 평가
    final_virtual = virtual_target_samples(evidence_samples, weights=None, bandwidth=0.19, N=600)
    
    z_eval = torch.randn(600, NOISE_DIM, device=DEVICE)
    sloss = SamplesLoss("sinkhorn", p=2, blur=0.06)
    
    with torch.no_grad():
        orig_out = generator(z_eval)
        pert_out = perturbed_gen(z_eval)
        
        ot_orig = sum(sloss(orig_out, ev).item() for ev in evidence_samples) / len(evidence_samples)
        ot_pert = sum(sloss(pert_out, ev).item() for ev in evidence_samples) / len(evidence_samples)
        
    improvement = (ot_orig - ot_pert) / ot_orig
    
    print(f"Multi-Marginal OT (Original to Evidence): {ot_orig:.4f}")
    print(f"Multi-Marginal OT (Perturbed to Evidence): {ot_pert:.4f}")
    print(f"Improvement: {improvement:.4f}")
    
    if args.plot:
        plot_all(generator, perturbed_gen, final_virtual, evidence_samples, centers, step_msg="Final")
        plt.savefig("section3_results.png", dpi=150)
        print("Plot saved to section3_results.png")

if __name__ == "__main__":
    main()