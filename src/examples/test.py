import torch
import numpy as np
import matplotlib.pyplot as plt
from geomloss import SamplesLoss
from weight_perturbation import (
    Generator, Critic, sample_evidence_domains, set_seed, compute_device
)

NOISE_DIM, DATA_DIM, HIDDEN_DIM = 2, 2, 256
BATCH_SIZE = 96

def kde_sampler(evidence, bandwidth=0.22, num_samples=160, device='cpu'):
    idx = torch.randint(0, evidence.shape[0], (num_samples,), device=device)
    means = evidence[idx]
    return means + bandwidth * torch.randn(num_samples, DATA_DIM, device=device)

def virtual_target_sampler_multi(evidence_samples, weights=None, bandwidth=0.22, num_samples=600, device='cpu'):
    num_domains = len(evidence_samples)
    if weights is None:
        weights = [1.0/num_domains]*num_domains
    choices = np.random.choice(num_domains, size=num_samples, p=weights)
    chunks = [np.sum(choices == i) for i in range(num_domains)]
    res = []
    for i, n in enumerate(chunks):
        if n > 0:
            res.append(kde_sampler(evidence_samples[i], bandwidth, n, device=device))
    return torch.cat(res, dim=0)

def plot_all(generator, perturbed_gen, virtual_samples, evidence_samples, centers, step_msg=""):
    z = torch.randn(1200, NOISE_DIM, device=next(generator.parameters()).device)
    orig = generator(z).detach().cpu().numpy()
    pert = perturbed_gen(z).detach().cpu().numpy()
    virt = virtual_samples.cpu().numpy()
    plt.figure(figsize=(8,8))
    plt.scatter(orig[:,0], orig[:,1], c='tab:blue', label='Original', alpha=0.11)
    plt.scatter(virt[:,0], virt[:,1], c='tab:green', label='Virtual Target', alpha=0.19)
    plt.scatter(pert[:,0], pert[:,1], c='tab:red', label='Perturbed', alpha=0.13)
    evidence_colors = ['orange', 'purple', 'cyan', 'brown']
    for idx, (ev, c) in enumerate(zip(evidence_samples, centers)):
        ev_np = ev.cpu().numpy()
        plt.scatter(ev_np[:,0], ev_np[:,1], color=evidence_colors[idx%len(evidence_colors)],
                    label=f"Evidence Domain {idx+1}", s=55, marker='x')
        plt.scatter([c[0]], [c[1]], color=evidence_colors[idx%len(evidence_colors)],
                    s=105, marker='*', edgecolor='black')
    plt.legend()
    plt.title(f"Barycentric Perturbed OT (Multi-evidence) [Step {step_msg}]")
    plt.grid(True)
    plt.axis('equal')
    plt.tight_layout()
    plt.savefig(f"section3_result_{step_msg}.png", dpi=150, bbox_inches='tight')
    print(f"Plot saved to section3_result_{step_msg}.png")
    plt.close()

def sample_real_mixture(batch_size, centers, std=0.7, device='cpu'):
    num_centers = len(centers)
    cnts = [batch_size // num_centers] * num_centers
    for i in range(batch_size % num_centers):
        cnts[i] += 1
    data = []
    for n, c in zip(cnts, centers):
        center_tensor = torch.tensor(c, device=device, dtype=torch.float32)
        data.append(center_tensor + std * torch.randn(n, DATA_DIM, device=device))
    return torch.cat(data, dim=0)

def pretrain_generator(device, centers):
    generator = Generator(NOISE_DIM, DATA_DIM, HIDDEN_DIM).to(device)
    critic = Critic(DATA_DIM, HIDDEN_DIM).to(device)
    optimizer_g = torch.optim.Adam(generator.parameters(), lr=2e-4, betas=(0.5,0.98))
    optimizer_c = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5,0.98))
    for ep in range(300):
        for _ in range(2):
            real = sample_real_mixture(BATCH_SIZE, centers, 0.7, device)
            z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
            fake = generator(z).detach()
            loss_c = -critic(real).mean() + critic(fake).mean()
            optimizer_c.zero_grad(); loss_c.backward(); optimizer_c.step()
        z = torch.randn(BATCH_SIZE, NOISE_DIM, device=device)
        fake = generator(z)
        loss_g = -critic(fake).mean()
        optimizer_g.zero_grad(); loss_g.backward(); optimizer_g.step()
        if ep % 80 == 0:
            print(f"Pretrain {ep:3d}: G={loss_g.item():.3f}, C={loss_c.item():.3f}")
    print("Pretraining completed.")
    return generator

def barycentric_ot_perturb(
    generator, evidence_samples, centers, device, epochs=100, eta_init=0.045, delta_theta_clip=0.23,
    lambda_entropy=0.012, lambda_virtual=0.8, lambda_multi=1.0, momentum=0.975,
    early_patience=6, eval_every=2
):
    pert_gen = Generator(NOISE_DIM, DATA_DIM, HIDDEN_DIM).to(device)
    pert_gen.load_state_dict(generator.state_dict())
    params = list(pert_gen.parameters())
    theta_prev = torch.cat([p.data.flatten() for p in params])
    delta_theta_prev = torch.zeros_like(theta_prev)
    eta = eta_init
    z_eval = torch.randn(600, NOISE_DIM, device=device)
    sloss = SamplesLoss("sinkhorn", p=2, blur=0.06)
    w2_hist, best_vec, best_loss, stopped = [], None, float('inf'), False
    for ep in range(epochs):
        z = torch.randn(600, NOISE_DIM, device=device)
        virt_bw = 0.22 + 0.07 * np.exp(-ep/10)
        virtuals = virtual_target_sampler_multi(evidence_samples, bandwidth=virt_bw, num_samples=600, device=device)
        x_gen = pert_gen(z)
        ot_loss_virtual = sloss(x_gen, virtuals)
        ot_loss_multi = sum([sloss(x_gen, ev) for ev in evidence_samples]) / len(evidence_samples)
        covar = torch.det(torch.cov(x_gen.T)+torch.eye(DATA_DIM, device=device)*0.025)
        entropy_reg = -lambda_entropy*torch.log(torch.clamp(covar, min=1e-5))
        total_loss = lambda_virtual*ot_loss_virtual + lambda_multi*ot_loss_multi + entropy_reg
        pert_gen.zero_grad(); total_loss.backward()
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

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--num_evidence_domains", type=int, default=3)
    parser.add_argument("--samples_per_domain", type=int, default=35)
    parser.add_argument("--random_shift", type=float, default=3.6)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    DEVICE = torch.device(args.device if args.device else compute_device())
    set_seed(args.seed)
    print(f"Device: {DEVICE}")
    evidence_samples, centers = sample_evidence_domains(
        num_domains=args.num_evidence_domains,
        samples_per_domain=args.samples_per_domain,
        random_shift=args.random_shift,
        std=0.7,
        device=DEVICE
    )
    generator = pretrain_generator(DEVICE, centers)
    perturbed_gen = barycentric_ot_perturb(
        generator, evidence_samples, centers, device=DEVICE,
        epochs=args.epochs, eta_init=0.045, delta_theta_clip=0.23,
        lambda_entropy=0.012, lambda_virtual=0.8, lambda_multi=1.0, momentum=0.975,
        early_patience=6, eval_every=2
    )
    if args.plot:
        final_virtual = virtual_target_sampler_multi(evidence_samples, bandwidth=0.19, num_samples=600, device=DEVICE)
        plot_all(generator, perturbed_gen, final_virtual, evidence_samples, centers, step_msg="Final")

if __name__ == "__main__":
    main()
