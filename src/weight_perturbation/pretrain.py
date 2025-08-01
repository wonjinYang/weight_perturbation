import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from typing import Callable, Optional

def compute_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """
    Compute the gradient penalty for WGAN-GP.

    This function calculates the gradient penalty term to enforce the 1-Lipschitz
    constraint on the critic. It interpolates between real and fake samples,
    computes the critic's output on these interpolates, and penalizes the L2 norm
    of the gradients if it deviates from 1.

    Args:
        critic (nn.Module): The critic (discriminator) model.
        real_samples (torch.Tensor): Batch of real samples.
        fake_samples (torch.Tensor): Batch of fake (generated) samples.
        device (torch.device): Device to perform computations on.

    Returns:
        torch.Tensor: Scalar gradient penalty value.

    Raises:
        ValueError: If real and fake samples have mismatched shapes.

    Example:
        >>> gp = compute_gradient_penalty(critic, real_data, fake_data, device)
        >>> gp.shape
        torch.Size([])
    """
    if real_samples.shape != fake_samples.shape:
        raise ValueError("Real and fake samples must have the same shape.")

    batch_size = real_samples.shape[0]
    alpha = torch.rand(batch_size, 1, 1, 1, device=device).expand_as(real_samples)

    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = Variable(interpolates, requires_grad=True)

    critic_inter = critic(interpolates)
    gradients = torch.autograd.grad(
        outputs=critic_inter,
        inputs=interpolates,
        grad_outputs=torch.ones_like(critic_inter),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]

    gradients = gradients.view(batch_size, -1)
    grad_norm = gradients.norm(2, dim=1)
    penalty = ((grad_norm - 1) ** 2).mean()

    return penalty

def pretrain_wgan_gp(
    generator: nn.Module,
    critic: nn.Module,
    real_sampler: Callable[[int, Optional[dict]], torch.Tensor],
    epochs: int = 300,
    batch_size: int = 64,
    lr: float = 2e-4,
    betas: tuple[float, float] = (0.0, 0.9),
    gp_lambda: float = 10.0,
    critic_iters: int = 5,
    noise_dim: int = 100,
    device: str = 'cpu',
    verbose: bool = True
) -> tuple[nn.Module, nn.Module]:
    """
    Pretrain a generator and critic using WGAN-GP (Wasserstein GAN with Gradient Penalty).

    This function trains the generator and critic models using the WGAN-GP algorithm.
    The critic is trained more frequently than the generator to ensure it approximates
    the Wasserstein distance accurately. Gradient penalty is used to enforce the
    1-Lipschitz constraint on the critic.

    Args:
        generator (nn.Module): Generator model to train.
        critic (nn.Module): Critic (discriminator) model to train.
        real_sampler (Callable[[int, Optional[dict]], torch.Tensor]): Function to sample real data.
            It takes batch_size and optional kwargs, returns a tensor of shape (batch_size, ...).
        epochs (int): Number of training epochs. Defaults to 300.
        batch_size (int): Batch size for training. Defaults to 64.
        lr (float): Learning rate for Adam optimizers. Defaults to 2e-4.
        betas (tuple[float, float]): Beta parameters for Adam. Defaults to (0.0, 0.9) as per WGAN-GP paper.
        gp_lambda (float): Gradient penalty coefficient. Defaults to 10.0.
        critic_iters (int): Number of critic updates per generator update. Defaults to 5.
        noise_dim (int): Dimension of the noise input to the generator. Defaults to 100.
        device (str): Device to train on ('cpu' or 'cuda'). Defaults to 'cpu'.
        verbose (bool): If True, print progress every 10 epochs. Defaults to True.

    Returns:
        tuple[nn.Module, nn.Module]: Trained generator and critic models.

    Raises:
        ValueError: If models are not on the specified device or sampler returns invalid data.

    Example:
        >>> from .models import Generator, Critic
        >>> from .samplers import sample_real_data
        >>> gen = Generator(noise_dim=100, data_dim=2, hidden_dim=256)
        >>> crit = Critic(data_dim=2, hidden_dim=256)
        >>> trained_gen, trained_crit = pretrain_wgan_gp(gen, crit, sample_real_data, epochs=100)
    """
    device = torch.device(device)
    generator.to(device)
    critic.to(device)

    if next(generator.parameters()).device != device or next(critic.parameters()).device != device:
        raise ValueError("Models must be on the specified device.")

    optim_g = optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optim_d = optim.Adam(critic.parameters(), lr=lr, betas=betas)

    generator.train()
    critic.train()

    for epoch in range(epochs):
        for _ in range(critic_iters):
            # Sample real data
            real = real_sampler(batch_size).to(device)
            if real.dim() != 2 or real.shape[0] != batch_size:
                raise ValueError("Sampler must return tensor of shape (batch_size, data_dim).")

            # Sample noise and generate fake data
            z = torch.randn(batch_size, noise_dim, device=device)
            fake = generator(z).detach()

            # Critic loss
            crit_real = critic(real).mean()
            crit_fake = critic(fake).mean()

            gp = compute_gradient_penalty(critic, real, fake, device)
            loss_d = -crit_real + crit_fake + gp_lambda * gp

            optim_d.zero_grad()
            loss_d.backward()
            optim_d.step()

        # Generator update
        z = torch.randn(batch_size, noise_dim, device=device)
        fake = generator(z)
        loss_g = -critic(fake).mean()

        optim_g.zero_grad()
        loss_g.backward()
        optim_g.step()

        if verbose and (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")

    if verbose:
        print("Pretraining completed.")

    return generator, critic
