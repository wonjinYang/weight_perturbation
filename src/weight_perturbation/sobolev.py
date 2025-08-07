"""
Weighted Sobolev space regularization for congested transport.

This module implements the weighted Sobolev H^1(Ω, σ) norm constraints
and regularization terms from the theoretical framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, Callable
import numpy as np


class WeightedSobolevRegularizer:
    """
    Weighted Sobolev H^1(Ω, σ) norm regularization for the critic.
    
    This class implements the theoretical constraint:
    ||u||_{H^1(Ω, σ)} = ∫(u² + |∇u|²)σ dx
    """
    
    def __init__(self, lambda_sobolev: float = 0.01, gradient_penalty_weight: float = 1.0):
        self.lambda_sobolev = lambda_sobolev
        self.gradient_penalty_weight = gradient_penalty_weight
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute weighted Sobolev norm regularization.
        
        Args:
            critic (nn.Module): Critic network (dual potential u).
            samples (torch.Tensor): Sample points where to evaluate.
            sigma (torch.Tensor): Spatial density weights.
            return_components (bool): If True, return individual components.
        
        Returns:
            torch.Tensor: Scalar Sobolev regularization loss.
        """
        samples.requires_grad_(True)
        
        # Compute critic values
        u_values = critic(samples)
        
        # Compute gradients
        gradients = torch.autograd.grad(
            outputs=u_values.sum(),
            inputs=samples,
            create_graph=True,
            retain_graph=True
        )[0]
        
        # Compute weighted L2 norm of function values
        l2_term = (u_values ** 2 * sigma.unsqueeze(1)).mean()
        
        # Compute weighted L2 norm of gradients
        gradient_term = ((gradients ** 2).sum(dim=1) * sigma).mean()
        
        # Sobolev norm
        sobolev_norm = l2_term + self.gradient_penalty_weight * gradient_term
        
        if return_components:
            return {
                'total': self.lambda_sobolev * sobolev_norm,
                'l2_term': l2_term,
                'gradient_term': gradient_term,
                'sobolev_norm': sobolev_norm
            }
        
        return self.lambda_sobolev * sobolev_norm


class AdaptiveSobolevRegularizer(WeightedSobolevRegularizer):
    """
    Adaptive weighted Sobolev regularizer that adjusts based on congestion levels.
    """
    
    def __init__(
        self,
        lambda_sobolev: float = 0.01,
        gradient_penalty_weight: float = 1.0,
        adaptation_factor: float = 0.5,
        congestion_threshold: float = 0.1
    ):
        super().__init__(lambda_sobolev, gradient_penalty_weight)
        self.adaptation_factor = adaptation_factor
        self.congestion_threshold = congestion_threshold
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        traffic_intensity: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute adaptive weighted Sobolev norm regularization.
        
        The regularization strength adapts based on local traffic intensity:
        - Higher congestion → stronger regularization
        - Lower congestion → weaker regularization
        """
        # Base Sobolev computation
        result = super().__call__(critic, samples, sigma, return_components=True)
        
        if traffic_intensity is not None:
            # Adaptive scaling based on congestion
            avg_intensity = traffic_intensity.mean()
            if avg_intensity > self.congestion_threshold:
                # Increase regularization in high congestion areas
                adaptation_scale = 1.0 + self.adaptation_factor * (avg_intensity - self.congestion_threshold)
            else:
                # Maintain or slightly reduce regularization in low congestion areas
                adaptation_scale = 1.0 - 0.1 * self.adaptation_factor
            
            result['total'] = result['total'] * adaptation_scale
            result['adaptation_scale'] = adaptation_scale
        
        if return_components:
            return result
        else:
            return result['total']


def sobolev_regularization(
    critic: torch.nn.Module,
    samples: torch.Tensor,
    sigma: torch.Tensor,
    lambda_sobolev: float = 0.01
) -> torch.Tensor:
    """
    Compute weighted Sobolev norm regularization for the critic.
    
    Implements the H^1(Ω, σ) norm constraint:
    ||u||_{H^1(Ω, σ)} = ∫(u² + |∇u|²)σ dx
    
    Args:
        critic (torch.nn.Module): Critic network (dual potential u).
        samples (torch.Tensor): Sample points where to evaluate.
        sigma (torch.Tensor): Spatial density weights.
        lambda_sobolev (float): Regularization strength. Defaults to 0.01.
    
    Returns:
        torch.Tensor: Scalar Sobolev regularization loss.
    """
    regularizer = WeightedSobolevRegularizer(lambda_sobolev)
    return regularizer(critic, samples, sigma)


class SobolevConstraintProjection:
    """
    Project critic parameters to satisfy Sobolev norm constraints.
    
    This implements a projection step to ensure the critic stays within
    the admissible Sobolev space during training.
    """
    
    def __init__(self, sobolev_bound: float = 1.0, projection_freq: int = 10):
        self.sobolev_bound = sobolev_bound
        self.projection_freq = projection_freq
        self.step_count = 0
    
    def project(
        self,
        critic: nn.Module,
        test_samples: torch.Tensor,
        sigma: torch.Tensor
    ) -> Dict[str, float]:
        """
        Project critic parameters to satisfy Sobolev bound.
        
        Args:
            critic (nn.Module): Critic to project.
            test_samples (torch.Tensor): Test samples for norm computation.
            sigma (torch.Tensor): Spatial density weights.
        
        Returns:
            Dict[str, float]: Projection statistics.
        """
        self.step_count += 1
        
        if self.step_count % self.projection_freq != 0:
            return {'projected': False, 'norm_before': 0.0, 'norm_after': 0.0}
        
        with torch.no_grad():
            # Compute current Sobolev norm
            regularizer = WeightedSobolevRegularizer(lambda_sobolev=1.0)
            current_norm = regularizer(critic, test_samples, sigma).item()
            
            if current_norm <= self.sobolev_bound:
                return {'projected': False, 'norm_before': current_norm, 'norm_after': current_norm}
            
            # Project by scaling parameters
            scale_factor = np.sqrt(self.sobolev_bound / current_norm)
            
            for param in critic.parameters():
                param.data *= scale_factor
            
            # Verify projection
            new_norm = regularizer(critic, test_samples, sigma).item()
            
            return {
                'projected': True,
                'norm_before': current_norm,
                'norm_after': new_norm,
                'scale_factor': scale_factor
            }


class SpectralNormalization(nn.Module):
    """
    Spectral normalization to control Lipschitz constant of critic.
    
    This provides an alternative approach to Sobolev regularization
    by directly constraining the spectral norm of weight matrices.
    """
    
    def __init__(self, module: nn.Module, name: str = 'weight', n_power_iterations: int = 1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        # Initialize spectral norm computation
        weight = getattr(module, name)
        with torch.no_grad():
            height = weight.data.shape[0]
            width = weight.data.view(height, -1).shape[1]
            
            u = nn.Parameter(torch.randn(height, 1), requires_grad=False)
            v = nn.Parameter(torch.randn(width, 1), requires_grad=False)
            
            self.register_parameter(name + "_u", u)
            self.register_parameter(name + "_v", v)
    
    def forward(self, *args, **kwargs):
        """Apply spectral normalization during forward pass."""
        weight = getattr(self.module, self.name)
        u = getattr(self, self.name + "_u")
        v = getattr(self, self.name + "_v")
        
        # Power iteration
        with torch.no_grad():
            for _ in range(self.n_power_iterations):
                v = F.normalize(torch.mv(weight.data.t(), u), dim=0, out=v)
                u = F.normalize(torch.mv(weight.data, v), dim=0, out=u)
        
        # Compute spectral norm
        sigma = torch.dot(u.flatten(), torch.mv(weight, v))
        
        # Normalize weight by spectral norm
        weight_normalized = weight / sigma
        
        # Temporarily replace weight
        setattr(self.module, self.name, weight_normalized)
        
        # Forward pass
        result = self.module(*args, **kwargs)
        
        # Restore original weight
        setattr(self.module, self.name, weight)
        
        return result


def apply_spectral_norm(module: nn.Module, name: str = 'weight') -> nn.Module:
    """
    Apply spectral normalization to a module.
    
    Args:
        module (nn.Module): Module to normalize.
        name (str): Name of weight parameter.
    
    Returns:
        nn.Module: Module wrapped with spectral normalization.
    """
    return SpectralNormalization(module, name)


class SobolevConstrainedCritic(nn.Module):
    """
    Critic network with built-in Sobolev space constraints.
    
    This extends the basic critic with automatic Sobolev regularization
    and optional spectral normalization.
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        activation=None,
        use_spectral_norm: bool = True,
        lambda_sobolev: float = 0.01,
        sobolev_bound: float = 1.0
    ):
        super().__init__()
        
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.use_spectral_norm = use_spectral_norm
        self.lambda_sobolev = lambda_sobolev
        
        # Build network layers
        layers = []
        
        # First layer
        linear1 = nn.Linear(data_dim, hidden_dim)
        if use_spectral_norm:
            linear1 = apply_spectral_norm(linear1)
        layers.extend([linear1, activation])
        
        # Hidden layers
        for _ in range(2):
            linear_hidden = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                linear_hidden = apply_spectral_norm(linear_hidden)
            layers.extend([linear_hidden, activation])
        
        # Output layer
        linear_out = nn.Linear(hidden_dim, 1)
        if use_spectral_norm:
            linear_out = apply_spectral_norm(linear_out)
        layers.append(linear_out)
        
        self.model = nn.Sequential(*layers)
        
        # Initialize Sobolev regularizer and projection
        self.sobolev_regularizer = WeightedSobolevRegularizer(lambda_sobolev)
        self.sobolev_projection = SobolevConstraintProjection(sobolev_bound)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for Linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic."""
        return self.model(x)
    
    def sobolev_regularization_loss(
        self,
        samples: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute Sobolev regularization loss."""
        return self.sobolev_regularizer(self, samples, sigma)
    
    def project_to_sobolev_ball(
        self,
        test_samples: torch.Tensor,
        sigma: torch.Tensor
    ) -> Dict[str, float]:
        """Project parameters to satisfy Sobolev constraint."""
        return self.sobolev_projection.project(self, test_samples, sigma)


def compute_sobolev_gradient_penalty(
    critic: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    sigma_real: torch.Tensor,
    sigma_fake: torch.Tensor,
    lambda_sobolev: float = 0.01,
    interpolation_lambda: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Sobolev-weighted gradient penalty for WGAN-GP.
    
    This extends the standard gradient penalty with Sobolev space constraints.
    
    Args:
        critic (nn.Module): Critic network.
        real_samples (torch.Tensor): Real data samples.
        fake_samples (torch.Tensor): Generated samples.
        sigma_real (torch.Tensor): Spatial density at real samples.
        sigma_fake (torch.Tensor): Spatial density at fake samples.
        lambda_sobolev (float): Sobolev regularization strength.
        interpolation_lambda (Optional[torch.Tensor]): Interpolation weights.
    
    Returns:
        torch.Tensor: Sobolev-weighted gradient penalty.
    """
    batch_size = real_samples.shape[0]
    device = real_samples.device
    
    # Create interpolation weights
    if interpolation_lambda is None:
        alpha = torch.rand(batch_size, 1, device=device)
        alpha = alpha.expand_as(real_samples)
    else:
        alpha = interpolation_lambda.view(-1, 1).expand_as(real_samples)
    
    # Interpolate samples and densities
    interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated_sigma = alpha.mean(dim=1) * sigma_real + (1 - alpha.mean(dim=1)) * sigma_fake
    
    interpolated_samples.requires_grad_(True)
    
    # Compute critic output
    critic_interpolated = critic(interpolated_samples)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=critic_interpolated,
        inputs=interpolated_samples,
        grad_outputs=torch.ones_like(critic_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Compute weighted gradient penalty
    gradient_norm = gradients.norm(2, dim=1)
    
    # Standard gradient penalty term
    gp_standard = ((gradient_norm - 1) ** 2).mean()
    
    # Sobolev-weighted term
    gp_sobolev = ((gradient_norm ** 2) * interpolated_sigma).mean()
    
    return gp_standard + lambda_sobolev * gp_sobolev


class SobolevWGANLoss:
    """
    Complete WGAN-GP loss with Sobolev space regularization.
    """
    
    def __init__(
        self,
        lambda_gp: float = 10.0,
        lambda_sobolev: float = 0.01,
        use_adaptive_sobolev: bool = True
    ):
        self.lambda_gp = lambda_gp
        self.lambda_sobolev = lambda_sobolev
        self.use_adaptive_sobolev = use_adaptive_sobolev
        
        if use_adaptive_sobolev:
            self.sobolev_regularizer = AdaptiveSobolevRegularizer(lambda_sobolev)
        else:
            self.sobolev_regularizer = WeightedSobolevRegularizer(lambda_sobolev)
    
    def critic_loss(
        self,
        critic: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        sigma_real: torch.Tensor,
        sigma_fake: torch.Tensor,
        traffic_intensity: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute critic loss with Sobolev regularization.
        
        Returns:
            Dict containing individual loss components.
        """
        # Standard WGAN critic loss
        critic_real = critic(real_samples).mean()
        critic_fake = critic(fake_samples).mean()
        wasserstein_loss = critic_fake - critic_real
        
        # Sobolev gradient penalty
        sobolev_gp = compute_sobolev_gradient_penalty(
            critic, real_samples, fake_samples,
            sigma_real, sigma_fake, self.lambda_sobolev
        )
        
        # Sobolev regularization on real samples
        if traffic_intensity is not None and self.use_adaptive_sobolev:
            sobolev_reg = self.sobolev_regularizer(
                critic, real_samples, sigma_real, traffic_intensity
            )
        else:
            sobolev_reg = self.sobolev_regularizer(critic, real_samples, sigma_real)
        
        # Total critic loss
        total_loss = wasserstein_loss + self.lambda_gp * sobolev_gp + sobolev_reg
        
        return {
            'total': total_loss,
            'wasserstein': wasserstein_loss,
            'gradient_penalty': sobolev_gp,
            'sobolev_regularization': sobolev_reg
        }
    
    def generator_loss(self, critic: nn.Module, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute generator loss."""
        return -critic(fake_samples).mean()