"""
Weighted Sobolev space regularization for congested transport.

This module implements the weighted Sobolev H^1(Ω, σ) norm constraints
and regularization terms from the theoretical framework.
Modified for improved numerical stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
from typing import Dict, Optional, Tuple, Callable
import numpy as np


class WeightedSobolevRegularizer:
    """
    Weighted Sobolev H^1(Ω, σ) norm regularization for the critic.
    
    This class implements the theoretical constraint:
    ||u||_{H^1(Ω, σ)} = ∫(u² + |∇u|²)σ dx
    Modified for numerical stability.
    """
    
    def __init__(self, lambda_sobolev: float = 0.01, gradient_penalty_weight: float = 0.5):
        # Clamp parameters for stability
        self.lambda_sobolev = max(0.001, min(lambda_sobolev, 0.1))
        self.gradient_penalty_weight = max(0.1, min(gradient_penalty_weight, 2.0))
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute weighted Sobolev norm regularization with stability improvements.
        
        Args:
            critic (nn.Module): Critic network (dual potential u).
            samples (torch.Tensor): Sample points where to evaluate.
            sigma (torch.Tensor): Spatial density weights.
            return_components (bool): If True, return individual components.
        
        Returns:
            torch.Tensor: Scalar Sobolev regularization loss.
        """
        # Ensure samples require gradients
        samples = samples.detach().clone()
        samples.requires_grad_(True)
        
        try:
            # Compute critic values
            u_values = critic(samples)
            
            # Compute gradients with error handling
            gradients = torch.autograd.grad(
                outputs=u_values.sum(),
                inputs=samples,
                create_graph=True,
                retain_graph=True,
                allow_unused=True
            )[0]
            
            if gradients is None:
                gradients = torch.zeros_like(samples)
            
        except RuntimeError as e:
            print(f"Warning: Sobolev gradient computation failed: {e}")
            gradients = torch.zeros_like(samples)
            u_values = critic(samples.detach())
        
        # Clamp inputs for stability
        u_values = torch.clamp(u_values, min=-100.0, max=100.0)
        gradients = torch.clamp(gradients, min=-10.0, max=10.0)
        sigma = torch.clamp(sigma, min=1e-6, max=10.0)
        
        # Ensure sigma has correct shape
        if sigma.dim() == 1 and len(sigma) == u_values.shape[0]:
            sigma_expanded = sigma.unsqueeze(1)
        else:
            sigma_expanded = sigma
        
        # Compute weighted L2 norm of function values
        l2_term = (u_values ** 2 * sigma_expanded).mean()
        l2_term = torch.clamp(l2_term, max=1000.0)
        
        # Compute weighted L2 norm of gradients
        gradient_term = ((gradients ** 2).sum(dim=1) * sigma).mean()
        gradient_term = torch.clamp(gradient_term, max=1000.0)
        
        # Sobolev norm with stability
        sobolev_norm = l2_term + self.gradient_penalty_weight * gradient_term
        sobolev_norm = torch.clamp(sobolev_norm, max=1000.0)
        
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
    Modified for improved stability.
    """
    
    def __init__(
        self,
        lambda_sobolev: float = 0.01,
        gradient_penalty_weight: float = 0.5,
        adaptation_factor: float = 0.2,  # Reduced adaptation factor
        congestion_threshold: float = 0.2  # Higher threshold
    ):
        super().__init__(lambda_sobolev, gradient_penalty_weight)
        self.adaptation_factor = max(0.1, min(adaptation_factor, 1.0))
        self.congestion_threshold = max(0.1, congestion_threshold)
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        traffic_intensity: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute adaptive weighted Sobolev norm regularization with stability.
        
        The regularization strength adapts based on local traffic intensity:
        - Higher congestion → stronger regularization
        - Lower congestion → weaker regularization
        """
        # Base Sobolev computation
        result = super().__call__(critic, samples, sigma, return_components=True)
        
        if traffic_intensity is not None:
            try:
                # More conservative adaptive scaling
                avg_intensity = traffic_intensity.mean()
                avg_intensity = torch.clamp(avg_intensity, max=10.0)  # Cap intensity
                
                if avg_intensity > self.congestion_threshold:
                    # More conservative increase in regularization
                    adaptation_scale = 1.0 + self.adaptation_factor * (avg_intensity - self.congestion_threshold) / 10.0
                else:
                    # Maintain regularization in low congestion areas
                    adaptation_scale = 1.0 - 0.05 * self.adaptation_factor
                
                adaptation_scale = max(0.5, min(adaptation_scale, 2.0))  # Clamp scaling
                result['total'] = result['total'] * adaptation_scale
                result['adaptation_scale'] = adaptation_scale
            except Exception as e:
                print(f"Warning: Adaptive Sobolev scaling failed: {e}")
                result['adaptation_scale'] = 1.0
        
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
    Compute weighted Sobolev norm regularization for the critic with stability.
    
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
    # Clamp lambda_sobolev for stability
    lambda_sobolev = max(0.001, min(lambda_sobolev, 0.1))
    regularizer = WeightedSobolevRegularizer(lambda_sobolev)
    return regularizer(critic, samples, sigma)


class SobolevConstraintProjection:
    """
    Project critic parameters to satisfy Sobolev norm constraints.
    
    This implements a projection step to ensure the critic stays within
    the admissible Sobolev space during training.
    Modified for improved stability.
    """
    
    def __init__(self, sobolev_bound: float = 2.0, projection_freq: int = 20):  # Higher bound, less frequent
        self.sobolev_bound = max(1.0, sobolev_bound)
        self.projection_freq = max(10, projection_freq)
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
        
        try:
            with torch.no_grad():
                # Compute current Sobolev norm
                regularizer = WeightedSobolevRegularizer(lambda_sobolev=1.0)
                current_norm = regularizer(critic, test_samples, sigma).item()
                
                if current_norm <= self.sobolev_bound:
                    return {'projected': False, 'norm_before': current_norm, 'norm_after': current_norm}
                
                # More conservative scaling
                scale_factor = min(0.9, np.sqrt(self.sobolev_bound / (current_norm + 1e-8)))
                scale_factor = max(0.5, scale_factor)  # Don't scale too aggressively
                
                for param in critic.parameters():
                    if param.requires_grad:
                        param.data *= scale_factor
                
                # Verify projection
                new_norm = regularizer(critic, test_samples, sigma).item()
                
                return {
                    'projected': True,
                    'norm_before': current_norm,
                    'norm_after': new_norm,
                    'scale_factor': scale_factor
                }
        except Exception as e:
            print(f"Warning: Sobolev projection failed: {e}")
            return {'projected': False, 'norm_before': 0.0, 'norm_after': 0.0}


class SobolevConstrainedCritic(nn.Module):
    """
    Critic network with built-in Sobolev space constraints.
    
    This extends the basic critic with automatic Sobolev regularization
    and optional spectral normalization.
    Modified for improved stability.
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        activation=None,
        use_spectral_norm: bool = True,
        lambda_sobolev: float = 0.01,
        sobolev_bound: float = 2.0  # Higher bound for stability
    ):
        super().__init__()
        
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.use_spectral_norm = use_spectral_norm
        self.lambda_sobolev = max(0.001, min(lambda_sobolev, 0.1))
        
        # Build network layers
        layers = []
        
        # First layer
        linear1 = nn.Linear(data_dim, hidden_dim)
        if use_spectral_norm:
            linear1 = spectral_norm(linear1)
        layers.extend([linear1, activation])
        
        # Hidden layers
        for _ in range(2):
            linear_hidden = nn.Linear(hidden_dim, hidden_dim)
            if use_spectral_norm:
                linear_hidden = spectral_norm(linear_hidden)
            layers.extend([linear_hidden, activation])
        
        # Output layer
        linear_out = nn.Linear(hidden_dim, 1)
        if use_spectral_norm:
            linear_out = spectral_norm(linear_out)
        layers.append(linear_out)
        
        self.model = nn.Sequential(*layers)
        
        # Initialize Sobolev regularizer and projection
        self.sobolev_regularizer = WeightedSobolevRegularizer(self.lambda_sobolev)
        self.sobolev_projection = SobolevConstraintProjection(sobolev_bound)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for Linear layers with smaller variance."""
        if isinstance(module, nn.Linear):
            # Use smaller initialization for stability
            nn.init.normal_(module.weight, 0.0, 0.01)  # Reduced from 0.02
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic with output clamping."""
        output = self.model(x)
        # Clamp output to prevent explosion
        return torch.clamp(output, min=-50.0, max=50.0)
    
    def sobolev_regularization_loss(
        self,
        samples: torch.Tensor,
        sigma: torch.Tensor
    ) -> torch.Tensor:
        """Compute Sobolev regularization loss with stability."""
        try:
            return self.sobolev_regularizer(self, samples, sigma)
        except Exception as e:
            print(f"Warning: Sobolev regularization failed: {e}")
            return torch.tensor(0.0, device=samples.device)
    
    def project_to_sobolev_ball(
        self,
        test_samples: torch.Tensor,
        sigma: torch.Tensor
    ) -> Dict[str, float]:
        """Project parameters to satisfy Sobolev constraint."""
        try:
            return self.sobolev_projection.project(self, test_samples, sigma)
        except Exception as e:
            print(f"Warning: Sobolev projection failed: {e}")
            return {'projected': False, 'norm_before': 0.0, 'norm_after': 0.0}


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
    Compute Sobolev-weighted gradient penalty for WGAN-GP with stability.
    
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
    # Clamp lambda_sobolev for stability
    lambda_sobolev = max(0.001, min(lambda_sobolev, 0.1))
    
    batch_size = real_samples.shape[0]
    device = real_samples.device
    
    try:
        # Create interpolation weights
        if interpolation_lambda is None:
            alpha = torch.rand(batch_size, 1, device=device)
            alpha = alpha.expand_as(real_samples)
        else:
            alpha = interpolation_lambda.view(-1, 1).expand_as(real_samples)
        
        # Interpolate samples and densities
        interpolated_samples = alpha * real_samples + (1 - alpha) * fake_samples
        interpolated_sigma = alpha.mean(dim=1) * sigma_real + (1 - alpha.mean(dim=1)) * sigma_fake
        
        # Clamp interpolated values
        interpolated_sigma = torch.clamp(interpolated_sigma, min=1e-6, max=10.0)
        
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
            only_inputs=True,
            allow_unused=True
        )[0]
        
        if gradients is None:
            return torch.tensor(0.0, device=device)
        
        # Compute weighted gradient penalty with stability
        gradient_norm = gradients.norm(2, dim=1)
        gradient_norm = torch.clamp(gradient_norm, max=10.0)
        
        # Standard gradient penalty term
        gp_standard = ((gradient_norm - 1) ** 2).mean()
        
        # Sobolev-weighted term with reduced impact
        gp_sobolev = ((gradient_norm ** 2) * interpolated_sigma).mean()
        
        # Combine with reduced Sobolev weight
        total_gp = gp_standard + 0.1 * lambda_sobolev * gp_sobolev
        return torch.clamp(total_gp, max=100.0)
        
    except Exception as e:
        print(f"Warning: Sobolev gradient penalty computation failed: {e}")
        return torch.tensor(0.1, device=device)



class SobolevWGANLoss:
    """
    Complete WGAN-GP loss with Sobolev space regularization.
    Modified for improved stability.
    """
    
    def __init__(
        self,
        lambda_gp: float = 0.5,
        lambda_sobolev: float = 0.01,
        use_adaptive_sobolev: bool = True
    ):
        # Clamp parameters for stability
        self.lambda_gp = max(0.1, min(lambda_gp, 2.0))
        self.lambda_sobolev = max(0.001, min(lambda_sobolev, 0.1))
        self.use_adaptive_sobolev = use_adaptive_sobolev
        
        if use_adaptive_sobolev:
            self.sobolev_regularizer = AdaptiveSobolevRegularizer(self.lambda_sobolev)
        else:
            self.sobolev_regularizer = WeightedSobolevRegularizer(self.lambda_sobolev)
    
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
        Compute critic loss with Sobolev regularization and stability.
        
        Returns:
            Dict containing individual loss components.
        """
        try:
            # Standard WGAN critic loss with clamping
            critic_real = critic(real_samples).mean()
            critic_fake = critic(fake_samples).mean()
            critic_real = torch.clamp(critic_real, min=-100.0, max=100.0)
            critic_fake = torch.clamp(critic_fake, min=-100.0, max=100.0)
            wasserstein_loss = critic_fake - critic_real
            
            # Sobolev gradient penalty with reduced weight
            sobolev_gp = compute_sobolev_gradient_penalty(
                critic, real_samples, fake_samples,
                sigma_real, sigma_fake, self.lambda_sobolev
            )
            
            # Sobolev regularization on real samples with reduced weight
            if traffic_intensity is not None and self.use_adaptive_sobolev:
                sobolev_reg = self.sobolev_regularizer(
                    critic, real_samples, sigma_real, traffic_intensity
                )
            else:
                sobolev_reg = self.sobolev_regularizer(critic, real_samples, sigma_real)
            
            # Total critic loss with reduced regularization weights
            total_loss = wasserstein_loss + 0.5 * self.lambda_gp * sobolev_gp + 0.1 * sobolev_reg
            total_loss = torch.clamp(total_loss, max=1000.0)
            
            return {
                'total': total_loss,
                'wasserstein': wasserstein_loss,
                'gradient_penalty': sobolev_gp,
                'sobolev_regularization': sobolev_reg
            }
            
        except Exception as e:
            print(f"Warning: Sobolev WGAN critic loss computation failed: {e}")
            # Fallback to simple loss
            try:
                critic_real = critic(real_samples).mean()
                critic_fake = critic(fake_samples).mean()
                simple_loss = critic_fake - critic_real
                return {
                    'total': simple_loss,
                    'wasserstein': simple_loss,
                    'gradient_penalty': torch.tensor(0.0, device=real_samples.device),
                    'sobolev_regularization': torch.tensor(0.0, device=real_samples.device)
                }
            except:
                return {
                    'total': torch.tensor(0.0, device=real_samples.device),
                    'wasserstein': torch.tensor(0.0, device=real_samples.device),
                    'gradient_penalty': torch.tensor(0.0, device=real_samples.device),
                    'sobolev_regularization': torch.tensor(0.0, device=real_samples.device)
                }
    
    def generator_loss(self, critic: nn.Module, fake_samples: torch.Tensor) -> torch.Tensor:
        """Compute generator loss with stability."""
        try:
            loss = -critic(fake_samples).mean()
            return torch.clamp(loss, max=1000.0)
        except Exception as e:
            print(f"Warning: Sobolev WGAN generator loss computation failed: {e}")
            return torch.tensor(0.0, device=fake_samples.device)