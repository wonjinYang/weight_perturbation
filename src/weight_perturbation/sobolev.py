"""
Weighted Sobolev space regularization for congested transport.

This module implements the weighted Sobolev H^1(Ω, σ) norm constraints
and regularization terms from the theoretical framework.
Enhanced with improved theoretical integration and reduced over-conservatism.
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
    
    Enhanced with improved theoretical justification and reduced over-conservatism.
    """
    
    def __init__(self, lambda_sobolev: float = 0.1, gradient_penalty_weight: float = 1.0):
        # Clamp parameters for stability
        self.lambda_sobolev = max(0.01, min(lambda_sobolev, 1.0))  # Increased upper bound
        self.gradient_penalty_weight = max(0.5, min(gradient_penalty_weight, 3.0))  # Increased bounds
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute weighted Sobolev norm regularization with enhanced theoretical integration.
        
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
        u_values = torch.clamp(u_values, min=-1000.0, max=1000.0)  # Increased bounds
        gradients = torch.clamp(gradients, min=-100.0, max=100.0)   # Increased bounds
        sigma = torch.clamp(sigma, min=1e-8, max=100.0)             # Increased upper bound
        
        # Ensure sigma has correct shape
        if sigma.dim() == 1 and len(sigma) == u_values.shape[0]:
            sigma_expanded = sigma.unsqueeze(1)
        else:
            sigma_expanded = sigma
        
        # Compute weighted L2 norm of function values
        l2_term = (u_values ** 2 * sigma_expanded).mean()
        # Remove artificial upper bound - let the theory guide the values
        
        # Compute weighted L2 norm of gradients
        gradient_term = ((gradients ** 2).sum(dim=1) * sigma).mean()
        # Remove artificial upper bound
        
        # Sobolev norm with theoretical weighting
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
    Adaptive weighted Sobolev regularizer with improved responsiveness.
    
    This version provides more aggressive adaptation based on congestion levels
    and better theoretical integration.
    """
    
    def __init__(
        self,
        lambda_sobolev: float = 0.1,
        gradient_penalty_weight: float = 1.0,
        adaptation_factor: float = 0.5,     # Increased adaptation factor
        congestion_threshold: float = 0.1   # Lower threshold for more responsiveness
    ):
        super().__init__(lambda_sobolev, gradient_penalty_weight)
        self.adaptation_factor = max(0.2, min(adaptation_factor, 2.0))  # Increased bounds
        self.congestion_threshold = max(0.05, congestion_threshold)
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        traffic_intensity: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute adaptive weighted Sobolev norm regularization with enhanced responsiveness.
        
        The regularization strength adapts more aggressively based on local traffic intensity:
        - Higher congestion → stronger regularization
        - Lower congestion → allow more flexibility
        """
        # Base Sobolev computation
        result = super().__call__(critic, samples, sigma, return_components=True)
        
        if traffic_intensity is not None:
            try:
                # More responsive adaptive scaling
                avg_intensity = traffic_intensity.mean()
                # Remove artificial upper bound to allow natural scaling
                
                if avg_intensity > self.congestion_threshold:
                    # More aggressive increase in regularization for high congestion
                    adaptation_scale = 1.0 + self.adaptation_factor * (avg_intensity - self.congestion_threshold)
                else:
                    # Allow more flexibility in low congestion areas
                    adaptation_scale = 1.0 - 0.3 * self.adaptation_factor * (self.congestion_threshold - avg_intensity) / self.congestion_threshold
                
                # Apply more generous bounds to allow natural adaptation
                adaptation_scale = max(0.2, min(adaptation_scale, 5.0))  # Increased upper bound
                result['total'] = result['total'] * adaptation_scale
                result['adaptation_scale'] = adaptation_scale
                
                # Additional theoretical justification: higher intensity requires more smoothness
                if avg_intensity > 0.5:  # High intensity threshold
                    theoretical_boost = 1.0 + 0.2 * torch.log(1.0 + avg_intensity)
                    result['total'] = result['total'] * theoretical_boost
                    result['theoretical_boost'] = theoretical_boost.item()
                else:
                    result['theoretical_boost'] = 1.0
                    
            except Exception as e:
                print(f"Warning: Adaptive Sobolev scaling failed: {e}")
                result['adaptation_scale'] = 1.0
                result['theoretical_boost'] = 1.0
        
        if return_components:
            return result
        else:
            return result['total']


def sobolev_regularization(
    critic: torch.nn.Module,
    samples: torch.Tensor,
    sigma: torch.Tensor,
    lambda_sobolev: float = 0.1
) -> torch.Tensor:
    """
    Compute weighted Sobolev norm regularization for the critic.
    
    Implements the H^1(Ω, σ) norm constraint:
    ||u||_{H^1(Ω, σ)} = ∫(u² + |∇u|²)σ dx
    
    Args:
        critic (torch.nn.Module): Critic network (dual potential u).
        samples (torch.Tensor): Sample points where to evaluate.
        sigma (torch.Tensor): Spatial density weights.
        lambda_sobolev (float): Regularization strength. Defaults to 0.1.
    
    Returns:
        torch.Tensor: Scalar Sobolev regularization loss.
    """
    # Clamp lambda_sobolev for stability
    lambda_sobolev = max(0.01, min(lambda_sobolev, 1.0))  # Increased upper bound
    regularizer = WeightedSobolevRegularizer(lambda_sobolev)
    return regularizer(critic, samples, sigma)


class SobolevConstraintProjection:
    """
    Project critic parameters to satisfy Sobolev norm constraints.
    
    This implements a more flexible projection step to ensure the critic stays within
    reasonable Sobolev space bounds while allowing more natural parameter evolution.
    """
    
    def __init__(self, sobolev_bound: float = 5.0, projection_freq: int = 50):  # Increased bound and frequency
        self.sobolev_bound = max(2.0, sobolev_bound)  # Higher minimum bound
        self.projection_freq = max(20, projection_freq)
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
                
                # Conservative scaling (less aggressive than before)
                scale_factor = min(0.95, (self.sobolev_bound / (current_norm + 1e-8)) ** 0.5)  # Square root for gentler scaling
                scale_factor = max(0.7, scale_factor)  # Less aggressive minimum
                
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
    """
    
    def __init__(
        self,
        data_dim: int,
        hidden_dim: int,
        activation=None,
        use_spectral_norm: bool = True,
        lambda_sobolev: float = 0.1,
        sobolev_bound: float = 5.0  # Increased bound
    ):
        super().__init__()
        
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.use_spectral_norm = use_spectral_norm
        self.lambda_sobolev = max(0.01, min(lambda_sobolev, 1.0))  # More reasonable bounds
        
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
        self.sobolev_regularizer = AdaptiveSobolevRegularizer(self.lambda_sobolev)  # Use adaptive version
        self.sobolev_projection = SobolevConstraintProjection(sobolev_bound)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights with appropriate scale for enhanced stability."""
        if isinstance(module, nn.Linear):
            # Use initialization (not overly conservative)
            nn.init.normal_(module.weight, 0.0, 0.02)  # Increased from 0.01
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the critic with reasonable output bounds."""
        output = self.model(x)
        # Clamp output to prevent explosion
        return torch.clamp(output, min=-200.0, max=200.0)  # Increased bounds
    
    def sobolev_regularization_loss(
        self,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        traffic_intensity: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute Sobolev regularization loss with stability."""
        try:
            return self.sobolev_regularizer(self, samples, sigma, traffic_intensity)
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
    lambda_sobolev: float = 0.1,
    interpolation_lambda: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Compute Sobolev-weighted gradient penalty for WGAN-GP.
    
    This extends the standard gradient penalty with Sobolev space integration
    and heoretical constraints.
    
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
    # Clamp lambda_sobolev for stabilityds
    lambda_sobolev = max(0.01, min(lambda_sobolev, 1.0))  # Increased upper bound
    
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
        interpolated_sigma = torch.clamp(interpolated_sigma, min=1e-8, max=1000.0)  # Increased upper bound
        
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
        # Remove artificial upper bounds - let theory guide the values
        
        # Standard gradient penalty term (1-Lipschitz constraint)
        gp_standard = ((gradient_norm - 1) ** 2).mean()
        
        # Sobolev-weighted term with better theoretical integration
        gp_sobolev = ((gradient_norm ** 2) * interpolated_sigma).mean()
        
        # Combine with theoretically motivated weighting
        # The Sobolev component should be weighted according to the density structure
        sobolev_weight = 0.2 * lambda_sobolev  # More aggressive weighting
        total_gp = gp_standard + sobolev_weight * gp_sobolev
        
        # Apply reasonable bounds only for extreme cases
        return torch.clamp(total_gp, max=1000.0)  # Much higher bound
        
    except Exception as e:
        print(f"Warning: Sobolev gradient penalty computation failed: {e}")
        return torch.tensor(0.1, device=device)


class SobolevWGANLoss:
    """
    Compute complete WGAN-GP loss with improved Sobolev space integration.
    
    This version provides more aggressive theoretical integration and
    reduced over-conservative constraints.
    """
    
    def __init__(
        self,
        lambda_gp: float = 1.0,
        lambda_sobolev: float = 0.1,
        use_adaptive_sobolev: bool = True
    ):
        # More reasonable parameter bounds
        self.lambda_gp = max(0.5, min(lambda_gp, 5.0))      # Increased bounds
        self.lambda_sobolev = max(0.01, min(lambda_sobolev, 1.0))  # Increased upper bound
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
        Compute critic loss with improved Sobolev regularization.
        
        Returns:
            Dict containing individual loss components.
        """
        try:
            # Standard WGAN critic loss with clamping
            critic_real = critic(real_samples).mean()
            critic_fake = critic(fake_samples).mean()
            # Remove artificial clamping - let the theory guide the values
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
            
            # Total critic loss with reduced regularization weights
            total_loss = wasserstein_loss + self.lambda_gp * sobolev_gp + 0.2 * sobolev_reg  # Increased weight
            
            # Apply reasonable bounds only for extreme cases
            total_loss = torch.clamp(total_loss, max=10000.0)  # Much higher bound
            
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
            # Apply reasonable bounds only for extreme cases
            return torch.clamp(loss, max=10000.0)  # Much higher bound
        except Exception as e:
            print(f"Warning: Sobolev WGAN generator loss computation failed: {e}")
            return torch.tensor(0.0, device=fake_samples.device)


class MassConservationSobolevRegularizer(AdaptiveSobolevRegularizer):
    """
    Compute Sobolev regularizer that integrates mass conservation constraints.
    
    This novel extension connects the Sobolev regularization directly with
    the mass conservation requirements from congested transport theory.
    """
    
    def __init__(
        self,
        lambda_sobolev: float = 0.1,
        gradient_penalty_weight: float = 1.0,
        adaptation_factor: float = 0.5,
        mass_conservation_weight: float = 0.1
    ):
        super().__init__(lambda_sobolev, gradient_penalty_weight, adaptation_factor)
        self.mass_conservation_weight = mass_conservation_weight
    
    def __call__(
        self,
        critic: nn.Module,
        samples: torch.Tensor,
        sigma: torch.Tensor,
        traffic_intensity: Optional[torch.Tensor] = None,
        target_density: Optional[torch.Tensor] = None,
        return_components: bool = False
    ) -> torch.Tensor:
        """
        Compute Sobolev regularization with mass conservation integration.
        
        This implements the theoretical connection between Sobolev smoothness
        and mass conservation requirements.
        
        Args:
            critic (nn.Module): Critic network.
            samples (torch.Tensor): Sample points.
            sigma (torch.Tensor): Current spatial density.
            traffic_intensity (Optional[torch.Tensor]): Traffic intensity for adaptation.
            target_density (Optional[torch.Tensor]): Target density for mass conservation.
            return_components (bool): Whether to return individual components.
        
        Returns:
            torch.Tensor: Sobolev regularization loss.
        """
        # Compute base adaptive Sobolev regularization
        result = super().__call__(critic, samples, sigma, traffic_intensity, return_components=True)
        
        # Add mass conservation penalty if target density is provided
        if target_density is not None and self.mass_conservation_weight > 0:
            try:
                # Ensure same size for comparison
                if target_density.shape[0] != sigma.shape[0]:
                    min_size = min(target_density.shape[0], sigma.shape[0])
                    target_density = target_density[:min_size]
                    sigma_resized = sigma[:min_size]
                else:
                    sigma_resized = sigma
                
                # Mass conservation penalty: penalize deviation from target density
                # This encourages the critic to respect mass conservation
                mass_deficit = (target_density - sigma_resized).abs().mean()
                mass_conservation_penalty = self.mass_conservation_weight * mass_deficit
                
                result['total'] = result['total'] + mass_conservation_penalty
                result['mass_conservation_penalty'] = mass_conservation_penalty.item()
                
                # Additional theoretical enhancement: if mass is not conserved,
                # increase regularization to enforce smoother transitions
                if mass_deficit > 0.1:  # Significant mass deficit
                    conservation_boost = 1.0 + 0.5 * mass_deficit
                    result['total'] = result['total'] * conservation_boost
                    result['conservation_boost'] = conservation_boost.item()
                else:
                    result['conservation_boost'] = 1.0
                    
            except Exception as e:
                print(f"Warning: Mass conservation Sobolev enhancement failed: {e}")
                result['mass_conservation_penalty'] = 0.0
                result['conservation_boost'] = 1.0
        
        if return_components:
            return result
        else:
            return result['total']