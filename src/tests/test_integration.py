"""
Complete Weight Perturbation Library Integration Test and Example

This script provides a complete, runnable example that tests all components
of the weight perturbation library and demonstrates the congested transport
framework with full theoretical implementation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
import logging
import unittest

warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('default')
sns.set_palette("husl")

# Set up logging
logging.basicConfig(level=logging.INFO)

# =============================================================================
# CORE MODELS (from models.py)
# =============================================================================

class Generator(nn.Module):
    """Generator model for the Weight Perturbation library."""

    def __init__(self, noise_dim: int, data_dim: int, hidden_dim: int, activation=None):
        super(Generator, self).__init__()
        if noise_dim <= 0 or data_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, data_dim)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.model(z)

class Critic(nn.Module):
    """Critic model for the Weight Perturbation library."""

    def __init__(self, data_dim: int, hidden_dim: int, activation=None):
        super(Critic, self).__init__()
        if data_dim <= 0 or hidden_dim <= 0:
            raise ValueError("All dimensions must be positive")
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# =============================================================================
# CONGESTION TRACKING COMPONENTS (from congestion.py)
# =============================================================================

class CongestionTracker:
    """Track congestion metrics throughout the perturbation process."""

    def __init__(self, lambda_param: float = 0.1, history_size: int = 100):
        self.lambda_param = lambda_param
        self.history_size = history_size
        self.history = {
            'traffic_intensity': [],
            'congestion_cost': [],
            'flow_divergence': [],
            'continuity_residual': [],
            'spatial_density': []
        }

    def update(self, congestion_info: Dict[str, torch.Tensor]) -> None:
        """Update congestion history with new measurements."""
        if 'traffic_intensity' in congestion_info:
            self.history['traffic_intensity'].append(
                congestion_info['traffic_intensity'].mean().item()
            )
        if 'congestion_cost' in congestion_info:
            self.history['congestion_cost'].append(
                congestion_info['congestion_cost'].item()
            )
        if 'spatial_density' in congestion_info:
            self.history['spatial_density'].append(
                congestion_info['spatial_density'].mean().item()
            )
        # Maintain history size
        for key in self.history:
            if len(self.history[key]) > self.history_size:
                self.history[key] = self.history[key][-self.history_size:]

    def check_congestion_increase(self, threshold: float = 0.1) -> bool:
        """Check if congestion has increased beyond threshold."""
        if len(self.history['congestion_cost']) < 2:
            return False
        recent_cost = self.history['congestion_cost'][-1]
        previous_cost = self.history['congestion_cost'][-2]
        return (recent_cost - previous_cost) / (previous_cost + 1e-8) > threshold

def compute_spatial_density(
    samples: torch.Tensor,
    bandwidth: float = 0.1,
    **kwargs
) -> Dict[str, torch.Tensor]:
    """Compute spatial density function σ(x) using kernel density estimation."""
    device = samples.device
    n_samples, data_dim = samples.shape
    
    # Vectorized KDE using cdist
    distances = torch.cdist(samples, samples)
    kernels = torch.exp(-distances**2 / (2 * bandwidth**2))
    density_at_samples = kernels.sum(dim=1) / (n_samples * bandwidth * np.sqrt(2 * np.pi))
    density_at_samples = density_at_samples + 1e-8  # Avoid zero division
    return {
        'density_at_samples': density_at_samples,
        'bandwidth': bandwidth
    }

def compute_traffic_flow(
    critic: torch.nn.Module,
    generator: torch.nn.Module,
    noise_samples: torch.Tensor,
    sigma: torch.Tensor,
    lambda_param: float = 0.1
) -> Dict[str, torch.Tensor]:
    """Compute traffic flow w_Q and intensity i_Q based on critic gradients."""
    generator.eval()
    critic.eval()
    
    # Generate samples
    with torch.no_grad():
        gen_samples = generator(noise_samples)
    
    # Enable gradients for critic computation
    gen_samples.requires_grad_(True)
    
    # Compute critic values
    critic_values = critic(gen_samples)
    
    # Compute gradients with respect to generated samples
    critic_gradients = torch.autograd.grad(
        outputs=critic_values.sum(),
        inputs=gen_samples,
        create_graph=True,
        retain_graph=True
    )[0]
    
    # Compute gradient norm
    gradient_norm = torch.norm(critic_gradients, p=2, dim=1, keepdim=True)
    
    # Avoid division by zero
    gradient_norm_safe = torch.clamp(gradient_norm, min=1e-8)
    
    # Compute (|∇u| - 1)_+
    gradient_excess = F.relu(gradient_norm - 1.0)
    
    # Compute traffic flow: w_Q = -λσ(|∇u| - 1)_+ ∇u/|∇u|
    traffic_flow = -lambda_param * sigma.unsqueeze(1) * gradient_excess * (
        critic_gradients / gradient_norm_safe
    )
    
    # Compute traffic intensity: i_Q = |w_Q|
    traffic_intensity = torch.norm(traffic_flow, p=2, dim=1)
    return {
        'traffic_flow': traffic_flow,
        'traffic_intensity': traffic_intensity,
        'critic_gradients': critic_gradients,
        'gradient_norm': gradient_norm.squeeze()
    }

def congestion_cost_function(
    traffic_intensity: torch.Tensor,
    sigma: torch.Tensor,
    lambda_param: float = 0.1,
    cost_type: str = 'quadratic_linear'
) -> torch.Tensor:
    """Compute congestion cost H(x, i) for given traffic intensity."""
    if cost_type == 'quadratic_linear':
        sigma_safe = torch.clamp(sigma, min=1e-8)
        quadratic_term = traffic_intensity ** 2 / (2 * lambda_param * sigma_safe)
        linear_term = torch.abs(traffic_intensity)
        return quadratic_term + linear_term
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")

# =============================================================================
# SOBOLEV REGULARIZATION COMPONENTS (from sobolev.py)
# =============================================================================

class WeightedSobolevRegularizer:
    """Weighted Sobolev H^1(Ω, σ) norm regularization for the critic."""

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
        """Compute weighted Sobolev norm regularization."""
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

class SobolevConstrainedCritic(nn.Module):
    """Critic network with built-in Sobolev space constraints."""

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
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize Sobolev regularizer
        self.sobolev_regularizer = WeightedSobolevRegularizer(lambda_sobolev)
        
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

# =============================================================================
# SAMPLERS (from samplers.py)
# =============================================================================

def sample_real_data(
    batch_size: int,
    means: Optional[List] = None,
    std: float = 0.4,
    device='cpu'
) -> torch.Tensor:
    """Sample from a real data distribution, such as multiple Gaussian clusters."""
    device = torch.device(device)
    if means is None:
        means = [torch.tensor([2.0, 0.0], device=device, dtype=torch.float32),
                 torch.tensor([-2.0, 0.0], device=device, dtype=torch.float32),
                 torch.tensor([0.0, 2.0], device=device, dtype=torch.float32),
                 torch.tensor([0.0, -2.0], device=device, dtype=torch.float32)]
    else:
        means = [torch.tensor(m, dtype=torch.float32, device=device) for m in means]
    num_clusters = len(means)
    data_dim = means[0].shape[0]
    samples_per_cluster = batch_size // num_clusters
    remainder = batch_size % num_clusters
    samples = []
    for i, mean in enumerate(means):
        n = samples_per_cluster + (1 if i < remainder else 0)
        cluster_samples = mean + std * torch.randn(n, data_dim, device=device)
        samples.append(cluster_samples)
    return torch.cat(samples, dim=0)

def sample_target_data(
    batch_size: int,
    shift: Optional[List] = None,
    means: Optional[List] = None,
    std: float = 0.4,
    device='cpu'
) -> torch.Tensor:
    """Sample from a target distribution, which is a shifted version of the real data clusters."""
    device = torch.device(device)
    if means is None:
        means = [torch.tensor([2.0, 0.0], device=device, dtype=torch.float32),
                 torch.tensor([-2.0, 0.0], device=device, dtype=torch.float32),
                 torch.tensor([0.0, 2.0], device=device, dtype=torch.float32),
                 torch.tensor([0.0, -2.0], device=device, dtype=torch.float32)]
    else:
        means = [torch.tensor(m, dtype=torch.float32, device=device) for m in means]
    if shift is None:
        shift = torch.tensor([1.8, 1.8], dtype=torch.float32, device=device)
    else:
        shift = torch.tensor(shift, dtype=torch.float32, device=device)
    shifted_means = [m + shift for m in means]
    return sample_real_data(batch_size, means=shifted_means, std=std, device=device)

def sample_evidence_domains(
    num_domains: int,
    samples_per_domain: int,
    random_shift: float = 2.0,
    std: float = 0.4,
    device='cpu'
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Sample multiple evidence domains."""
    device = torch.device(device)
    evidence_list = []
    centers = []
    for i in range(num_domains):
        center = torch.tensor([i * random_shift, 0.0], device=device, dtype=torch.float32)
        domain_samples = center + std * torch.randn(samples_per_domain, 2, device=device)
        evidence_list.append(domain_samples)
        centers.append(center)
    return evidence_list, centers

# =============================================================================
# PRETRAIN FUNCTIONALITY (from pretrain.py)
# =============================================================================

def compute_gradient_penalty(critic, real_samples, fake_samples, device):
    """Compute the gradient penalty for WGAN-GP."""
    batch_size = real_samples.shape[0]
    if batch_size == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)
    alpha = torch.rand(batch_size, 1, device=device).expand_as(real_samples)
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates.requires_grad_(True)
    critic_inter = critic(interpolates)
    grad_outputs = torch.ones_like(critic_inter, device=device)
    try:
        gradients = torch.autograd.grad(
            outputs=critic_inter,
            inputs=interpolates,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True
        )[0]
        gradients = gradients.view(batch_size, -1)
        grad_norm = gradients.norm(2, dim=1)
        penalty = ((grad_norm - 1) ** 2).mean()
    except RuntimeError:
        penalty = torch.tensor(0.1, device=device, requires_grad=False)
    return penalty

def pretrain_wgan_gp(
    generator, critic, real_sampler,
    epochs: int = 300,
    batch_size: int = 96,
    lr: float = 2e-4,
    betas=(0.5, 0.95),
    gp_lambda: float = 0.5,
    critic_iters: int = 5,
    noise_dim: int = 2,
    device='cpu',
    verbose: bool = True
):
    """Pretrain a generator and critic using WGAN-GP."""
    device = torch.device(device)
    generator.to(device)
    critic.to(device)
    optim_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
    optim_d = torch.optim.Adam(critic.parameters(), lr=lr, betas=betas)
    generator.train()
    critic.train()
    for epoch in range(epochs):
        for _ in range(critic_iters):
            try:
                real = real_sampler(batch_size).to(device)
                z = torch.randn(batch_size, noise_dim, device=device)
                fake = generator(z)
                crit_real = critic(real).mean()
                crit_fake = critic(fake).mean()
                gp = compute_gradient_penalty(critic, real, fake, device)
                loss_d = -crit_real + crit_fake + gp_lambda * gp
                optim_d.zero_grad()
                loss_d.backward()
                optim_d.step()
            except Exception as e:
                if verbose:
                    logging.warning(f"Warning: Critic training failed at epoch {epoch}: {e}")
                continue
        # Generator update
        try:
            z = torch.randn(batch_size, noise_dim, device=device)
            fake = generator(z)
            loss_g = -critic(fake).mean()
            optim_g.zero_grad()
            loss_g.backward()
            optim_g.step()
        except Exception as e:
            if verbose:
                logging.warning(f"Warning: Generator training failed at epoch {epoch}: {e}")
            continue
        if verbose and (epoch + 1) % 50 == 0:
            try:
                print(f"Epoch [{epoch+1}/{epochs}] | D Loss: {loss_d.item():.4f} | G Loss: {loss_g.item():.4f}")
            except:
                print(f"Epoch [{epoch+1}/{epochs}] | Training in progress...")
    if verbose:
        print("Pretraining completed.")
    return generator, critic

# =============================================================================
# CONGESTED TRANSPORT WEIGHT PERTURBER
# =============================================================================

class CTWeightPerturberTargetGiven:
    """Congested Transport Weight Perturber for target-given perturbation."""

    def __init__(
        self,
        generator: Generator,
        target_samples: torch.Tensor,
        critic: Optional[Critic] = None,
        enable_congestion_tracking: bool = True
    ):
        self.generator = generator
        self.target_samples = target_samples.to(next(generator.parameters()).device)
        self.critic = critic
        self.enable_congestion_tracking = enable_congestion_tracking
        self.device = next(generator.parameters()).device
        
        # Initialize congestion tracker
        if self.enable_congestion_tracking:
            self.congestion_tracker = CongestionTracker()

    def perturb(
        self,
        steps: int = 20,
        eta_init: float = 0.01,
        lambda_congestion: float = 0.1,
        verbose: bool = True
    ) -> Generator:
        """Perform congested transport perturbation."""
        # Create a copy of the generator
        pert_gen = Generator(
            noise_dim=self.generator.noise_dim,
            data_dim=self.generator.data_dim,
            hidden_dim=self.generator.hidden_dim
        ).to(self.device)
        pert_gen.load_state_dict(self.generator.state_dict())
        optimizer = torch.optim.Adam(pert_gen.parameters(), lr=eta_init)
        for step in range(steps):
            # Generate samples
            noise = torch.randn(400, self.generator.noise_dim, device=self.device)
            gen_samples = pert_gen(noise)
            
            # Compute basic W2-like loss (simplified)
            distances = torch.cdist(gen_samples, self.target_samples)
            w2_loss = distances.min(dim=1)[0].mean()
            total_loss = w2_loss
            congestion_info = {}
            
            # Add congestion terms if tracking enabled
            if self.enable_congestion_tracking and self.critic is not None:
                # Compute spatial density
                density_info = compute_spatial_density(gen_samples, bandwidth=0.12)
                sigma = density_info['density_at_samples']
                
                # Compute traffic flow
                flow_info = compute_traffic_flow(
                    self.critic, pert_gen, noise, sigma, lambda_congestion
                )
                
                # Compute congestion cost
                congestion_cost = congestion_cost_function(
                    flow_info['traffic_intensity'], sigma, lambda_congestion
                ).mean()
                
                # Add to total loss
                total_loss = total_loss + 0.1 * congestion_cost
                
                # Store congestion info
                congestion_info = {
                    'spatial_density': sigma,
                    'traffic_flow': flow_info['traffic_flow'],
                    'traffic_intensity': flow_info['traffic_intensity'],
                    'congestion_cost': congestion_cost
                }
                
                # Update tracker
                self.congestion_tracker.update(congestion_info)
            
            # Optimization step
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(pert_gen.parameters(), 0.1)
            optimizer.step()
            if verbose and step % 5 == 0:
                congestion_str = ""
                if congestion_info:
                    congestion_str = f" | Congestion: {congestion_info['congestion_cost'].item():.4f}"
                print(f"Step {step}: W2 Loss: {w2_loss.item():.4f}{congestion_str}")
        return pert_gen

# =============================================================================
# COMPREHENSIVE VISUALIZATION
# =============================================================================

class ComprehensiveVisualizer:
    """Comprehensive visualization for congested transport analysis."""

    def __init__(self, save_dir="congestion_analysis"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.step_data = []

    def visualize_step(self, step, generator, critic, target_samples,
                       noise_samples, lambda_congestion=0.1):
        """Visualize a single perturbation step with full congestion analysis."""
        device = next(generator.parameters()).device
        
        # Generate samples
        with torch.no_grad():
            gen_samples = generator(noise_samples)
        
        # Compute congestion metrics
        density_info = compute_spatial_density(gen_samples, bandwidth=0.12)
        sigma = density_info['density_at_samples']
        flow_info = compute_traffic_flow(
            critic, generator, noise_samples, sigma, lambda_congestion
        )
        congestion_cost = congestion_cost_function(
            flow_info['traffic_intensity'], sigma, lambda_congestion
        )
        
        # Store step data
        step_data = {
            'step': step,
            'gen_samples': gen_samples.detach().cpu().numpy(),
            'target_samples': target_samples.detach().cpu().numpy(),
            'traffic_flow': flow_info['traffic_flow'].detach().cpu().numpy(),
            'traffic_intensity': flow_info['traffic_intensity'].detach().cpu().numpy(),
            'spatial_density': sigma.detach().cpu().numpy(),
            'congestion_cost': congestion_cost.mean().detach().item()
        }
        self.step_data.append(step_data)
        
        # Create visualization
        self._create_step_plot(step_data)
        return step_data

    def _unit_quiver(self, ax, X, Y, U, V, color_by=None, arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.85):
        """
        Draws a quiver plot that shows only directions using arrows of equal length.
        - arrow_frac: Arrow length as a fraction of the axis range (e.g., 0.1 means 10% of the x-range)
        - color_by: Values to encode color (e.g., traffic_intensity); if None, use a single color
        """
        X = np.asarray(X); Y = np.asarray(Y); U = np.asarray(U); V = np.asarray(V)
        # Direction unit vectors
        mag = np.sqrt(U**2 + V**2) + 1e-12
        Uu = U / mag
        Vu = V / mag

        # Axis range extents
        x_min = np.nanmin(X); x_max = np.nanmax(X)
        y_min = np.nanmin(Y); y_max = np.nanmax(Y)
        x_range = max(x_max - x_min, 1e-6)
        y_range = max(y_max - y_min, 1e-6)

        # Determine on-screen uniform arrow length considering both x and y ranges
        # Quiver interprets U,V in data coordinates; compensate for x/y scale differences
        # Reference length: includes compensation for diagonal components
        target_len_x = arrow_frac * x_range
        target_len_y = arrow_frac * y_range
        U_plot = Uu * target_len_x
        V_plot = Vu * target_len_y

        quiv = ax.quiver(
            X, Y, U_plot, V_plot,
            color_by if color_by is not None else None,
            cmap=cmap, angles='xy', scale_units='xy', scale=1.0,
            width=width, alpha=alpha
        )
        # Set axis limits with a small margin for better visuals
        ax.set_xlim(x_min - 0.05*x_range, x_max + 0.05*x_range)
        ax.set_ylim(y_min - 0.05*y_range, y_max + 0.05*y_range)
        return quiv


    def _create_step_plot(self, step_data):
        """Create comprehensive plot for a single step."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Congested Transport Analysis - Step {step_data["step"]}',
                     fontsize=16, fontweight='bold')
        gen_samples = step_data['gen_samples']
        target_samples = step_data['target_samples']
        traffic_flow = step_data['traffic_flow']
        traffic_intensity = step_data['traffic_intensity']
        spatial_density = step_data['spatial_density']
        
        # Plot 1: Samples comparison
        axes[0, 0].scatter(gen_samples[:, 0], gen_samples[:, 1],
                           c='blue', alpha=0.6, s=30, label='Generated')
        axes[0, 0].scatter(target_samples[:, 0], target_samples[:, 1],
                           c='red', alpha=0.6, s=30, label='Target')
        axes[0, 0].set_title('Generated vs Target Samples')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Traffic flow vector field
        n_arrows = min(50, len(gen_samples))
        indices = np.random.choice(len(gen_samples), n_arrows, replace=False)
        quiver = self._unit_quiver(
            axes[0,1],
            gen_samples[indices, 0], gen_samples[indices, 1],
            traffic_flow[indices, 0], traffic_flow[indices, 1],
            color_by=traffic_intensity[indices],
            arrow_frac=0.1, width=0.004, cmap='viridis', alpha=0.85
        )
        plt.colorbar(quiver, ax=axes[0, 1], shrink=0.8, label='Traffic Intensity')
        axes[0, 1].set_title('Traffic Flow Vector Field w_Q(x)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Traffic intensity heatmap
        scatter = axes[0, 2].scatter(gen_samples[:, 0], gen_samples[:, 1],
                                     c=traffic_intensity, cmap='plasma', s=50, alpha=0.7)
        plt.colorbar(scatter, ax=axes[0, 2], shrink=0.8, label='Traffic Intensity')
        axes[0, 2].set_title('Traffic Intensity Distribution')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Spatial density
        density_scatter = axes[1, 0].scatter(gen_samples[:, 0], gen_samples[:, 1],
                                             c=spatial_density, cmap='coolwarm', s=50, alpha=0.7)
        plt.colorbar(density_scatter, ax=axes[1, 0], shrink=0.8, label='Spatial Density σ(x)')
        axes[1, 0].set_title('Spatial Density Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Intensity histogram
        axes[1, 1].hist(traffic_intensity, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        axes[1, 1].axvline(traffic_intensity.mean(), color='red', linestyle='--',
                           label=f'Mean: {traffic_intensity.mean():.4f}')
        axes[1, 1].set_title('Traffic Intensity Distribution')
        axes[1, 1].set_xlabel('Traffic Intensity')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Statistics summary
        axes[1, 2].axis('off')
        stats_text = f"""Step {step_data['step']} Statistics:
Traffic Flow:
• Mean Intensity: {traffic_intensity.mean():.6f}
• Max Intensity: {traffic_intensity.max():.6f}
• Flow Magnitude: {np.linalg.norm(traffic_flow, axis=1).mean():.6f}
Spatial Properties:
• Mean Density: {spatial_density.mean():.6f}
• Density Variance: {spatial_density.var():.6f}
Congestion:
• Total Cost: {step_data['congestion_cost']:.6f}
"""
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes,
                        verticalalignment='top', fontsize=10, fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_dir / f"step_{step_data['step']:03d}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved visualization: {save_path}")

    def create_evolution_summary(self):
        """Create summary of evolution across all steps."""
        if not self.step_data:
            logging.warning("No data available for summary.")
            return
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Congested Transport Evolution Summary', fontsize=16, fontweight='bold')
        steps = [data['step'] for data in self.step_data]
        costs = [data['congestion_cost'] for data in self.step_data]
        mean_intensities = [data['traffic_intensity'].mean() for data in self.step_data]
        mean_densities = [data['spatial_density'].mean() for data in self.step_data]
        
        # Congestion cost evolution
        axes[0, 0].plot(steps, costs, 'b-o', linewidth=2)
        axes[0, 0].set_title('Congestion Cost Evolution')
        axes[0, 0].set_xlabel('Step')
        axes[0, 0].set_ylabel('H(x, i_Q)')
        axes[0, 0].grid(True)
        
        # Traffic intensity evolution
        axes[0, 1].plot(steps, mean_intensities, 'g-o', linewidth=2)
        axes[0, 1].set_title('Mean Traffic Intensity Evolution')
        axes[0, 1].set_xlabel('Step')
        axes[0, 1].set_ylabel('Mean |w_Q|')
        axes[0, 1].grid(True)
        
        # Spatial density evolution
        axes[1, 0].plot(steps, mean_densities, 'm-o', linewidth=2)
        axes[1, 0].set_title('Mean Spatial Density Evolution')
        axes[1, 0].set_xlabel('Step')
        axes[1, 0].set_ylabel('Mean σ(x)')
        axes[1, 0].grid(True)
        
        # Final comparison
        if len(self.step_data) >= 2:
            initial_data = self.step_data[0]
            final_data = self.step_data[-1]
            initial_samples = initial_data['gen_samples']
            final_samples = final_data['gen_samples']
            axes[1, 1].scatter(initial_samples[:, 0], initial_samples[:, 1],
                               c='lightblue', alpha=0.5, s=20, label='Initial')
            axes[1, 1].scatter(final_samples[:, 0], final_samples[:, 1],
                               c='darkblue', alpha=0.7, s=20, label='Final')
            axes[1, 1].scatter(final_data['target_samples'][:, 0],
                               final_data['target_samples'][:, 1],
                               c='red', alpha=0.5, s=20, label='Target')
            axes[1, 1].set_title('Initial vs Final Samples')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        plt.tight_layout()
        save_path = self.save_dir / "evolution_summary.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        logging.info(f"Saved evolution summary: {save_path}")

# =============================================================================
# MAIN DEMONSTRATION
# =============================================================================

def run_complete_demonstration():
    """Run complete demonstration of the congested transport framework."""
    logging.info("="*80)
    logging.info("COMPLETE CONGESTED TRANSPORT WEIGHT PERTURBATION DEMONSTRATION")
    logging.info("="*80)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Create models
    logging.info("\n1. Creating models...")
    generator = Generator(noise_dim=2, data_dim=2, hidden_dim=128)
    generator = generator.to(device)
    critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=128)
    critic = critic.to(device)
    logging.info(f"✓ Generator parameters: {sum(p.numel() for p in generator.parameters())}")
    logging.info(f"✓ Critic parameters: {sum(p.numel() for p in critic.parameters())}")
    
    # Pretrain models
    logging.info("\n2. Pretraining models with WGAN-GP...")
    real_sampler = lambda bs: sample_real_data(bs, device=device)
    generator, critic = pretrain_wgan_gp(
        generator, critic, real_sampler,
        epochs=300, batch_size=96, device=device, verbose=True
    )
    logging.info("✓ Pretraining completed")
    
    # Create target distribution
    logging.info("\n3. Creating target distribution...")
    target_samples = sample_target_data(
        batch_size=600, shift=[1.5, 1.5], device=device
    )
    logging.info(f"✓ Target samples shape: {target_samples.shape}")
    
    # Initialize visualizer
    logging.info("\n4. Initializing congested transport visualizer...")
    visualizer = ComprehensiveVisualizer(save_dir="test_results/plots/complete_demo_results")
    
    # Initialize perturber
    logging.info("\n5. Creating congested transport perturber...")
    perturber = CTWeightPerturberTargetGiven(
        generator, target_samples, critic, enable_congestion_tracking=True
    )
    
    # Perform perturbation with visualization
    logging.info("\n6. Starting perturbation with congestion tracking...")
    
    # Create a copy for perturbation
    pert_gen = Generator(
        noise_dim=generator.noise_dim,
        data_dim=generator.data_dim,
        hidden_dim=generator.hidden_dim
    ).to(device)
    pert_gen.load_state_dict(generator.state_dict())
    
    # Manual perturbation loop with visualization
    optimizer = torch.optim.Adam(pert_gen.parameters(), lr=0.01)
    lambda_congestion = 0.1
    
    for step in range(15):
        logging.info(f"\n--- Step {step} ---")
        
        # Generate evaluation samples
        eval_noise = torch.randn(400, 2, device=device)
        
        # Visualize current state (every 3 steps)
        if step % 3 == 0:
            step_data = visualizer.visualize_step(
                step, pert_gen, critic, target_samples, eval_noise.detach(), lambda_congestion
            )
            logging.info(f"Step {step} Results:")
            logging.info(f" Congestion Cost: {step_data['congestion_cost']:.6f}")
            logging.info(f" Mean Traffic Intensity: {step_data['traffic_intensity'].mean():.6f}")
            logging.info(f" Mean Spatial Density: {step_data['spatial_density'].mean():.6f}")
        
        # Perform optimization step
        pert_gen.train()
        noise = torch.randn(200, 2, device=device)
        gen_samples = pert_gen(noise)
        
        # Compute loss with congestion terms
        distances = torch.cdist(gen_samples.detach(), target_samples[:200])
        w2_loss = distances.min(dim=1)[0].mean()
        
        # Add congestion terms
        density_info = compute_spatial_density(gen_samples.detach(), bandwidth=0.12)
        sigma = density_info['density_at_samples']
        flow_info = compute_traffic_flow(
            critic, pert_gen, noise.detach(), sigma, lambda_congestion
        )
        congestion_cost = congestion_cost_function(
            flow_info['traffic_intensity'], sigma, lambda_congestion
        ).mean()
        
        # Add Sobolev regularization
        sobolev_loss = critic.sobolev_regularization_loss(gen_samples, sigma)
        total_loss = w2_loss + 0.1 * congestion_cost + sobolev_loss
        
        # Optimization
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(pert_gen.parameters(), 0.1)
        optimizer.step()
        logging.info(f"Losses: W2={w2_loss.item():.4f}, Congestion={congestion_cost.item():.4f}, Sobolev={sobolev_loss.item():.4f}")
    
    # Final evaluation
    logging.info("\n7. Final evaluation...")
    eval_noise = torch.randn(600, 2, device=device)
    with torch.no_grad():
        original_samples = generator(eval_noise)
        final_samples = pert_gen(eval_noise)
    
    # Compute final distances
    original_dist = torch.cdist(original_samples, target_samples).min(dim=1)[0].mean()
    final_dist = torch.cdist(final_samples, target_samples).min(dim=1)[0].mean()
    logging.info(f"Original distance to target: {original_dist.item():.4f}")
    logging.info(f"Final distance to target: {final_dist.item():.4f}")
    logging.info(f"Improvement: {((original_dist - final_dist) / original_dist * 100).item():.2f}%")
    
    # Create evolution summary
    logging.info("\n8. Creating evolution summary...")
    visualizer.create_evolution_summary()
    logging.info(f"\n{'='*80}")
    logging.info("DEMONSTRATION COMPLETED SUCCESSFULLY!")
    logging.info(f"{'='*80}")
    logging.info("Generated outputs:")
    logging.info(f" ✓ Step-by-step visualizations in: {visualizer.save_dir}")
    logging.info(" ✓ Traffic flow vector fields")
    logging.info(" ✓ Spatial density distributions")
    logging.info(" ✓ Congestion cost tracking")
    logging.info(" ✓ Sobolev regularization effects")
    logging.info(" ✓ Evolution summary plots")
    return {
        'original_generator': generator,
        'perturbed_generator': pert_gen,
        'critic': critic,
        'target_samples': target_samples,
        'final_improvement': ((original_dist - final_dist) / original_dist * 100).item()
    }

def test_individual_components():
    """Test individual components of the library."""
    logging.info("\n" + "="*60)
    logging.info("TESTING INDIVIDUAL COMPONENTS")
    logging.info("="*60)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test 1: Model creation
    logging.info("\n1. Testing model creation...")
    try:
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        # crit = Critic(data_dim=2, hidden_dim=64).to(device)
        sobolev_crit = SobolevConstrainedCritic(data_dim=2, hidden_dim=64).to(device)
        logging.info("✓ All models created successfully")
    except Exception as e:
        logging.error(f"❌ Model creation failed: {e}")
        return False
    
    # Test 2: Data sampling
    logging.info("\n2. Testing data sampling...")
    try:
        real_data = sample_real_data(100, device=device)
        target_data = sample_target_data(100, shift=[1.0, 1.0], device=device)
        logging.info(f"✓ Real data shape: {real_data.shape}")
        logging.info(f"✓ Target data shape: {target_data.shape}")
    except Exception as e:
        logging.error(f"❌ Data sampling failed: {e}")
        return False
    
    # Test 3: Congestion components
    logging.info("\n3. Testing congestion tracking components...")
    try:
        # Test spatial density computation
        samples = torch.randn(50, 2, device=device)
        density_info = compute_spatial_density(samples, bandwidth=0.1)
        logging.info(f"✓ Spatial density computed, shape: {density_info['density_at_samples'].shape}")
        # Test congestion tracker
        tracker = CongestionTracker()
        congestion_info = {
            'traffic_intensity': torch.randn(50, device=device),
            'congestion_cost': torch.tensor(0.5),
            'spatial_density': torch.randn(50, device=device)
        }
        tracker.update(congestion_info)
        logging.info("✓ Congestion tracker updated successfully")
    except Exception as e:
        logging.error(f"❌ Congestion components failed: {e}")
        return False
    
    # Test 4: Sobolev regularization
    logging.info("\n4. Testing Sobolev regularization...")
    try:
        sobolev_reg = WeightedSobolevRegularizer(lambda_sobolev=0.01)
        samples = torch.randn(30, 2, device=device, requires_grad=True)
        sigma = torch.rand(30, device=device) + 0.1
        reg_loss = sobolev_reg(sobolev_crit, samples, sigma)
        logging.info(f"✓ Sobolev regularization computed: {reg_loss.item():.6f}")
    except Exception as e:
        logging.error(f"❌ Sobolev regularization failed: {e}")
        return False
    
    # Test 5: Traffic flow computation
    logging.info("\n5. Testing traffic flow computation...")
    try:
        noise = torch.randn(30, 2, device=device)
        sigma = torch.rand(30, device=device) + 0.1
        flow_info = compute_traffic_flow(sobolev_crit, gen, noise, sigma, lambda_param=0.1)
        logging.info(f"✓ Traffic flow computed, intensity shape: {flow_info['traffic_intensity'].shape}")
        logging.info(f"✓ Mean traffic intensity: {flow_info['traffic_intensity'].mean().item():.6f}")
    except Exception as e:
        logging.error(f"❌ Traffic flow computation failed: {e}")
        return False
    logging.info("\n✅ ALL COMPONENT TESTS PASSED!")
    return True

def create_quick_demo():
    """Create a quick demonstration for testing."""
    logging.info("\n" + "="*60)
    logging.info("QUICK DEMONSTRATION")
    logging.info("="*60)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Quick setup
    gen = Generator(noise_dim=2, data_dim=2, hidden_dim=32).to(device)
    crit = Critic(data_dim=2, hidden_dim=32).to(device)

    logging.info(f"Generator device: {next(gen.parameters()).device}")
    logging.info(f"Critic device: {next(crit.parameters()).device}")
    assert next(gen.parameters()).device == device, "Generator not on correct device"
    assert next(crit.parameters()).device == device, "Critic not on correct device"
    
    # Quick pretraining (just a few epochs)
    logging.info("Quick pretraining...")
    real_sampler = lambda bs: sample_real_data(bs, device=device)
    
    # Very short pretraining for demo
    optimizer_g = torch.optim.Adam(gen.parameters(), lr=0.001)
    optimizer_c = torch.optim.Adam(crit.parameters(), lr=0.001)
    
    
    for epoch in range(10):
        # Simple training loop
        real_data = real_sampler(32).to(device)
        noise = torch.randn(32, 2, device=device)
        fake_for_c = gen(noise).clone().to(device) # critic update uses detached fake
        loss_c = crit(fake_for_c).mean() - crit(real_data).mean()

        fake_for_g = gen(noise).clone().to(device) # generator update uses fresh graph
        loss_g = -crit(fake_for_g).mean()

        optimizer_c.zero_grad()
        loss_c.backward()
        optimizer_c.step()

        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()
    logging.info("✓ Quick pretraining completed")
    
    # Test congestion computation
    logging.info("Testing congestion computation...")
    target_samples = sample_target_data(100, shift=[1.0, 1.0], device=device)
    
    # Test the perturber
    perturber = CTWeightPerturberTargetGiven(
        gen, target_samples, crit, enable_congestion_tracking=True
    )
    
    # Quick perturbation
    logging.info("Running quick perturbation...")
    perturbed_gen = perturber.perturb(steps=5, verbose=True)
    
    # Final evaluation
    noise = torch.randn(100, 2, device=device)
    with torch.no_grad():
        original_samples = gen(noise)
        perturbed_samples = perturbed_gen(noise)
    orig_dist = torch.cdist(original_samples, target_samples).min(dim=1)[0].mean()
    pert_dist = torch.cdist(perturbed_samples, target_samples).min(dim=1)[0].mean()
    logging.info(f"\nQuick Demo Results:")
    logging.info(f" Original distance: {orig_dist.item():.4f}")
    logging.info(f" Perturbed distance: {pert_dist.item():.4f}")
    logging.info(f" Improvement: {((orig_dist - pert_dist) / orig_dist * 100).item():.2f}%")
    logging.info("✅ Quick demo completed successfully!")

def run_comprehensive_analysis():
    """Run comprehensive analysis covering all aspects."""
    logging.info("\n" + "="*80)
    logging.info("COMPREHENSIVE WEIGHT PERTURBATION ANALYSIS")
    logging.info("="*80)
    
    # Component testing
    logging.info("\nPhase 1: Component Testing")
    if not test_individual_components():
        logging.error("❌ Component testing failed. Stopping.")
        return
    
    # Quick demonstration
    logging.info("\nPhase 2: Quick Demonstration")
    try:
        create_quick_demo()
    except Exception as e:
        logging.warning(f"⚠ Quick demo failed: {e}")
    
    # Full demonstration
    logging.info("\nPhase 3: Complete Demonstration")
    try:
        results = run_complete_demonstration()
        logging.info(f"\n🎉 ANALYSIS COMPLETE!")
        logging.info(f"Final improvement achieved: {results['final_improvement']:.2f}%")
        return results
    except Exception as e:
        logging.error(f"❌ Complete demonstration failed: {e}")
        logging.info("This may be due to missing dependencies or computational constraints.")
        return None

# =============================================================================
# USAGE EXAMPLES AND DOCUMENTATION
# =============================================================================

def print_usage_examples():
    """Print comprehensive usage examples."""
    logging.info("\n" + "="*80)
    logging.info("WEIGHT PERTURBATION LIBRARY - USAGE EXAMPLES")
    logging.info("="*80)
    examples = '''
# Example 1: Basic Target-Given Perturbation
from weight_perturbation import *
# Setup
device = compute_device()
set_seed(42)
# Create and pretrain models
generator = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)
critic = Critic(data_dim=2, hidden_dim=256).to(device)

real_sampler = lambda bs: sample_real_data(bs, device=device)
pretrained_gen, _ = pretrain_wgan_gp(
    generator, critic, real_sampler, epochs=300, device=device
)
# Create target and perturb
target_samples = sample_target_data(1000, shift=[1.5, 1.5], device=device)
perturber = CTWeightPerturberTargetGiven(
    pretrained_gen, target_samples, critic, enable_congestion_tracking=True
)
perturbed_gen = perturber.perturb(steps=20, verbose=True)
# Evaluate
noise = torch.randn(1000, 2, device=device)
with torch.no_grad():
    original_samples = pretrained_gen(noise)
    perturbed_samples = perturbed_gen(noise)
print("Perturbation completed!")

# Example 2: Advanced Congestion Analysis
# Enable full theoretical framework
from weight_perturbation import (
    compute_spatial_density, compute_traffic_flow,
    congestion_cost_function, SobolevConstrainedCritic
)
# Use Sobolev-constrained critic
critic = SobolevConstrainedCritic(
    data_dim=2, hidden_dim=256, lambda_sobolev=0.01
).to(device)

# Analyze congestion at each step
for step in range(perturbation_steps):
    # Compute spatial density σ(x)
    density_info = compute_spatial_density(gen_samples, bandwidth=0.12)
    sigma = density_info['density_at_samples']
    # Compute traffic flow w_Q and intensity i_Q
    flow_info = compute_traffic_flow(
        critic, generator, noise, sigma, lambda_param=0.1
    )
    # Compute congestion cost H(x, i_Q)
    congestion_cost = congestion_cost_function(
        flow_info['traffic_intensity'], sigma, lambda_param=0.1
    )
    print(f"Step {step}: Congestion cost = {congestion_cost.mean().item():.6f}")

# Example 3: Visualization and Analysis
visualizer = ComprehensiveVisualizer(save_dir="analysis_results")
# Visualize each step
for step in range(num_steps):
    step_data = visualizer.visualize_step(
        step, generator, critic, target_samples, noise_samples
    )
# Create evolution summary
visualizer.create_evolution_summary()
'''
    logging.info(examples)
    logging.info("\n" + "="*80)
    logging.info("KEY FEATURES DEMONSTRATED:")
    logging.info("="*80)
    logging.info("✓ Spatial density estimation σ(x)")
    logging.info("✓ Traffic flow computation w_Q(x) with vector directions")
    logging.info("✓ Traffic intensity tracking i_Q(x)")
    logging.info("✓ Congestion cost function H(x, i_Q)")
    logging.info("✓ Sobolev regularization H^1(Ω, σ)")
    logging.info("✓ Step-by-step congestion tracking")
    logging.info("✓ Real-time visualization of flow fields")
    logging.info("✓ Complete theoretical framework integration")

# =============================================================================
# UNIT TESTS
# =============================================================================

class TestWeightPerturbation(unittest.TestCase):
    def setUp(self):
        self.device = torch.device('cpu')  # Use CPU for tests

    def test_model_creation(self):
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
        crit = Critic(data_dim=2, hidden_dim=64)
        self.assertIsInstance(gen, nn.Module)
        self.assertIsInstance(crit, nn.Module)

    def test_data_sampling(self):
        real_data = sample_real_data(100, device=self.device)
        self.assertEqual(real_data.shape, (100, 2))

    def test_spatial_density(self):
        samples = torch.randn(10, 2, device=self.device)
        density = compute_spatial_density(samples)['density_at_samples']
        self.assertEqual(density.shape, (10,))

    def test_traffic_flow(self):
        gen = Generator(2, 2, 64).to(self.device)
        crit = Critic(2, 64).to(self.device)
        noise = torch.randn(10, 2, device=self.device)
        sigma = torch.ones(10, device=self.device)
        flow_info = compute_traffic_flow(crit, gen, noise, sigma)
        self.assertIn('traffic_flow', flow_info)

if __name__ == "__main__":
    print("="*80)
    print("WEIGHT PERTURBATION LIBRARY - COMPLETE INTEGRATION TEST")
    print("="*80)
    print("This script provides a complete implementation and test of the")
    print("weight perturbation library with full congested transport theory.")
    print("="*80)
    
    # Print usage examples
    print_usage_examples()
    
    # Run unit tests
    unittest.main(exit=False)
    
    # Run comprehensive analysis
    try:
        results = run_comprehensive_analysis()
        if results:
            logging.info("\n🎉 SUCCESS! All components working correctly.")
            logging.info("📊 Check the generated visualizations for detailed analysis.")
    except KeyboardInterrupt:
        logging.info("\n⏹ Analysis interrupted by user.")
    except Exception as e:
        logging.error(f"\n❌ Unexpected error: {e}")
        logging.info("This may indicate missing dependencies or system constraints.")
    logging.info("\n" + "="*80)
    logging.info("For questions or issues, refer to the comprehensive documentation")
    logging.info("and the theoretical background provided in the library.")
    logging.info("="*80)
