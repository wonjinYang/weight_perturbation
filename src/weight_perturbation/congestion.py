"""
Congestion tracking and spatial density computation for congested transport.

This module implements the theoretical components from the congested transport framework:
- Spatial density estimation σ(x)
- Traffic flow computation w_Q
- Traffic intensity tracking i_Q  
- Congestion cost functions H(x, i)
- Continuity equation verification
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from abc import ABC, abstractmethod


class CongestionTracker:
    """
    Track congestion metrics throughout the perturbation process.
    
    This class implements the theoretical congestion tracking framework,
    monitoring traffic intensity, flow divergence, and continuity equation satisfaction.
    """
    
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
    
    def get_average_congestion(self, window: int = 10) -> float:
        """Get average congestion over recent window."""
        if not self.history['congestion_cost']:
            return 0.0
        
        window = min(window, len(self.history['congestion_cost']))
        return sum(self.history['congestion_cost'][-window:]) / window


def compute_spatial_density(
    samples: torch.Tensor,
    bandwidth: float = 0.1,
    grid_size: int = 50,
    domain_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute spatial density function σ(x) using kernel density estimation.
    
    This function estimates the density at grid points and sample locations,
    implementing the theoretical σ(x) from the congested transport framework.
    
    Args:
        samples (torch.Tensor): Sample points, shape (n_samples, data_dim).
        bandwidth (float): KDE bandwidth parameter. Defaults to 0.1.
        grid_size (int): Number of grid points per dimension. Defaults to 50.
        domain_bounds (Optional[Tuple[torch.Tensor, torch.Tensor]]): Min and max bounds for domain.
            If None, computed from samples with padding.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'density_at_samples': Density values at sample locations
            - 'grid_points': Grid point coordinates
            - 'density_at_grid': Density values at grid points
            - 'bandwidth': Effective bandwidth used
    """
    device = samples.device
    n_samples, data_dim = samples.shape
    
    # Determine domain bounds
    if domain_bounds is None:
        mins = samples.min(dim=0)[0] - 3 * bandwidth
        maxs = samples.max(dim=0)[0] + 3 * bandwidth
    else:
        mins, maxs = domain_bounds
    
    # Create grid for density estimation
    if data_dim == 2:
        x = torch.linspace(mins[0], maxs[0], grid_size, device=device)
        y = torch.linspace(mins[1], maxs[1], grid_size, device=device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing='ij')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    else:
        # For higher dimensions, use random grid points
        grid_points = torch.rand(grid_size**2, data_dim, device=device)
        grid_points = grid_points * (maxs - mins) + mins
    
    # Compute KDE at grid points
    density_at_grid = torch.zeros(grid_points.shape[0], device=device)
    density_at_samples = torch.zeros(n_samples, device=device)
    
    # Gaussian kernel
    def gaussian_kernel(x, y, bandwidth):
        dist_sq = ((x - y) ** 2).sum(dim=-1)
        return torch.exp(-dist_sq / (2 * bandwidth ** 2)) / ((2 * np.pi * bandwidth ** 2) ** (data_dim / 2))
    
    # Compute density at grid points
    for i in range(n_samples):
        density_at_grid += gaussian_kernel(grid_points, samples[i:i+1], bandwidth)
    density_at_grid /= n_samples
    
    # Compute density at sample points (leave-one-out)
    for i in range(n_samples):
        mask = torch.ones(n_samples, dtype=torch.bool, device=device)
        mask[i] = False
        other_samples = samples[mask]
        density_at_samples[i] = gaussian_kernel(samples[i:i+1], other_samples, bandwidth).sum() / (n_samples - 1)
    
    # Add small epsilon to avoid division by zero
    density_at_samples = density_at_samples + 1e-8
    density_at_grid = density_at_grid + 1e-8
    
    return {
        'density_at_samples': density_at_samples,
        'grid_points': grid_points,
        'density_at_grid': density_at_grid,
        'bandwidth': bandwidth
    }


def compute_traffic_flow(
    critic: torch.nn.Module,
    generator: torch.nn.Module,
    noise_samples: torch.Tensor,
    sigma: torch.Tensor,
    lambda_param: float = 0.1
) -> Dict[str, torch.Tensor]:
    """
    Compute traffic flow w_Q and intensity i_Q based on critic gradients.
    
    Implements the theoretical relationship:
    w_Q = -λσ(|∇u| - 1)_+ ∇u/|∇u|
    
    Args:
        critic (torch.nn.Module): Trained critic (dual potential u).
        generator (torch.nn.Module): Generator model.
        noise_samples (torch.Tensor): Noise input samples.
        sigma (torch.Tensor): Spatial density values.
        lambda_param (float): Congestion parameter λ. Defaults to 0.1.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'traffic_flow': Vector field w_Q
            - 'traffic_intensity': Scalar field i_Q
            - 'critic_gradients': Raw gradients ∇u
            - 'gradient_norm': |∇u|
    """
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
    traffic_flow = -lambda_param * sigma.unsqueeze(1) * gradient_excess * (critic_gradients / gradient_norm_safe)
    
    # Compute traffic intensity: i_Q = |w_Q|
    traffic_intensity = torch.norm(traffic_flow, p=2, dim=1)
    
    return {
        'traffic_flow': traffic_flow,
        'traffic_intensity': traffic_intensity,
        'critic_gradients': critic_gradients,
        'gradient_norm': gradient_norm.squeeze()
    }


class CongestionCostFunction(ABC):
    """
    Abstract base class for congestion cost functions H(x, i).
    """
    
    @abstractmethod
    def __call__(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute congestion cost H(x, i) for given traffic intensity."""
        pass
    
    @abstractmethod
    def derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute derivative H'(x, i) with respect to traffic intensity."""
        pass


class QuadraticLinearCost(CongestionCostFunction):
    """
    Quadratic-linear congestion cost: H(x, z) = (1/2λσ(x))z² + |z|
    """
    
    def __init__(self, lambda_param: float = 0.1):
        self.lambda_param = lambda_param
    
    def __call__(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        sigma_safe = torch.clamp(sigma, min=1e-8)
        quadratic_term = traffic_intensity ** 2 / (2 * self.lambda_param * sigma_safe)
        linear_term = torch.abs(traffic_intensity)
        return quadratic_term + linear_term
    
    def derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        sigma_safe = torch.clamp(sigma, min=1e-8)
        return traffic_intensity / (self.lambda_param * sigma_safe) + torch.sign(traffic_intensity)


class PowerLawCost(CongestionCostFunction):
    """
    Power law congestion cost: H(x, z) = (m/(m-1))z^(m/(m-1)) for m > 1
    """
    
    def __init__(self, m: float = 2.0):
        if m <= 1:
            raise ValueError("Parameter m must be > 1")
        self.m = m
        self.exponent = m / (m - 1)
        self.coefficient = m / (m - 1)
    
    def __call__(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        z_safe = torch.clamp(traffic_intensity, min=1e-8)
        return self.coefficient * torch.pow(z_safe, self.exponent)
    
    def derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        z_safe = torch.clamp(traffic_intensity, min=1e-8)
        return torch.pow(z_safe, 1.0 / (self.m - 1))


class LogarithmicCost(CongestionCostFunction):
    """
    Logarithmic congestion cost: H(x, z) = z log(z) - z + 1
    """
    
    def __call__(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        z_safe = torch.clamp(traffic_intensity, min=1e-8)
        return z_safe * torch.log(z_safe) - z_safe + 1
    
    def derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        z_safe = torch.clamp(traffic_intensity, min=1e-8)
        return torch.log(z_safe)


def congestion_cost_function(
    traffic_intensity: torch.Tensor,
    sigma: torch.Tensor,
    lambda_param: float = 0.1,
    cost_type: str = 'quadratic_linear'
) -> torch.Tensor:
    """
    Compute congestion cost H(x, i) for given traffic intensity.
    
    Factory function for different congestion cost functions.
    
    Args:
        traffic_intensity (torch.Tensor): Traffic intensity i_Q(x).
        sigma (torch.Tensor): Spatial density σ(x).
        lambda_param (float): Congestion parameter. Defaults to 0.1.
        cost_type (str): Type of cost function. Defaults to 'quadratic_linear'.
    
    Returns:
        torch.Tensor: Congestion cost values.
    """
    if cost_type == 'quadratic_linear':
        cost_fn = QuadraticLinearCost(lambda_param)
    elif cost_type == 'power_law':
        cost_fn = PowerLawCost(m=2.0)
    elif cost_type == 'logarithmic':
        cost_fn = LogarithmicCost()
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")
    
    return cost_fn(traffic_intensity, sigma)


def verify_continuity_equation(
    flow: torch.Tensor,
    density_change: torch.Tensor,
    grid_points: torch.Tensor,
    epsilon: float = 1e-3
) -> Dict[str, torch.Tensor]:
    """
    Verify the continuity equation ∂_t ρ + ∇·w = 0 for mass conservation.
    
    Args:
        flow (torch.Tensor): Traffic flow vector field w.
        density_change (torch.Tensor): Time derivative of density ∂_t ρ.
        grid_points (torch.Tensor): Grid points where fields are evaluated.
        epsilon (float): Tolerance for divergence computation. Defaults to 1e-3.
    
    Returns:
        Dict[str, torch.Tensor]: Dictionary containing:
            - 'divergence': Divergence of flow field
            - 'residual': Continuity equation residual
            - 'max_violation': Maximum violation of continuity
            - 'is_satisfied': Boolean tensor indicating satisfaction
    """
    # Approximate divergence using finite differences
    # This is a simplified 2D implementation
    if grid_points.shape[1] != 2:
        raise NotImplementedError("Currently only 2D divergence is implemented")
    
    # Find grid structure (assuming regular grid)
    x_unique = torch.unique(grid_points[:, 0])
    y_unique = torch.unique(grid_points[:, 1])
    nx, ny = len(x_unique), len(y_unique)
    
    if nx * ny != grid_points.shape[0]:
        raise ValueError("Grid points must form a regular grid for divergence computation")
    
    # Reshape flow to grid
    flow_grid = flow.reshape(nx, ny, 2)
    
    # Compute divergence using central differences
    dx = x_unique[1] - x_unique[0] if nx > 1 else 1.0
    dy = y_unique[1] - y_unique[0] if ny > 1 else 1.0
    
    div = torch.zeros(nx, ny, device=flow.device)
    
    # ∂w_x/∂x
    div[1:-1, :] += (flow_grid[2:, :, 0] - flow_grid[:-2, :, 0]) / (2 * dx)
    div[0, :] = (flow_grid[1, :, 0] - flow_grid[0, :, 0]) / dx
    div[-1, :] = (flow_grid[-1, :, 0] - flow_grid[-2, :, 0]) / dx
    
    # ∂w_y/∂y
    div[:, 1:-1] += (flow_grid[:, 2:, 1] - flow_grid[:, :-2, 1]) / (2 * dy)
    div[:, 0] += (flow_grid[:, 1, 1] - flow_grid[:, 0, 1]) / dy
    div[:, -1] += (flow_grid[:, -1, 1] - flow_grid[:, -2, 1]) / dy
    
    div_flat = div.flatten()
    
    # Compute residual: ∂_t ρ + ∇·w
    residual = density_change + div_flat
    
    # Check satisfaction
    max_violation = torch.abs(residual).max()
    is_satisfied = max_violation < epsilon
    
    return {
        'divergence': div_flat,
        'residual': residual,
        'max_violation': max_violation,
        'is_satisfied': is_satisfied
    }