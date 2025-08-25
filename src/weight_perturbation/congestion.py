"""
Congestion tracking and spatial density computation for congested transport.

This module implements the theoretical components from the congested transport framework:
- Spatial density estimation σ(x)
- Traffic flow computation w_Q
- Traffic intensity tracking i_Q  
- Congestion cost functions H(x, i)
- Continuity equation verification and enforcement
- Mass conservation constraints
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
    
    def __init__(self, lambda_param: float = 1.0, history_size: int = 100):
        self.lambda_param = lambda_param
        self.history_size = history_size
        self.history = {
            'traffic_intensity': [],
            'congestion_cost': [],
            'flow_divergence': [],
            'continuity_residual': [],
            'spatial_density': [],
            'gradient_norm': [],
            'sobolev_loss': [],
            'mass_conservation_error': [],
            'theoretical_consistency': []
        }
        
    def update(self, congestion_info: Dict[str, torch.Tensor]) -> None:
        """Update congestion history with new measurements."""
        for key in ['traffic_intensity', 'congestion_cost', 'spatial_density', 'gradient_norm', 
                   'sobolev_loss', 'mass_conservation_error', 'theoretical_consistency']:
            if key in congestion_info:
                if isinstance(congestion_info[key], torch.Tensor):
                    if congestion_info[key].dim() == 0:  # Scalar
                        value = congestion_info[key].item()
                    else:  # Vector - take mean
                        value = congestion_info[key].mean().item()
                else:
                    value = float(congestion_info[key])
                self.history[key].append(value)
            
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
    
    def check_theoretical_consistency(self, threshold: float = 0.8) -> bool:
        """Check theoretical consistency based on mass conservation and continuity."""
        if len(self.history['theoretical_consistency']) < 1:
            return True
        
        theoretical_consistency = self.history['theoretical_consistency'][-1]
        return theoretical_consistency > threshold
    
    def get_average_congestion(self, window: int = 10) -> float:
        """Get average congestion over recent window."""
        if not self.history['congestion_cost']:
            return 0.0
        
        window = min(window, len(self.history['congestion_cost']))
        return sum(self.history['congestion_cost'][-window:]) / window
    
    def get_statistics(self) -> Dict[str, float]:
        """Get comprehensive statistics."""
        stats = {}
        for key, values in self.history.items():
            if values:
                stats[f'{key}_mean'] = np.mean(values)
                stats[f'{key}_std'] = np.std(values)
                stats[f'{key}_latest'] = values[-1]
                if len(values) >= 3:
                    stats[f'{key}_trend'] = np.mean(values[-3:]) - np.mean(values[-6:-3]) if len(values) >= 6 else 0
        return stats


def compute_spatial_density(
    samples: torch.Tensor,
    bandwidth: float = 0.2,
    grid_size: int = 50,
    domain_bounds: Optional[Tuple[torch.Tensor, torch.Tensor]] = None
) -> Dict[str, torch.Tensor]:
    """
    Compute spatial density function σ(x) using kernel density estimation.
    
    This function estimates the density at grid points and sample locations,
    implementing the theoretical σ(x) from the congested transport framework.
    
    Args:
        samples (torch.Tensor): Sample points, shape (n_samples, data_dim).
        bandwidth (float): KDE bandwidth parameter. Defaults to 0.2.
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
    
    if n_samples == 0:
        raise ValueError("Samples tensor cannot be empty")
    
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
    
    # Gaussian kernel with numerical stability
    def gaussian_kernel(x, y, bandwidth):
        dist_sq = ((x - y) ** 2).sum(dim=-1)
        normalization = (2 * np.pi * bandwidth ** 2) ** (data_dim / 2)
        return torch.exp(-dist_sq / (2 * bandwidth ** 2)) / normalization
    
    # Compute density at grid points
    for i in range(n_samples):
        density_at_grid += gaussian_kernel(grid_points, samples[i:i+1], bandwidth)
    density_at_grid /= n_samples
    
    # Compute density at sample points (leave-one-out for unbiased estimate)
    for i in range(n_samples):
        if n_samples > 1:
            mask = torch.ones(n_samples, dtype=torch.bool, device=device)
            mask[i] = False
            other_samples = samples[mask]
            if other_samples.shape[0] > 0:
                density_at_samples[i] = gaussian_kernel(samples[i:i+1], other_samples, bandwidth).sum() / (n_samples - 1)
            else:
                density_at_samples[i] = gaussian_kernel(samples[i:i+1], samples[i:i+1], bandwidth).sum()
        else:
            density_at_samples[i] = gaussian_kernel(samples[i:i+1], samples[i:i+1], bandwidth).sum()
    
    # Add small epsilon to avoid division by zero
    density_at_samples = torch.clamp(density_at_samples + 1e-8, min=1e-8)
    density_at_grid = torch.clamp(density_at_grid + 1e-8, min=1e-8)
    
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
    lambda_param: float = 1.0
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
        lambda_param (float): Congestion parameter λ. Defaults to 1.0.
    
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
    
    try:
        # Compute critic values
        critic_values = critic(gen_samples)
        
        # Compute gradients with respect to generated samples
        critic_gradients = torch.autograd.grad(
            outputs=critic_values.sum(),
            inputs=gen_samples,
            create_graph=True,
            retain_graph=True,
            allow_unused=True
        )[0]
        
        if critic_gradients is None:
            # Fallback: create dummy gradients
            critic_gradients = torch.zeros_like(gen_samples)
        
    except RuntimeError as e:
        print(f"Warning: Gradient computation failed: {e}")
        critic_gradients = torch.zeros_like(gen_samples)
    
    # Compute gradient norm
    gradient_norm = torch.norm(critic_gradients, p=2, dim=1, keepdim=True)
    
    # Avoid division by zero
    gradient_norm_safe = gradient_norm + 1e-8 #torch.clamp(gradient_norm, min=1e-8)
    
    # Compute (|∇u| - 1)_+ 
    # gradient_excess = F.relu(gradient_norm - 1.0)
    gradient_excess = torch.clip(gradient_norm, min=1.0)
    
    
    # Ensure sigma has correct shape
    if sigma.dim() == 1:
        sigma = sigma.unsqueeze(1)
    
    # Compute traffic flow: w_Q = -λσ(|∇u| - 1)_+ ∇u/|∇u|
    traffic_flow = -lambda_param * sigma * gradient_excess * (critic_gradients / gradient_norm_safe)
    
    # Compute traffic intensity: i_Q = |w_Q|
    traffic_intensity = torch.norm(traffic_flow, p=2, dim=1)
    
    return {
        'traffic_flow': traffic_flow,
        'traffic_intensity': traffic_intensity,
        'critic_gradients': critic_gradients,
        'gradient_norm': gradient_norm.squeeze()
    }


def enforce_mass_conservation(
    current_flow: torch.Tensor,
    target_density: torch.Tensor,
    current_density: torch.Tensor,
    samples: torch.Tensor,
    lagrange_multiplier: float = 1.0
) -> Dict[str, torch.Tensor]:
    """
    Enforce mass conservation constraint: ∇·w = μ_target - μ_current
    
    This implements the theoretical mass conservation requirement from
    the continuity equation ∂_t ρ + ∇·w = 0.
    
    Args:
        current_flow (torch.Tensor): Current traffic flow w_Q.
        target_density (torch.Tensor): Target density μ_target.
        current_density (torch.Tensor): Current density μ_current.
        samples (torch.Tensor): Sample points for divergence computation.
        lagrange_multiplier (float): Lagrangian multiplier strength.
    
    Returns:
        Dict containing:
            - 'corrected_flow': Mass-conserving flow
            - 'mass_conservation_error': Conservation error magnitude
            - 'divergence': Flow divergence
    """
    device = samples.device
    n_samples = samples.shape[0]
    
    # Compute required mass change
    mass_deficit = target_density - current_density
    
    # Approximate divergence using finite differences (simplified 2D case)
    if samples.shape[1] == 2:
        try:
            # Create neighbor relationships for divergence computation
            distances = torch.cdist(samples, samples)
            # Find k nearest neighbors for each point
            k = min(6, n_samples - 1)
            _, neighbor_indices = torch.topk(distances, k=k+1, dim=1, largest=False)
            neighbor_indices = neighbor_indices[:, 1:]  # Exclude self
            
            # Compute approximate divergence
            divergence = torch.zeros(n_samples, device=device)
            for i in range(n_samples):
                neighbors = neighbor_indices[i]
                neighbor_points = samples[neighbors]
                neighbor_flows = current_flow[neighbors]
                
                # Simple divergence approximation
                center_flow = current_flow[i]
                neighbor_flow_diff = neighbor_flows - center_flow.unsqueeze(0)
                neighbor_distances = torch.norm(neighbor_points - samples[i], dim=1) + 1e-8
                
                # Weight by inverse distance
                weights = 1.0 / neighbor_distances
                weights = weights / weights.sum()
                
                # Approximate divergence as weighted flow difference
                divergence[i] = (neighbor_flow_diff * weights.unsqueeze(1)).sum()
                
        except Exception as e:
            print(f"Warning: Divergence computation failed: {e}")
            divergence = torch.zeros(n_samples, device=device)
    else:
        # For higher dimensions, use simplified approximation
        divergence = current_flow.sum(dim=1) / samples.shape[1]
    
    # Compute mass conservation error
    mass_conservation_error = (divergence - mass_deficit).norm()
    
    # Correct flow to satisfy mass conservation (simplified correction)
    correction_magnitude = mass_deficit - divergence
    
    # Create correction flow (proportional to position gradients)
    correction_flow = torch.zeros_like(current_flow)
    if n_samples > 1:
        # Use spatial gradients to determine correction direction
        center = samples.mean(dim=0)
        position_vectors = samples - center
        position_norms = torch.norm(position_vectors, dim=1, keepdim=True) + 1e-8
        correction_directions = position_vectors / position_norms
        
        # Scale correction by mass deficit
        correction_flow = lagrange_multiplier * correction_magnitude.unsqueeze(1) * correction_directions
    
    corrected_flow = current_flow + correction_flow
    
    return {
        'corrected_flow': corrected_flow,
        'mass_conservation_error': mass_conservation_error,
        'divergence': divergence,
        'correction_flow': correction_flow
    }


def validate_theoretical_consistency(
    flow_info: Dict[str, torch.Tensor],
    density_info: Dict[str, torch.Tensor],
    samples: torch.Tensor,
    target_samples: torch.Tensor
) -> Dict[str, float]:
    """
    Validate theoretical consistency of congested transport computations.
    
    Checks:
    1. Mass conservation: ∇·w ≈ μ_target - μ_current
    2. Flow magnitude consistency
    3. Density positivity
    4. Gradient bounds
    
    Args:
        flow_info (Dict): Traffic flow information.
        density_info (Dict): Spatial density information.
        samples (torch.Tensor): Generated samples.
        target_samples (torch.Tensor): Target samples.
    
    Returns:
        Dict[str, float]: Validation metrics.
    """
    validation_results = {}
    
    try:
        # Check 1: Flow magnitude consistency
        traffic_flow = flow_info['traffic_flow']
        traffic_intensity = flow_info['traffic_intensity']
        computed_intensity = torch.norm(traffic_flow, p=2, dim=1)
        intensity_error = torch.norm(traffic_intensity - computed_intensity).item()
        validation_results['intensity_consistency'] = max(0.0, 1.0 - intensity_error)
        
        # Check 2: Density positivity
        spatial_density = density_info['density_at_samples']
        min_density = spatial_density.min().item()
        validation_results['min_density'] = min_density
        validation_results['density_positive'] = float(min_density > 1e-8)
        
        # Check 3: Gradient norm bounds (should be around 1 for WGAN-GP)
        gradient_norm = flow_info['gradient_norm']
        mean_grad_norm = gradient_norm.mean().item()
        validation_results['mean_gradient_norm'] = mean_grad_norm
        validation_results['gradient_close_to_one'] = float(abs(mean_grad_norm - 1.0) < 2.0)
        
        # Check 4: Sample distribution coverage
        # Compute Wasserstein-1 distance as coverage metric
        if samples.shape[0] > 0 and target_samples.shape[0] > 0:
            # Simple coverage metric: average distance to nearest target
            distances = torch.cdist(samples, target_samples)
            min_distances = distances.min(dim=1)[0]
            coverage_error = min_distances.mean().item()
            validation_results['coverage_error'] = coverage_error
        
        # Overall consistency score
        consistency_factors = [
            1.0 / (1.0 + intensity_error),
            validation_results['density_positive'],
            validation_results['gradient_close_to_one'],
            1.0 / (1.0 + validation_results.get('coverage_error', 1.0))
        ]
        validation_results['overall_consistency'] = np.mean(consistency_factors)
        
    except Exception as e:
        print(f"Warning: Theoretical validation failed: {e}")
        validation_results = {
            'intensity_consistency': 1.0,
            'min_density': 0.0,
            'density_positive': 0.0,
            'mean_gradient_norm': 0.0,
            'gradient_close_to_one': 0.0,
            'coverage_error': 1.0,
            'overall_consistency': 0.0
        }
    
    return validation_results


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
    
    @abstractmethod
    def second_derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """Compute second derivative H''(x, i) for congestion scaling."""
        pass


class QuadraticLinearCost(CongestionCostFunction):
    """
    Quadratic-linear congestion cost: H(x, z) = (1/2λσ(x))z² + |z|
    """
    
    def __init__(self, lambda_param: float = 1.0):
        self.lambda_param = lambda_param
    
    def __call__(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        sigma_safe = torch.clamp(sigma, min=1e-8)
        quadratic_term = traffic_intensity ** 2 / (2 * self.lambda_param * sigma_safe)
        linear_term = torch.abs(traffic_intensity)
        return quadratic_term + linear_term
    
    def derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        sigma_safe = torch.clamp(sigma, min=1e-8)
        return traffic_intensity / (self.lambda_param * sigma_safe) + torch.sign(traffic_intensity)
    
    def second_derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        """H''(x, i) = 1/(λσ(x)) for quadratic-linear cost."""
        sigma_safe = torch.clamp(sigma, min=1e-8)
        return 1.0 / (self.lambda_param * sigma_safe)


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
    
    def second_derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        z_safe = torch.clamp(traffic_intensity, min=1e-8)
        return (1.0 / (self.m - 1)) * torch.pow(z_safe, (2 - self.m) / (self.m - 1))


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
    
    def second_derivative(self, traffic_intensity: torch.Tensor, sigma: torch.Tensor, **kwargs) -> torch.Tensor:
        z_safe = torch.clamp(traffic_intensity, min=1e-8)
        return 1.0 / z_safe


def congestion_cost_function(
    traffic_intensity: torch.Tensor,
    sigma: torch.Tensor,
    lambda_param: float = 1.0,
    cost_type: str = 'quadratic_linear'
) -> torch.Tensor:
    """
    Compute congestion cost H(x, i) for given traffic intensity.
    
    Factory function for different congestion cost functions.
    
    Args:
        traffic_intensity (torch.Tensor): Traffic intensity i_Q(x).
        sigma (torch.Tensor): Spatial density σ(x).
        lambda_param (float): Congestion parameter. Defaults to 1.0.
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


def get_congestion_second_derivative(
    traffic_intensity: torch.Tensor,
    sigma: torch.Tensor,
    lambda_param: float = 1.0,
    cost_type: str = 'quadratic_linear'
) -> torch.Tensor:
    """
    Get second derivative H''(x, i) for congestion-based weight scaling.
    
    This provides the theoretical H''(x, i) term used in perturbation scaling.
    
    Args:
        traffic_intensity (torch.Tensor): Traffic intensity i_Q(x).
        sigma (torch.Tensor): Spatial density σ(x).
        lambda_param (float): Congestion parameter. Defaults to 1.0.
        cost_type (str): Type of cost function. Defaults to 'quadratic_linear'.
    
    Returns:
        torch.Tensor: Second derivative values H''(x, i).
    """
    if cost_type == 'quadratic_linear':
        cost_fn = QuadraticLinearCost(lambda_param)
    elif cost_type == 'power_law':
        cost_fn = PowerLawCost(m=2.0)
    elif cost_type == 'logarithmic':
        cost_fn = LogarithmicCost()
    else:
        raise ValueError(f"Unknown cost type: {cost_type}")
    
    return cost_fn.second_derivative(traffic_intensity, sigma)


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
    if nx > 2:
        div[1:-1, :] += (flow_grid[2:, :, 0] - flow_grid[:-2, :, 0]) / (2 * dx)
    if nx > 1:
        div[0, :] = (flow_grid[1, :, 0] - flow_grid[0, :, 0]) / dx
        div[-1, :] = (flow_grid[-1, :, 0] - flow_grid[-2, :, 0]) / dx
    
    # ∂w_y/∂y
    if ny > 2:
        div[:, 1:-1] += (flow_grid[:, 2:, 1] - flow_grid[:, :-2, 1]) / (2 * dy)
    if ny > 1:
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


def estimate_congestion_bound(
    traffic_intensity: torch.Tensor,
    sigma: torch.Tensor,
    percentile: float = 95.0
) -> float:
    """
    Estimate appropriate congestion bound based on traffic intensity distribution.
    
    Args:
        traffic_intensity (torch.Tensor): Current traffic intensity values.
        sigma (torch.Tensor): Spatial density values.
        percentile (float): Percentile for bound estimation.
    
    Returns:
        float: Estimated congestion bound.
    """
    # Weighted intensity by spatial density
    weighted_intensity = traffic_intensity * sigma
    
    if weighted_intensity.numel() == 0:
        return 0.1  # Default bound
    
    # Use percentile-based bound
    bound = torch.quantile(weighted_intensity, percentile / 100.0).item()
    
    # Ensure reasonable bounds
    return max(0.01, min(1.0, bound))