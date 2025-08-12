"""
Additional loss functions integrating congestion theory and Sobolev regularization.

This module extends the basic losses.py with the theoretical components:
- Global W2 loss with congestion tracking
- Multi-marginal OT with congestion awareness
- Integration with spatial density and traffic flow
"""

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from typing import Callable, List, Optional, Tuple, Dict

from .congestion import (
    compute_spatial_density, 
    compute_traffic_flow, 
    congestion_cost_function,
    CongestionTracker
)
from .sobolev import sobolev_regularization, SobolevWGANLoss
from .losses import barycentric_ot_map

def global_w2_loss_and_grad_with_congestion(
    generator: torch.nn.Module,
    target_samples: torch.Tensor,
    noise_samples: torch.Tensor,
    critic: Optional[torch.nn.Module] = None,
    lambda_congestion: float = 1.0,
    lambda_sobolev: float = 0.1,
    track_congestion: bool = True,
    map_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = barycentric_ot_map,
    use_direct_w2: bool = True,
    w2_weight: float = 1.0,
    map_weight: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Enhanced version of global_w2_loss_and_grad with congestion tracking.
    
    This function extends the original by incorporating spatial density estimation,
    traffic flow computation, and Sobolev regularization as per the theoretical framework.
    
    Args:
        generator (torch.nn.Module): Pre-trained generator model.
        target_samples (torch.Tensor): Target distribution samples.
        noise_samples (torch.Tensor): Input noise for generator.
        critic (Optional[torch.nn.Module]): Critic for traffic flow computation.
        lambda_congestion (float): Congestion parameter. Defaults to 0.1.
        lambda_sobolev (float): Sobolev regularization strength. Defaults to 0.01.
        track_congestion (bool): Whether to compute and return congestion metrics.
        map_fn (Callable): Function to compute OT map.
        use_direct_w2 (bool): Whether to include direct W2 loss.
        w2_weight (float): Weight for direct W2 loss.
        map_weight (float): Weight for mapping loss.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]: 
            - loss: Total loss scalar
            - grads: Flattened gradients
            - congestion_info: Dictionary with congestion metrics (if track_congestion=True)
    """
    generator.train()
    
    # Enable gradients for parameters
    for param in generator.parameters():
        param.requires_grad_(True)
    
    gen_out = generator(noise_samples)
    
    if gen_out.shape[1] != target_samples.shape[1]:
        raise ValueError("Generator output dimension must match target dimension.")
    
    total_loss = 0.0
    congestion_info = {} if track_congestion else None
    
    # Direct W2 loss component
    if use_direct_w2:
        sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
        w2_loss = sinkhorn(gen_out, target_samples)
        total_loss += w2_weight * w2_loss
    
    # Barycentric mapping loss component
    if map_weight > 0:
        matched = map_fn(gen_out, target_samples)
        map_loss = ((gen_out - matched) ** 2).sum(dim=1).mean()
        total_loss += map_weight * map_loss
    
    # Add congestion-aware components if critic is provided
    if critic is not None and track_congestion:
        try:
            # Compute spatial density
            density_info = compute_spatial_density(gen_out, bandwidth=0.15)
            sigma = density_info['density_at_samples']
            
            # Compute traffic flow
            flow_info = compute_traffic_flow(
                critic, generator, noise_samples, sigma, lambda_congestion
            )

            # Compute congestion cost
            congestion_cost = congestion_cost_function(
                flow_info['traffic_intensity'], sigma, lambda_congestion
            ).mean()
            total_loss *= congestion_cost


            # Add Sobolev regularization
            sobolev_loss = sobolev_regularization(critic, gen_out, sigma, lambda_sobolev)
            total_loss += sobolev_loss

            # Store congestion information
            congestion_info = {
                'spatial_density': sigma,
                'traffic_flow': flow_info['traffic_flow'],
                'traffic_intensity': flow_info['traffic_intensity'],
                'congestion_cost': congestion_cost,
                'sobolev_loss': sobolev_loss,
                'gradient_norm': flow_info['gradient_norm']
            }
        except Exception as e:
            print(f"Warning: Congestion computation failed: {e}")
            congestion_info = {
                'spatial_density': torch.zeros(gen_out.shape[0], device=gen_out.device),
                'traffic_flow': torch.zeros_like(gen_out),
                'traffic_intensity': torch.zeros(gen_out.shape[0], device=gen_out.device),
                'congestion_cost': torch.tensor(0.0, device=gen_out.device),
                'sobolev_loss': torch.tensor(0.0, device=gen_out.device),
                'gradient_norm': torch.zeros(gen_out.shape[0], device=gen_out.device)
            }
    
    # Add diversity regularization
    if gen_out.shape[0] > 1:
        try:
            gen_cov = torch.cov(gen_out.t())
            diversity_loss = -torch.logdet(gen_cov + 1e-6 * torch.eye(gen_out.shape[1], device=gen_out.device))
            total_loss += 0.01 * diversity_loss
        except:
            # Fallback diversity term
            gen_std = gen_out.std(dim=0).mean()
            diversity_loss = -torch.log(gen_std + 1e-6)
            total_loss += 0.01 * diversity_loss
    
    # Compute gradients
    generator.zero_grad()
    
    try:
        total_loss.backward()
        grads = torch.cat([p.grad.view(-1) for p in generator.parameters() if p.grad is not None])
        
        if grads.numel() == 0:
            total_params = sum(p.numel() for p in generator.parameters())
            grads = torch.zeros(total_params, device=total_loss.device)
            
    except RuntimeError as e:
        print(f"Warning: Gradient computation failed: {e}")
        total_params = sum(p.numel() for p in generator.parameters())
        grads = torch.zeros(total_params, device=total_loss.device)
    
    return total_loss.detach(), grads, congestion_info


def multi_marginal_ot_loss_with_congestion(
    generator_outputs: torch.Tensor,
    evidence_list: List[torch.Tensor],
    virtual_targets: torch.Tensor,
    critics: Optional[List[torch.nn.Module]] = None,
    weights: Optional[List[float]] = None,
    blur: float = 0.06,
    lambda_virtual: float = 0.8,
    lambda_multi: float = 1.0,
    lambda_entropy: float = 0.012,
    lambda_congestion: float = 1.0,
    lambda_sobolev: float = 0.1,
    track_congestion: bool = True
) -> Tuple[torch.Tensor, Optional[Dict[str, List[Dict]]]]:
    """
    Enhanced multi-marginal OT loss with congestion tracking for each evidence domain.
    
    Args:
        generator_outputs (torch.Tensor): Generated samples.
        evidence_list (List[torch.Tensor]): List of evidence tensors.
        virtual_targets (torch.Tensor): Virtual target samples.
        critics (Optional[List[torch.nn.Module]]): Critics for each evidence domain.
        weights (Optional[List[float]]): Weights for each evidence domain.
        blur (float): Entropic blur for Sinkhorn.
        lambda_virtual (float): Coefficient for virtual target OT loss.
        lambda_multi (float): Coefficient for multi-marginal evidence OT loss.
        lambda_entropy (float): Coefficient for entropy regularization.
        lambda_congestion (float): Congestion parameter.
        lambda_sobolev (float): Sobolev regularization strength.
        track_congestion (bool): Whether to track congestion metrics.
    
    Returns:
        Tuple[torch.Tensor, Optional[Dict]]: 
            - loss: Total multi-marginal loss
            - congestion_info: Multi-domain congestion information (if track_congestion=True)
    """
    num_domains = len(evidence_list)
    if num_domains == 0:
        raise ValueError("Evidence list must not be empty.")
    
    if weights is None:
        weights = [1.0 / num_domains] * num_domains
    elif len(weights) != num_domains:
        raise ValueError("Weights must match the number of evidence domains.")
    
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
    
    # Virtual target loss
    loss_virtual = sinkhorn(generator_outputs, virtual_targets)

    # Multi-marginal evidence loss with congestion tracking
    loss_multi = 0.0
    multi_congestion_info = {'domains': []} if track_congestion else None
    
    for i, evidence in enumerate(evidence_list):
        if generator_outputs.shape[1] != evidence.shape[1]:
            raise ValueError("All tensors must have the same data dimension.")
        
        # Standard pairwise OT loss
        pairwise_loss = sinkhorn(generator_outputs, evidence)
        loss_multi += weights[i] * pairwise_loss
        
        # Add congestion tracking for this domain
        if track_congestion and critics is not None and i < len(critics):
            critic = critics[i]
            
            try:
                # Compute spatial density for this evidence domain
                all_samples = torch.cat([generator_outputs, evidence], dim=0)
                density_info = compute_spatial_density(all_samples, bandwidth=0.15)
                sigma_gen = density_info['density_at_samples'][:generator_outputs.shape[0]]
                
                # Compute traffic flow for this domain
                def dummy_generator(z):
                    return generator_outputs
                
                flow_info = compute_traffic_flow(
                    critic, dummy_generator, 
                    torch.randn(generator_outputs.shape[0], 2, device=generator_outputs.device), 
                    sigma_gen, lambda_congestion
                )
                
                # Compute congestion cost
                congestion_cost = congestion_cost_function(
                    flow_info['traffic_intensity'], sigma_gen, lambda_congestion
                ).mean()
                loss_multi *= congestion_cost

                # Add Sobolev regularization for this domain
                sobolev_loss = sobolev_regularization(critic, generator_outputs, sigma_gen, lambda_sobolev)
                loss_multi += sobolev_loss
                
                domain_info = {
                    'domain_id': i,
                    'spatial_density': sigma_gen,
                    'traffic_flow': flow_info['traffic_flow'],
                    'traffic_intensity': flow_info['traffic_intensity'],
                    'congestion_cost': congestion_cost,
                    'sobolev_loss': sobolev_loss,
                    'gradient_norm': flow_info['gradient_norm']
                }
                
                multi_congestion_info['domains'].append(domain_info)
                
            except Exception as e:
                print(f"Warning: Congestion computation failed for domain {i}: {e}")
                # Create dummy domain info
                domain_info = {
                    'domain_id': i,
                    'spatial_density': torch.zeros(generator_outputs.shape[0], device=generator_outputs.device),
                    'traffic_flow': torch.zeros_like(generator_outputs),
                    'traffic_intensity': torch.zeros(generator_outputs.shape[0], device=generator_outputs.device),
                    'congestion_cost': torch.tensor(0.0, device=generator_outputs.device),
                    'sobolev_loss': torch.tensor(0.0, device=generator_outputs.device),
                    'gradient_norm': torch.zeros(generator_outputs.shape[0], device=generator_outputs.device)
                }
                if multi_congestion_info:
                    multi_congestion_info['domains'].append(domain_info)
    
    # Entropy regularization (e.g., via covariance determinant for diversity)
    data_dim = generator_outputs.shape[1]
    try:
        cov = torch.cov(generator_outputs.t()) + torch.eye(data_dim, device=generator_outputs.device) * 1e-5
        eigenvals = torch.linalg.eigvals(cov).real
        eigenvals = torch.clamp(eigenvals, min=1e-8)
        log_det = torch.sum(torch.log(eigenvals))
        entropy_reg = -lambda_entropy * log_det
    except:
        # Fallback entropy regularization
        entropy_reg = -lambda_entropy * torch.log(generator_outputs.std(dim=0).mean() + 1e-8)

    total_loss = lambda_virtual * loss_virtual + lambda_multi * loss_multi + entropy_reg
    return total_loss, multi_congestion_info


class CongestionAwareLossFunction:
    """
    Comprehensive loss function that integrates all congestion transport components.
    """
    
    def __init__(
        self,
        lambda_congestion: float = 1.0,
        lambda_sobolev: float = 0.1,
        lambda_entropy: float = 0.012,
        congestion_cost_type: str = 'quadratic_linear',
        use_adaptive_weights: bool = True
    ):
        self.lambda_congestion = lambda_congestion
        self.lambda_sobolev = lambda_sobolev
        self.lambda_entropy = lambda_entropy
        self.congestion_cost_type = congestion_cost_type
        self.use_adaptive_weights = use_adaptive_weights
        
        # Initialize congestion tracker
        self.congestion_tracker = CongestionTracker(lambda_congestion)
        
        # Initialize Sobolev loss
        self.sobolev_loss = SobolevWGANLoss(
            lambda_sobolev=lambda_sobolev,
            use_adaptive_sobolev=use_adaptive_weights
        )
    
    def compute_target_given_loss(
        self,
        generator: torch.nn.Module,
        target_samples: torch.Tensor,
        noise_samples: torch.Tensor,
        critic: Optional[torch.nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute enhanced loss for target-given perturbation.
        
        Returns:
            Tuple[loss, gradients, congestion_info]
        """
        return global_w2_loss_and_grad_with_congestion(
            generator, target_samples, noise_samples, critic,
            lambda_congestion=self.lambda_congestion,
            lambda_sobolev=self.lambda_sobolev,
            track_congestion=True
        )
    
    def compute_evidence_based_loss(
        self,
        generator_outputs: torch.Tensor,
        evidence_list: List[torch.Tensor],
        virtual_targets: torch.Tensor,
        critics: Optional[List[torch.nn.Module]] = None,
        weights: Optional[List[float]] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute enhanced loss for evidence-based perturbation.
        
        Returns:
            Tuple[loss, multi_congestion_info]
        """
        return multi_marginal_ot_loss_with_congestion(
            generator_outputs, evidence_list, virtual_targets, critics,
            weights=weights,
            lambda_congestion=self.lambda_congestion,
            lambda_sobolev=self.lambda_sobolev,
            lambda_entropy=self.lambda_entropy,
            track_congestion=True
        )
    
    def update_congestion_history(self, congestion_info: Dict) -> None:
        """Update congestion tracking history."""
        self.congestion_tracker.update(congestion_info)
    
    def check_congestion_constraints(self, threshold: float = 0.15) -> bool:
        """Check if congestion constraints are satisfied."""
        return not self.congestion_tracker.check_congestion_increase(threshold)
    
    def get_congestion_statistics(self) -> Dict[str, float]:
        """Get summary statistics of congestion tracking."""
        return {
            'average_congestion': self.congestion_tracker.get_average_congestion(),
            'recent_congestion': self.congestion_tracker.history['congestion_cost'][-1] if self.congestion_tracker.history['congestion_cost'] else 0.0,
            'congestion_trend': self.congestion_tracker.get_average_congestion(window=3) - self.congestion_tracker.get_average_congestion(window=10) if len(self.congestion_tracker.history['congestion_cost']) >= 10 else 0.0
        }


def compute_convergence_metrics(
    generator: torch.nn.Module,
    target_or_evidence: torch.Tensor,
    noise_samples: torch.Tensor,
    history_window: int = 10
) -> Dict[str, float]:
    """
    Compute convergence metrics for perturbation monitoring.
    
    Args:
        generator (torch.nn.Module): Current generator.
        target_or_evidence (torch.Tensor): Target or evidence samples.
        noise_samples (torch.Tensor): Noise input.
        history_window (int): Window size for trend analysis.
    
    Returns:
        Dict[str, float]: Convergence metrics.
    """
    with torch.no_grad():
        gen_samples = generator(noise_samples)
    
    # Compute Wasserstein distances
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    w2_distance = sinkhorn(gen_samples, target_or_evidence).item()
    
    # Compute sample statistics
    gen_mean = gen_samples.mean(dim=0)
    target_mean = target_or_evidence.mean(dim=0)
    mean_shift = torch.norm(gen_mean - target_mean).item()
    
    gen_std = gen_samples.std(dim=0)
    target_std = target_or_evidence.std(dim=0)
    std_ratio = (gen_std / (target_std + 1e-8)).mean().item()
    
    return {
        'w2_distance': w2_distance,
        'mean_shift': mean_shift,
        'std_ratio': std_ratio,
        'convergence_score': 1.0 / (1.0 + w2_distance + mean_shift + abs(std_ratio - 1.0))
    }