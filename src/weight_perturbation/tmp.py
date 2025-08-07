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
    lambda_congestion: float = 0.1,
    lambda_sobolev: float = 0.01,
    track_congestion: bool = True,
    map_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = barycentric_ot_map,
    use_direct_w2: bool = True,
    w2_weight: float = 0.7,
    map_weight: float = 0.3
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
            
            # Add Sobolev regularization
            sobolev_loss = sobolev_regularization(critic, gen_out, sigma, lambda_sobolev)
            total_loss += sobolev_loss
            
            # Compute congestion cost
            congestion_cost = congestion_cost_function(
                flow_info['traffic_intensity'], sigma, lambda_congestion
            ).mean()
            
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
    lambda_congestion: float = 0.1,
    lambda_sobolev: float = 0.01,
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
    elif len(weights