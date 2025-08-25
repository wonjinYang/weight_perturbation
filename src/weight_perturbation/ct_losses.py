"""
Loss functions with congestion theory and Sobolev regularization.

This module extends losses.py with theoretical components:
- Global W2 loss with congestion tracking and mass conservation
- Multi-marginal OT with theoretical integration
- Integration with spatial density and traffic flow
- Theoretical validation and consistency checks
"""

import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from typing import Callable, List, Optional, Tuple, Dict

from .congestion import (
    compute_spatial_density, 
    compute_traffic_flow, 
    congestion_cost_function,
    get_congestion_second_derivative,
    enforce_mass_conservation,
    validate_theoretical_consistency,
    CongestionTracker,
)
from .sobolev import sobolev_regularization, SobolevWGANLoss
from .losses import barycentric_ot_map
from .utils import safe_density_resize

# Export list
__all__ = [
    'global_w2_loss_and_grad_with_congestion', 
    'multi_marginal_ot_loss_with_congestion',
    'CongestionAwareLossFunction',
    'compute_convergence_metrics'
]

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
    map_weight: float = 0.5,
    mass_conservation_weight: float = 1.0,
    enforce_mass_conservation_flag: bool = True,
    theoretical_validation: bool = True
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
    """
    Global W2 loss with congestion tracking.
    
    This function extends the original by incorporating:
    - Spatial density estimation and traffic flow computation
    - Mass conservation enforcement 
    - Theoretical validation
    - Congestion-aware loss scaling
    
    Args:
        generator (torch.nn.Module): Pre-trained generator model.
        target_samples (torch.Tensor): Target distribution samples.
        noise_samples (torch.Tensor): Input noise for generator.
        critic (Optional[torch.nn.Module]): Critic for traffic flow computation.
        lambda_congestion (float): Congestion parameter. Defaults to 1.0.
        lambda_sobolev (float): Sobolev regularization strength. Defaults to 0.1.
        track_congestion (bool): Whether to compute and return congestion metrics.
        map_fn (Callable): Function to compute OT map.
        use_direct_w2 (bool): Whether to include direct W2 loss.
        w2_weight (float): Weight for direct W2 loss.
        map_weight (float): Weight for mapping loss.
        enforce_mass_conservation_flag (bool): Whether to enforce mass conservation.
        theoretical_validation (bool): Whether to perform theoretical validation.
    
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
            
            # Apply congestion scaling to loss
            if congestion_cost > 1e-6:
                # Get H''(x,i) for theoretical scaling
                h_second = get_congestion_second_derivative(
                    flow_info['traffic_intensity'], sigma, lambda_congestion
                )
                
                # Congestion scaling factor
                congestion_scale = 1.0 + 0.05 * (h_second * flow_info['traffic_intensity']).mean()
                congestion_scale = max(0.5, min(congestion_scale, 1.5))
                
                total_loss = total_loss * congestion_scale
                congestion_info['congestion_scale'] = congestion_scale
            else:
                congestion_info['congestion_scale'] = 1.0

            # Add Sobolev regularization with adaptive strength
            if flow_info['traffic_intensity'].mean() > 0.1:
                adaptive_lambda_sobolev = lambda_sobolev * (0.5 + 0.2 * flow_info['traffic_intensity'].mean())
            else:
                adaptive_lambda_sobolev = lambda_sobolev
                
            sobolev_loss = sobolev_regularization(critic, gen_out, sigma, adaptive_lambda_sobolev)
            total_loss += sobolev_loss

            # Mass conservation enforcement
            if enforce_mass_conservation_flag:
                try:
                    target_density_info = compute_spatial_density(target_samples, bandwidth=0.25)
                    target_density = target_density_info['density_at_samples']
                    
                    # Use density resizing
                    target_density_resized, sigma_resized = safe_density_resize(
                        target_density, sigma, gen_out.shape[0]
                    )
                    
                    mass_conservation = enforce_mass_conservation(
                        flow_info['traffic_flow'],
                        target_density_resized,
                        sigma_resized,
                        gen_out,
                        lagrange_multiplier=mass_conservation_weight,
                    )
                    
                    # Add mass conservation penalty
                    mass_penalty = 1.0 * mass_conservation['mass_conservation_error']
                    total_loss += mass_penalty
                    
                    congestion_info['mass_conservation_error'] = mass_conservation['mass_conservation_error']
                    congestion_info['mass_penalty'] = mass_penalty
                    
                except Exception as e:
                    print(f"Warning: Mass conservation enforcement failed: {e}")
                    congestion_info['mass_conservation_error'] = torch.tensor(0.0, device=gen_out.device)
                    congestion_info['mass_penalty'] = torch.tensor(0.0, device=gen_out.device)

            # Store congestion information
            congestion_info.update({
                'spatial_density': sigma,
                'traffic_flow': flow_info['traffic_flow'],
                'traffic_intensity': flow_info['traffic_intensity'],
                'congestion_cost': congestion_cost,
                'sobolev_loss': sobolev_loss,
                'gradient_norm': flow_info['gradient_norm'],
                'adaptive_lambda_sobolev': adaptive_lambda_sobolev
            })
            
            # Theoretical validation
            if theoretical_validation:
                validation_results = validate_theoretical_consistency(
                    flow_info, density_info, gen_out, target_samples
                )
                congestion_info['theoretical_consistency'] = validation_results.get('overall_consistency', 0.0)
                congestion_info['validation_results'] = validation_results
            
        except Exception as e:
            print(f"Warning: Congestion computation failed: {e}")
            congestion_info = {
                'spatial_density': torch.zeros(gen_out.shape[0], device=gen_out.device),
                'traffic_flow': torch.zeros_like(gen_out),
                'traffic_intensity': torch.zeros(gen_out.shape[0], device=gen_out.device),
                'congestion_cost': torch.tensor(0.0, device=gen_out.device),
                'sobolev_loss': torch.tensor(0.0, device=gen_out.device),
                'gradient_norm': torch.zeros(gen_out.shape[0], device=gen_out.device),
                'congestion_scale': 1.0,
                'mass_conservation_error': torch.tensor(0.0, device=gen_out.device),
                'theoretical_consistency': 0.0
            }
    
    # Add diversity regularization
    # if gen_out.shape[0] > 1:
    #     try:
    #         gen_cov = torch.cov(gen_out.t())
    #         diversity_loss = -torch.logdet(gen_cov + 1e-6 * torch.eye(gen_out.shape[1], device=gen_out.device))
    #         total_loss += 0.01 * diversity_loss
    #         if congestion_info is not None:
    #             congestion_info['diversity_loss'] = diversity_loss
    #     except:
    #         # Fallback diversity term
    #         gen_std = gen_out.std(dim=0).mean()
    #         diversity_loss = -torch.log(gen_std + 1e-6)
    #         total_loss += 0.01 * diversity_loss
    #         if congestion_info is not None:
    #             congestion_info['diversity_loss'] = diversity_loss
    
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
    track_congestion: bool = True,
    mass_conservation_weight: float = 1.0,
    enforce_mass_conservation_flag: bool = True,
    theoretical_validation: bool = True
) -> Tuple[torch.Tensor, Optional[Dict[str, List[Dict]]]]:
    """
    Multi-marginal OT loss with congestion tracking.
    
    This version includes:
    - Congestion tracking for each evidence domain
    - Multi-domain mass conservation
    - Theoretical validation across domains
    - Adaptive regularization based on domain characteristics
    
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
        mass_conservation_weight (float): Coefficient for mass conservation.
        enforce_mass_conservation_flag (bool): Whether to enforce mass conservation.
        theoretical_validation (bool): Whether to perform theoretical validation.
    
    Returns:
        Tuple[torch.Tensor, Optional[Dict]]: 
            - loss: Total multi-marginal loss
            - multi_congestion_info: Multi-domain congestion information (if track_congestion=True)
    """
    num_domains = len(evidence_list)
    if num_domains == 0:
        raise ValueError("Evidence list must not be empty.")
    
    if weights is None:
        weights = [1.0 / num_domains] * num_domains
    elif len(weights) != num_domains:
        raise ValueError("Weights must match the number of evidence domains.")
    
        # Check data dimensions before processing
    gen_data_dim = generator_outputs.shape[1]
    virtual_data_dim = virtual_targets.shape[1]
    
    if gen_data_dim != virtual_data_dim:
        print(f"Warning: Generator output dim ({gen_data_dim}) != Virtual target dim ({virtual_data_dim})")
        # Adjust virtual targets to match generator output dimension
        if virtual_data_dim > gen_data_dim:
            virtual_targets = virtual_targets[:, :gen_data_dim]
        elif virtual_data_dim < gen_data_dim:
            # Pad with zeros or repeat last column
            padding = torch.zeros(virtual_targets.shape[0], gen_data_dim - virtual_data_dim, 
                                device=virtual_targets.device)
            virtual_targets = torch.cat([virtual_targets, padding], dim=1)
    
    # Check evidence dimensions and fix if needed
    for i, evidence in enumerate(evidence_list):
        if evidence.shape[1] != gen_data_dim:
            print(f"Warning: Evidence {i} dim ({evidence.shape[1]}) != Generator dim ({gen_data_dim})")
            # Adjust evidence to match generator output dimension
            if evidence.shape[1] > gen_data_dim:
                evidence_list[i] = evidence[:, :gen_data_dim]
            elif evidence.shape[1] < gen_data_dim:
                # Pad with zeros
                padding = torch.zeros(evidence.shape[0], gen_data_dim - evidence.shape[1], 
                                    device=evidence.device)
                evidence_list[i] = torch.cat([evidence, padding], dim=1)
    
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=blur)
    
    # Virtual target loss
    loss_virtual = sinkhorn(generator_outputs, virtual_targets)

    # Multi-marginal evidence loss with congestion tracking
    loss_multi = 0.0
    multi_congestion_info = {'domains': []} if track_congestion else None
    
    # For multi-domain mass conservation
    all_target_densities = []
    all_current_densities = []
    
    for i, evidence in enumerate(evidence_list):
        # Standard pairwise OT loss
        pairwise_loss = sinkhorn(generator_outputs, evidence)
        domain_weight = weights[i]
        
        # Add congestion tracking for this domain
        if track_congestion and critics is not None and i < len(critics):
            critic = critics[i]
            
            try:
                # Compute spatial density for this evidence domain
                all_samples = torch.cat([generator_outputs, evidence], dim=0)
                density_info = compute_spatial_density(all_samples, bandwidth=0.25)
                sigma_gen = density_info['density_at_samples'][:generator_outputs.shape[0]]
                
                # Compute traffic flow for this domain
                class DummyGenerator(torch.nn.Module):
                    def __init__(self, fixed_output):
                        super().__init__()
                        self.fixed_output = fixed_output
                    
                    def forward(self, z):
                        return self.fixed_output
                dummy_generator = DummyGenerator(generator_outputs).to(generator_outputs.device)
                                
                flow_info = compute_traffic_flow(
                    critic, dummy_generator, 
                    torch.randn(generator_outputs.shape[0], 2, device=generator_outputs.device), 
                    sigma_gen, lambda_congestion
                )
                
                # Compute congestion cost
                congestion_cost = congestion_cost_function(
                    flow_info['traffic_intensity'], sigma_gen, lambda_congestion
                ).mean()
                
                # Congestion scaling for this domain
                if congestion_cost > 1e-6:
                    h_second = get_congestion_second_derivative(
                        flow_info['traffic_intensity'], sigma_gen, lambda_congestion
                    )
                    domain_congestion_scale = 1.0 + 0.1 * (h_second * flow_info['traffic_intensity']).mean()
                    domain_congestion_scale = max(0.5, min(domain_congestion_scale, 1.5))
                    pairwise_loss = pairwise_loss * domain_congestion_scale
                else:
                    domain_congestion_scale = 1.0

                # Add adaptive Sobolev regularization for this domain
                domain_intensity_mean = flow_info['traffic_intensity'].mean()
                if domain_intensity_mean > 0.1:
                    adaptive_lambda = lambda_sobolev * (0.5 + 0.3 * domain_intensity_mean)
                else:
                    adaptive_lambda = lambda_sobolev
                    
                sobolev_loss = sobolev_regularization(critic, generator_outputs, sigma_gen, adaptive_lambda)
                pairwise_loss += sobolev_loss
                
                # Collect densities for multi-domain mass conservation
                if enforce_mass_conservation_flag:
                    evidence_density_info = compute_spatial_density(evidence, bandwidth=0.25)
                    evidence_density = evidence_density_info['density_at_samples']
                    all_target_densities.append(evidence_density)
                    all_current_densities.append(sigma_gen)
                
                domain_info = {
                    'domain_id': i,
                    'spatial_density': sigma_gen,
                    'traffic_flow': flow_info['traffic_flow'],
                    'traffic_intensity': flow_info['traffic_intensity'],
                    'congestion_cost': congestion_cost,
                    'sobolev_loss': sobolev_loss,
                    'gradient_norm': flow_info['gradient_norm'],
                    'congestion_scale': domain_congestion_scale,
                    'adaptive_lambda_sobolev': adaptive_lambda,
                    'domain_weight': domain_weight
                }
                
                # Theoretical validation for this domain
                if theoretical_validation:
                    validation_results = validate_theoretical_consistency(
                        flow_info, density_info, generator_outputs, evidence
                    )
                    domain_info['theoretical_consistency'] = validation_results.get('overall_consistency', 0.0)
                    domain_info['validation_results'] = validation_results
                
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
                    'gradient_norm': torch.zeros(generator_outputs.shape[0], device=generator_outputs.device),
                    'congestion_scale': 1.0,
                    'theoretical_consistency': 0.0,
                    'domain_weight': domain_weight
                }
                if multi_congestion_info:
                    multi_congestion_info['domains'].append(domain_info)
        
        loss_multi += domain_weight * pairwise_loss
    
    # Multi-domain mass conservation enforcement
    if enforce_mass_conservation_flag and all_target_densities and all_current_densities:
        try:
            # Stack densities directly
            target_size = generator_outputs.shape[0]
            
            # Resize each density tensor to match generator output size
            resized_target_densities = []
            resized_current_densities = []
            
            for target_dens, current_dens in zip(all_target_densities, all_current_densities):
                target_resized, current_resized = safe_density_resize(
                    target_dens, current_dens, target_size
                )
                resized_target_densities.append(target_resized)
                resized_current_densities.append(current_resized)
            
            # Average densities across domains
            avg_target_density = torch.stack(resized_target_densities).mean(dim=0)
            avg_current_density = torch.stack(resized_current_densities).mean(dim=0)
            
            # Enforce multi-domain mass conservation
            multi_domain_conservation = enforce_mass_conservation(
                torch.zeros_like(generator_outputs),  # Dummy flow
                avg_target_density,
                avg_current_density,
                generator_outputs,
                lagrange_multiplier=mass_conservation_weight
            )
            
            # Add multi-domain mass conservation penalty
            mass_penalty = 1.0 * multi_domain_conservation['mass_conservation_error']
            loss_multi += mass_penalty
            
            if multi_congestion_info:
                multi_congestion_info['multi_domain_mass_error'] = multi_domain_conservation['mass_conservation_error'].item()
                multi_congestion_info['multi_domain_mass_penalty'] = mass_penalty.item()
                
        except Exception as e:
            print(f"Warning: Multi-domain mass conservation failed: {e}")
            if multi_congestion_info:
                multi_congestion_info['multi_domain_mass_error'] = 0.0
                multi_congestion_info['multi_domain_mass_penalty'] = 0.0
    
    # Entropy regularization
    data_dim = generator_outputs.shape[1]
    try:
        cov = torch.cov(generator_outputs.t()) + torch.eye(data_dim, device=generator_outputs.device) * 1e-5
        eigenvals = torch.linalg.eigvals(cov).real
        eigenvals = torch.clamp(eigenvals, min=1e-8)
        log_det = torch.sum(torch.log(eigenvals))
        entropy_reg = -lambda_entropy * log_det
        
        # Adaptive entropy regularization based on domain diversity
        if multi_congestion_info and multi_congestion_info['domains']:
            domain_consistency_scores = [
                d.get('theoretical_consistency', 0.0) for d in multi_congestion_info['domains']
            ]
            avg_consistency = sum(domain_consistency_scores) / len(domain_consistency_scores)
            # Less consistent domains need more entropy regularization
            entropy_scale = 1.0 + 0.5 * (1.0 - avg_consistency)
            entropy_reg = entropy_reg * entropy_scale
            
            if multi_congestion_info:
                multi_congestion_info['entropy_scale'] = entropy_scale
                multi_congestion_info['avg_domain_consistency'] = avg_consistency
        
    except:
        # Fallback entropy regularization
        entropy_reg = -lambda_entropy * torch.log(generator_outputs.std(dim=0).mean() + 1e-8)
        if multi_congestion_info:
            multi_congestion_info['entropy_scale'] = 1.0

    total_loss = lambda_virtual * loss_virtual + lambda_multi * loss_multi + entropy_reg
    
    # Store additional global information
    if multi_congestion_info:
        multi_congestion_info['loss_virtual'] = loss_virtual.item()
        multi_congestion_info['loss_multi'] = loss_multi.item()
        multi_congestion_info['entropy_reg'] = entropy_reg.item()
        multi_congestion_info['total_loss'] = total_loss.item()
    
    return total_loss, multi_congestion_info


class CongestionAwareLossFunction:
    """
    Loss function with theoretical integration.
    
    This version includes:
    - Mass conservation enforcement
    - Theoretical validation
    - Adaptive congestion scaling
    - Multi-domain coordination
    """
    
    def __init__(
        self,
        lambda_congestion: float = 1.0,
        lambda_sobolev: float = 0.1,
        lambda_entropy: float = 0.012,
        congestion_cost_type: str = 'quadratic_linear',
        use_adaptive_weights: bool = True,
        mass_conservation_weight: float = 1.0,
        enable_mass_conservation: bool = True,
        enable_theoretical_validation: bool = True
    ):
        self.lambda_congestion = lambda_congestion
        self.lambda_sobolev = lambda_sobolev
        self.lambda_entropy = lambda_entropy
        self.congestion_cost_type = congestion_cost_type
        self.use_adaptive_weights = use_adaptive_weights
        self.mass_conservation_weight = mass_conservation_weight
        self.enable_mass_conservation = enable_mass_conservation
        self.enable_theoretical_validation = enable_theoretical_validation
        
        # Initialize congestion tracker
        self.congestion_tracker = CongestionTracker(lambda_congestion)
        
        # Initialize Sobolev loss
        self.sobolev_loss = SobolevWGANLoss(
            lambda_sobolev=lambda_sobolev,
            use_adaptive_sobolev=use_adaptive_weights
        )
        
        # Track theoretical consistency over time
        self.consistency_history = []
    
    def compute_target_given_loss(
        self,
        generator: torch.nn.Module,
        target_samples: torch.Tensor,
        noise_samples: torch.Tensor,
        critic: Optional[torch.nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute loss for target-given perturbation.
        
        Returns:
            Tuple[loss, gradients, congestion_info]
        """
        return global_w2_loss_and_grad_with_congestion(
            generator, target_samples, noise_samples, critic,
            lambda_congestion=self.lambda_congestion,
            lambda_sobolev=self.lambda_sobolev,
            track_congestion=True,
            mass_conservation_weight=self.mass_conservation_weight,
            enforce_mass_conservation_flag=self.enable_mass_conservation,
            theoretical_validation=self.enable_theoretical_validation
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
        Compute loss for evidence-based perturbation.
        
        Returns:
            Tuple[loss, multi_congestion_info]
        """
        return multi_marginal_ot_loss_with_congestion(
            generator_outputs, evidence_list, virtual_targets, critics,
            weights=weights,
            lambda_congestion=self.lambda_congestion,
            lambda_sobolev=self.lambda_sobolev,
            lambda_entropy=self.lambda_entropy,
            track_congestion=True,
            mass_conservation_weight=self.mass_conservation_weight,
            enforce_mass_conservation_flag=self.enable_mass_conservation,
            theoretical_validation=self.enable_theoretical_validation
        )
    
    def update_congestion_history(self, congestion_info: Dict) -> None:
        """Update congestion tracking history."""
        self.congestion_tracker.update(congestion_info)
        
        # Track theoretical consistency
        if 'theoretical_consistency' in congestion_info:
            self.consistency_history.append(congestion_info['theoretical_consistency'])
            # Keep last 100 entries
            if len(self.consistency_history) > 100:
                self.consistency_history = self.consistency_history[-100:]
    
    def check_congestion_constraints(self, threshold: float = 0.3) -> bool:
        """Congestion constraint checking with theoretical validation."""
        # Check basic congestion increase
        congestion_ok = not self.congestion_tracker.check_congestion_increase(threshold)
        
        # Check theoretical consistency
        theoretical_ok = self.congestion_tracker.check_theoretical_consistency()
        
        return congestion_ok and theoretical_ok
    
    def get_congestion_statistics(self) -> Dict[str, float]:
        """Get statistics including theoretical metrics."""
        stats = {
            'average_congestion': self.congestion_tracker.get_average_congestion(),
            'recent_congestion': self.congestion_tracker.history['congestion_cost'][-1] if self.congestion_tracker.history['congestion_cost'] else 0.0,
            'congestion_trend': self.congestion_tracker.get_average_congestion(window=3) - self.congestion_tracker.get_average_congestion(window=10) if len(self.congestion_tracker.history['congestion_cost']) >= 10 else 0.0
        }
        
        # Add theoretical consistency statistics
        if self.consistency_history:
            stats['avg_theoretical_consistency'] = sum(self.consistency_history) / len(self.consistency_history)
            stats['recent_theoretical_consistency'] = self.consistency_history[-1]
            if len(self.consistency_history) >= 5:
                recent_avg = sum(self.consistency_history[-3:]) / 3
                prev_avg = sum(self.consistency_history[-6:-3]) / 3 if len(self.consistency_history) >= 6 else recent_avg
                stats['consistency_trend'] = recent_avg - prev_avg
        
        # Add mass conservation statistics
        if self.congestion_tracker.history['mass_conservation_error']:
            stats['avg_mass_conservation_error'] = sum(self.congestion_tracker.history['mass_conservation_error']) / len(self.congestion_tracker.history['mass_conservation_error'])
            stats['recent_mass_conservation_error'] = self.congestion_tracker.history['mass_conservation_error'][-1]
        
        return stats
    
    def adapt_parameters(self, statistics: Dict[str, float]) -> None:
        """Adapt loss function parameters based on performance statistics."""
        if not self.use_adaptive_weights:
            return
        
        # Adapt lambda_sobolev based on theoretical consistency
        consistency = statistics.get('recent_theoretical_consistency', 0.7)
        if consistency < 0.5:  # Low consistency needs more regularization
            self.lambda_sobolev = min(self.lambda_sobolev * 1.1, 0.5)
        elif consistency > 0.9:  # High consistency can reduce regularization
            self.lambda_sobolev = max(self.lambda_sobolev * 0.95, 0.01)
        
        # Adapt lambda_congestion based on congestion trends
        congestion_trend = statistics.get('congestion_trend', 0.0)
        if congestion_trend > 0.2:  # Increasing congestion
            self.lambda_congestion = min(self.lambda_congestion * 1.05, 2.0)
        elif congestion_trend < -0.1:  # Decreasing congestion
            self.lambda_congestion = max(self.lambda_congestion * 0.98, 0.1)


def compute_convergence_metrics(
    generator: torch.nn.Module,
    target_or_evidence: torch.Tensor,
    noise_samples: torch.Tensor,
    history_window: int = 10,
    include_theoretical_metrics: bool = True
) -> Dict[str, float]:
    """
    Convergence metrics with theoretical validation.
    
    Args:
        generator (torch.nn.Module): Current generator.
        target_or_evidence (torch.Tensor): Target or evidence samples.
        noise_samples (torch.Tensor): Noise input.
        history_window (int): Window size for trend analysis.
        include_theoretical_metrics (bool): Whether to include theoretical metrics.
    
    Returns:
        Dict[str, float]: Convergence metrics.
    """
    with torch.no_grad():
        gen_samples = generator(noise_samples)
    
    # Compute standard Wasserstein distances
    sinkhorn = SamplesLoss(loss="sinkhorn", p=2, blur=0.05)
    w2_distance = sinkhorn(gen_samples, target_or_evidence).item()
    
    # Compute sample statistics
    gen_mean = gen_samples.mean(dim=0)
    target_mean = target_or_evidence.mean(dim=0)
    mean_shift = torch.norm(gen_mean - target_mean).item()
    
    gen_std = gen_samples.std(dim=0)
    target_std = target_or_evidence.std(dim=0)
    std_ratio = (gen_std / (target_std + 1e-8)).mean().item()
    
    metrics = {
        'w2_distance': w2_distance,
        'mean_shift': mean_shift,
        'std_ratio': std_ratio,
        'convergence_score': 1.0 / (1.0 + w2_distance + mean_shift + abs(std_ratio - 1.0))
    }
    
    # Add theoretical metrics if requested
    if include_theoretical_metrics:
        try:
            # Compute spatial density metrics
            gen_density_info = compute_spatial_density(gen_samples, bandwidth=0.25)
            target_density_info = compute_spatial_density(target_or_evidence, bandwidth=0.25)
            
            gen_density = gen_density_info['density_at_samples']
            target_density = target_density_info['density_at_samples']
            
            # Use density resizing for comparison
            gen_density_resized, target_density_resized = safe_density_resize(
                gen_density, target_density, min(gen_density.shape[0], target_density.shape[0])
            )
            
            # Density distribution similarity
            density_kl_div = F.kl_div(
                torch.log(gen_density_resized + 1e-8), 
                target_density_resized + 1e-8, 
                reduction='mean'
            ).item()
            
            metrics['density_kl_divergence'] = density_kl_div
            metrics['density_similarity'] = 1.0 / (1.0 + density_kl_div)
            
            # Coverage metrics
            distances = torch.cdist(gen_samples, target_or_evidence)
            min_distances = distances.min(dim=1)[0]
            coverage_score = torch.exp(-min_distances.mean()).item()
            
            metrics['coverage_score'] = coverage_score
            
            # Improved convergence score with theoretical components
            metrics['improved_convergence_score'] = (
                0.4 * metrics['convergence_score'] +
                0.3 * metrics['density_similarity'] +
                0.3 * coverage_score
            )
            
        except Exception as e:
            print(f"Warning: Theoretical metrics computation failed: {e}")
            metrics['improved_convergence_score'] = metrics['convergence_score']
    
    return metrics
