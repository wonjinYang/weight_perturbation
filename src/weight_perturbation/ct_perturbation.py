"""
Perturbation classes with full congested transport integration.

This module provides the perturbation classes that integrate the theoretical components: 
spatial density, traffic flow, Sobolev regularization, and congestion tracking.
Enhanced with theoretical validation and mass conservation enforcement.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod

from .models import Generator
from .samplers import virtual_target_sampler
from .utils import parameters_to_vector, vector_to_parameters, load_config
from .congestion import (
    CongestionTracker,
    compute_spatial_density,
    compute_traffic_flow,
    congestion_cost_function,
    get_congestion_second_derivative,
    enforce_mass_conservation,
    validate_theoretical_consistency
)
from .ct_losses import (
    global_w2_loss_and_grad_with_congestion,
    CongestionAwareLossFunction,
)
from .losses import compute_wasserstein_distance, multi_marginal_ot_loss


class CTWeightPerturber(ABC):
    """
    Abstract base class with full congested transport integration.
    
    This class extends the original WeightPerturber with theoretical components:
    - Spatial density estimation σ(x)
    - Traffic flow computation w_Q
    - Sobolev regularization
    - Multi-marginal congestion tracking
    - Mass conservation enforcement
    - Theoretical validation
    """
    
    def __init__(
        self, 
        generator: Generator, 
        config: Optional[Dict] = None,
        enable_congestion_tracking: bool = True,
        use_sobolev_critic: bool = True
    ):
        self.generator = generator
        self.device = next(generator.parameters()).device
        self.enable_congestion_tracking = enable_congestion_tracking
        self.use_sobolev_critic = use_sobolev_critic
        
        # Load config with fallback defaults
        if config is None:
            try:
                self.config = load_config()
            except:
                self.config = self._get_default_config()
        else:
            self.config = config
            
        self.noise_dim = self.config.get('noise_dim', 2)
        self.eval_batch_size = self.config.get('eval_batch_size', 600)
        
        # Initialize congestion tracking
        if self.enable_congestion_tracking:
            self.congestion_tracker = CongestionTracker(
                lambda_param=self.config.get('lambda_congestion', 1.0)
            )
        
        # Initialize loss function with theoretical components
        self.loss_function = CongestionAwareLossFunction(
            lambda_congestion=self.config.get('lambda_congestion', 1.0),
            lambda_sobolev=self.config.get('lambda_sobolev', 0.1),
            lambda_entropy=self.config.get('lambda_entropy', 0.012),
            mass_conservation_weight=self.config.get('mass_conservation_weight', 1.0),
        )
        
        # Initialize common parameters
        self._initialize_common_params()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration with reduced over-conservative parameters."""
        return {
            'noise_dim': 2,
            'eval_batch_size': 600,
            'eta_init': 0.045,             # Increased from 0.045
            'eta_min': 5e-6,              # Slightly higher minimum
            'eta_max': 0.8,               # Higher maximum
            'eta_decay_factor': 0.90,     # Less aggressive
            'eta_boost_factor': 1.05,     # Slightly more aggressive
            'clip_norm': 0.6,             # Increased from 0.4
            'momentum': 0.85,             # Slightly reduced
            'patience': 15,               # Reduced patience
            'rollback_patience': 10,       # Reduced rollback patience
            'improvement_threshold': 5e-6, # More lenient
            'lambda_congestion': 1.0,
            'lambda_sobolev': 0.1,
            'lambda_entropy': 0.012,
            'congestion_threshold': 0.2,  # More lenient
            'mass_conservation_weight': 1.0,  # New parameter
            'theoretical_validation': True,   # New parameter
        }
    
    def _initialize_common_params(self) -> None:
        """Initialize common parameters used across all perturbation methods."""
        self.critic = None  # Will be set if congestion tracking is enabled
    
    def _create_generator_copy(self, data_dim: int) -> Generator:
        """
        Create a copy of the generator with the same architecture.
        
        Args:
            data_dim (int): Output dimension for the new generator.
            
        Returns:
            Generator: A new Generator instance with copied weights.
        """
        # Infer architecture from the original generator
        first_layer = None
        for module in self.generator.modules():
            if isinstance(module, nn.Linear):
                first_layer = module
                break
        
        if first_layer is None:
            raise ValueError("Could not find Linear layers in generator to infer architecture")
        
        hidden_dim = first_layer.out_features
        
        print(f"Creating generator copy: noise_dim={self.noise_dim}, data_dim={data_dim}, hidden_dim={hidden_dim}")
        
        # Create new generator with same architecture
        pert_gen = Generator(
            noise_dim=self.noise_dim,
            data_dim=data_dim,
            hidden_dim=hidden_dim
        ).to(self.device)
        
        # Verify the new generator has parameters
        param_count = sum(p.numel() for p in pert_gen.parameters())
        print(f"New generator parameter count: {param_count}")
        
        if param_count == 0:
            raise RuntimeError(f"Created generator has no parameters. Architecture: noise_dim={self.noise_dim}, data_dim={data_dim}, hidden_dim={hidden_dim}")
        
        # Copy state dict if architectures match
        try:
            pert_gen.load_state_dict(self.generator.state_dict())
            print("Successfully copied weights from original generator")
        except RuntimeError as e:
            print(f"Warning: Could not copy weights due to architecture mismatch: {e}")
            print("Using randomly initialized weights instead.")
        
        return pert_gen
    
    def _compute_theoretical_weight_perturbation(
        self, 
        grads: torch.Tensor, 
        eta: float, 
        congestion_info: Optional[Dict] = None,
        target_samples: Optional[torch.Tensor] = None,
        gen_samples: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Compute theoretically justified weight perturbation using congested transport.
        
        This implements the theoretical relationship:
        ∂W_2^{congested}/∂θ = ∫ ∇_x φ^{congested}(G_θ(z)) · J_θ(z) dz
        
        Args:
            grads (torch.Tensor): Flattened gradients.
            eta (float): Learning rate.
            congestion_info (Optional[Dict]): Congestion metrics.
            target_samples (Optional[torch.Tensor]): Target samples for validation.
            gen_samples (Optional[torch.Tensor]): Generated samples for validation.
        
        Returns:
            Tuple[torch.Tensor, Dict]: (delta_theta, theoretical_info)
        """
        theoretical_info = {}
        
        # Base perturbation
        delta_theta = -eta * grads
        
        # Apply congestion-based theoretical scaling
        if congestion_info is not None:
            try:
                traffic_intensity = congestion_info.get('traffic_intensity')
                spatial_density = congestion_info.get('spatial_density')
                
                if traffic_intensity is not None and spatial_density is not None:
                    # Compute H''(x, i) for theoretical scaling
                    h_second = get_congestion_second_derivative(
                        traffic_intensity, 
                        spatial_density, 
                        lambda_param=self.config.get('lambda_congestion', 1.0)
                    )
                    
                    # Theoretical congestion scaling: 1 / (1 + H''(x,i) * i_Q)
                    # This is derived from the second-order optimality conditions
                    congestion_factor = h_second * traffic_intensity
                    congestion_scale = 1.0 / (1.0 + congestion_factor.mean())
                    
                    # Apply bounded scaling (less conservative than before)
                    congestion_scale = max(0.3, min(congestion_scale, 2.0))
                    delta_theta = delta_theta * congestion_scale
                    
                    theoretical_info['congestion_scale'] = congestion_scale
                    theoretical_info['h_second_mean'] = h_second.mean().item()
                    theoretical_info['congestion_factor_mean'] = congestion_factor.mean().item()
                    
            except Exception as e:
                print(f"Warning: Congestion scaling failed: {e}")
                theoretical_info['congestion_scale'] = 1.0
        
        # Mass conservation adjustment
        if (congestion_info is not None and 
            target_samples is not None and 
            gen_samples is not None and
            self.config.get('mass_conservation_weight', 0.0) > 0):
            
            try:
                # Enforce mass conservation in data space, translate to weight space
                current_density = congestion_info.get('spatial_density')
                if current_density is not None:
                    # Approximate target density
                    target_density_info = compute_spatial_density(target_samples)
                    target_density = target_density_info['density_at_samples']
                    
                    # Mass conservation enforcement
                    mass_conservation = enforce_mass_conservation(
                        congestion_info.get('traffic_flow', torch.zeros_like(gen_samples)),
                        target_density[:gen_samples.shape[0]] if target_density.shape[0] >= gen_samples.shape[0] else target_density,
                        current_density,
                        gen_samples,
                        lagrange_multiplier=self.config.get('mass_conservation_weight', 1.0)
                    )
                    
                    # The mass conservation correction affects the data space,
                    # which influences the weight space through the Jacobian
                    mass_error = mass_conservation['mass_conservation_error'].item()
                    if mass_error > 0.1:  # Apply correction if significant error
                        mass_correction_scale = 1.0 + 0.1 * mass_error
                        delta_theta = delta_theta * mass_correction_scale
                        
                    theoretical_info['mass_conservation_error'] = mass_error
                    theoretical_info['mass_correction_applied'] = mass_error > 0.1
                    
            except Exception as e:
                print(f"Warning: Mass conservation enforcement failed: {e}")
                theoretical_info['mass_conservation_error'] = 0.0
                theoretical_info['mass_correction_applied'] = False
        
        return delta_theta, theoretical_info
    
    def _compute_delta_theta_with_congestion(self, grads: torch.Tensor, eta: float, clip_norm: float,
                                            momentum: float, prev_delta: torch.Tensor,
                                            congestion_info: Optional[Dict] = None,
                                            target_samples: Optional[torch.Tensor] = None,
                                            gen_samples: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Enhanced delta_theta computation with theoretical congestion scaling.
        
        This incorporates the theoretical H''(x,i) second-order information
        and mass conservation requirements.
        
        Args:
            grads (torch.Tensor): Flattened gradients.
            eta (float): Learning rate.
            clip_norm (float): Clipping norm (congestion bound).
            momentum (float): Momentum factor.
            prev_delta (torch.Tensor): Previous delta_theta.
            congestion_info (Optional[Dict]): Congestion metrics for adaptive scaling.
            target_samples (Optional[torch.Tensor]): Target samples for validation.
            gen_samples (Optional[torch.Tensor]): Generated samples for validation.
        
        Returns:
            torch.Tensor: Computed delta_theta.
        """
        # Safety check: ensure grads is a tensor
        if not isinstance(grads, torch.Tensor):
            print(f"Warning: grads is not a tensor (type: {type(grads)}), creating zero tensor")
            if hasattr(self, 'generator') and self.generator is not None:
                total_params = sum(p.numel() for p in self.generator.parameters())
                device = next(self.generator.parameters()).device
            else:
                total_params = 10000  # fallback
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            grads = torch.zeros(total_params, device=device)
        
        # Check for problematic gradients
        if torch.isnan(grads).any() or torch.isinf(grads).any():
            print("Warning: NaN or inf detected in gradients, using zero gradients")
            grads = torch.zeros_like(grads)
        
        # Apply adaptive gradient scaling based on magnitude
        grad_norm = grads.norm()
        if grad_norm > 50.0:  # Increased threshold
            grads = grads * (20.0 / grad_norm)  # Less aggressive scaling
            print(f"Warning: Large gradient norm {grad_norm:.2f}, scaled down")
        
        # Compute theoretical perturbation
        delta_theta, theoretical_info = self._compute_theoretical_weight_perturbation(
            grads, eta, congestion_info, target_samples, gen_samples
        )
        
        # Store theoretical information for tracking
        if hasattr(self, 'congestion_tracker') and self.congestion_tracker:
            self.congestion_tracker.update(theoretical_info)
        
        # Clipping with less conservative parameter-count aware scaling
        norm = delta_theta.norm()
        param_count = len(grads)
        
        # Less conservative parameter scaling
        param_scale = max(1.0, min(param_count / 30000.0, 2.0))  # Increased scaling
        max_norm = clip_norm * param_scale
        
        if norm > max_norm:
            delta_theta = delta_theta * (max_norm / (norm + 1e-8))
        
        # Momentum with reduced stability constraints
        prev_norm = prev_delta.norm()
        if prev_norm > 10.0:  # Increased threshold
            momentum = momentum * 0.5  # Less aggressive reduction
        elif prev_norm > 5.0:
            momentum = momentum * 0.8
        
        # Apply momentum
        delta_theta = momentum * prev_delta + (1 - momentum) * delta_theta
        
        # Final clamping (less restrictive)
        delta_theta = torch.clamp(delta_theta, min=-1.0, max=1.0)  # Increased bounds
        
        # Less aggressive explosion prevention
        final_norm = delta_theta.norm()
        if final_norm > 2.0:  # Increased threshold
            delta_theta = delta_theta * (1.5 / final_norm)  # Less conservative scaling
        
        return delta_theta
    
    def _compute_delta_theta(self, grads: torch.Tensor, eta: float, clip_norm: float,
                             momentum: float, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute weight update delta_theta with momentum and clipping.
        
        This is shared across all perturbation methods as it implements the core
        congested transport mechanism with gradient clipping and momentum.
        
        Args:
            grads (torch.Tensor): Flattened gradients.
            eta (float): Learning rate.
            clip_norm (float): Clipping norm (congestion bound).
            momentum (float): Momentum factor.
            prev_delta (torch.Tensor): Previous delta_theta.
        
        Returns:
            torch.Tensor: Computed delta_theta.
        """
        # Call without congestion info for backward compatibility
        return self._compute_delta_theta_with_congestion(
            grads, eta, clip_norm, momentum, prev_delta, congestion_info=None
        )
    
    def _apply_parameter_update(self, pert_gen: Generator, theta_prev: torch.Tensor, 
                               delta_theta: torch.Tensor) -> torch.Tensor:
        """
        Apply parameter update to the generator with stability checks.
        
        Args:
            pert_gen (Generator): Generator to update.
            theta_prev (torch.Tensor): Previous parameter vector.
            delta_theta (torch.Tensor): Parameter update.
            
        Returns:
            torch.Tensor: New parameter vector.
        """
        # NaN/inf checking
        if torch.isnan(delta_theta).any() or torch.isinf(delta_theta).any():
            print("Warning: NaN or inf detected in delta_theta, using zero update")
            delta_theta = torch.zeros_like(delta_theta)
        
        # Check for excessively large updates (less restrictive)
        delta_norm = delta_theta.norm()
        if delta_norm > 5.0:  # Increased threshold
            print(f"Warning: Very large delta_theta norm {delta_norm:.2f}, scaling down")
            delta_theta = delta_theta * (2.0 / delta_norm)  # Less aggressive scaling
        
        theta_new = theta_prev + delta_theta
        
        # Parameter checking
        if torch.isnan(theta_new).any() or torch.isinf(theta_new).any():
            print("Warning: NaN or inf detected in new parameters, keeping old parameters")
            theta_new = theta_prev.clone()
        
        # Check for parameter explosion (less restrictive)
        param_norm = theta_new.norm()
        if param_norm > 500.0:  # Increased threshold
            print(f"Warning: Parameter explosion detected (norm: {param_norm:.2f}), reverting")
            theta_new = theta_prev.clone()
        
        try:
            vector_to_parameters(theta_new, pert_gen.parameters())
        except Exception as e:
            print(f"Warning: Parameter update failed: {e}, reverting")
            vector_to_parameters(theta_prev, pert_gen.parameters())
            theta_new = theta_prev.clone()
        
        return theta_new
    
    def _adapt_learning_rate(self, eta: float, improvement: float, step: int, 
                           no_improvement_count: int, loss_history: List[float]) -> Tuple[float, int]:
        """
        Adaptive learning rate with less conservative adjustments.
        
        Args:
            eta (float): Current learning rate.
            improvement (float): Current improvement value.
            step (int): Current step/epoch.
            no_improvement_count (int): Consecutive steps without improvement.
            loss_history (List[float]): Recent loss history for trend analysis.
            
        Returns:
            Tuple[float, int]: (new_eta, updated_no_improvement_count)
        """
        eta_min = self.config.get('eta_min', 5e-6) 
        eta_max = self.config.get('eta_max', 0.8)
        eta_decay_factor = self.config.get('eta_decay_factor', 0.92)
        eta_boost_factor = self.config.get('eta_boost_factor', 1.08)
        improvement_threshold = self.config.get('improvement_threshold', 5e-6)
        
        # Adaptive threshold based on loss magnitude (less conservative)
        if loss_history and loss_history[-1] > 100.0:
            improvement_threshold *= 20  # Less lenient than before
        elif loss_history and loss_history[-1] > 10.0:
            improvement_threshold *= 5   # Less lenient than before
        
        # Trend analysis
        is_stagnating = False
        is_oscillating = False
        if len(loss_history) >= 6:
            recent_losses = loss_history[-6:]
            # Check for stagnation
            recent_var = np.var(recent_losses)
            is_stagnating = recent_var < improvement_threshold / 50  # Less sensitive
            
            # Check for oscillation
            diffs = np.diff(recent_losses)
            if len(diffs) >= 4:
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                is_oscillating = sign_changes >= 4  # More tolerant
        
        # Learning rate adaptation logic (less conservative)
        if improvement > improvement_threshold:
            # Good improvement: boost learning rate
            new_eta = min(eta * eta_boost_factor, eta_max)
            new_no_improvement_count = 0
            
        elif improvement < -improvement_threshold * 5:  # Less sensitive
            # Significant degradation: decay learning rate
            decay_factor = eta_decay_factor if step > 3 else 0.96  # Less aggressive early on
            new_eta = max(eta * decay_factor, eta_min)
            new_no_improvement_count = no_improvement_count + 1
            
        elif is_oscillating:
            # Oscillation detected: reduce learning rate to stabilize
            new_eta = max(eta * 0.95, eta_min)  # Less aggressive
            new_no_improvement_count = no_improvement_count + 1
            if step % 10 == 0:  # Less frequent logging
                print(f"Oscillation detected, reducing learning rate to {new_eta:.6f}")
                
        elif is_stagnating:
            # Stagnation: small reduction
            new_eta = max(eta * 0.995, eta_min)  # Very minimal decay
            new_no_improvement_count = no_improvement_count + 1
            
        else:
            # Marginal change: maintain learning rate
            new_eta = eta
            new_no_improvement_count = no_improvement_count + 1
        
        # Prevent learning rate from being reduced too aggressively early on
        if step < 10 and new_eta < eta * 0.7:  # Less restrictive
            new_eta = eta * 0.9
            
        # Safety bounds
        new_eta = max(eta_min, min(new_eta, eta_max))
        
        return new_eta, new_no_improvement_count
    
    def _check_rollback_condition_with_congestion(self, loss_hist: List[float], 
                                                 no_improvement_count: int) -> bool:
        """
        Enhanced rollback condition checking with theoretical consistency.
        
        Args:
            loss_hist (List[float]): History of loss values.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            bool: True if rollback should be triggered.
        """
        # Check standard rollback conditions with higher thresholds
        should_rollback = self._check_rollback_condition(loss_hist, no_improvement_count)
        
        # Additional theoretical consistency checks
        if self.enable_congestion_tracking and self.congestion_tracker:
            # Check congestion increase
            congestion_threshold = self.config.get('congestion_threshold', 0.2)
            if self.congestion_tracker.check_congestion_increase(congestion_threshold):
                print("Rollback triggered due to excessive congestion increase")
                return True
            
            # Check theoretical consistency
            if not self.congestion_tracker.check_theoretical_consistency():
                print("Rollback triggered due to theoretical inconsistency")
                return True
                
        return should_rollback
    
    def _check_rollback_condition(self, loss_hist: List[float], 
                                 no_improvement_count: int) -> bool:
        """
        Rollback condition checking with less conservative triggers.
        
        Args:
            loss_hist (List[float]): History of loss values.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            bool: True if rollback should be triggered.
        """
        rollback_patience = self.config.get('rollback_patience', 5)
        
        # Trigger rollback if no improvement for many consecutive steps
        if no_improvement_count >= rollback_patience * 3:  # More patient
            return True
            
        # Check for consistent severe loss increase (less sensitive)
        if len(loss_hist) >= rollback_patience + 5:
            recent_losses = loss_hist[-(rollback_patience + 5):]
            # Check if loss has been consistently increasing
            increasing_count = sum(1 for i in range(1, len(recent_losses)) 
                                 if recent_losses[i] >= recent_losses[i-1])
            if increasing_count >= len(recent_losses) * 0.8:  # 80% must be increasing
                return True
        
        # Check for sudden severe spike in loss (less sensitive)
        if len(loss_hist) >= 6:
            recent_avg = sum(loss_hist[-3:]) / 3
            prev_avg = sum(loss_hist[-6:-3]) / 3
            if recent_avg > prev_avg * 10.0:  # 10x increase
                return True
        
        # Additional check for loss explosion (less sensitive)
        if len(loss_hist) >= 3:
            if loss_hist[-1] > 5000.0 and loss_hist[-1] > loss_hist[-3] * 20:
                print("Loss explosion detected, triggering rollback")
                return True
                
        return False
    
    def _update_best_state(self, current_loss: float, pert_gen: Generator,
                          best_loss: float, best_vec: Optional[torch.Tensor]) -> Tuple[float, torch.Tensor]:
        """
        Update best state if current loss is better.
        
        Args:
            current_loss (float): Current loss value.
            pert_gen (Generator): Current generator.
            best_loss (float): Best loss so far.
            best_vec (Optional[torch.Tensor]): Best parameter vector so far.
            
        Returns:
            Tuple[float, torch.Tensor]: Updated (best_loss, best_vec).
        """
        # Less restrictive improvement requirement
        improvement_required = max(0.0005, best_loss * 0.00005)  # Reduced requirement
        
        if current_loss < best_loss - improvement_required:
            best_loss = current_loss
            best_vec = parameters_to_vector(pert_gen.parameters()).clone()
        elif best_vec is None:  # Always store initial state
            best_loss = current_loss
            best_vec = parameters_to_vector(pert_gen.parameters()).clone()
            
        return best_loss, best_vec
    
    def _restore_best_state(self, pert_gen: Generator, best_vec: Optional[torch.Tensor]) -> None:
        """
        Restore the best parameter state to the generator.
        
        Args:
            pert_gen (Generator): Generator to restore.
            best_vec (Optional[torch.Tensor]): Best parameter vector.
        """
        if best_vec is not None:
            try:
                vector_to_parameters(best_vec, pert_gen.parameters())
                print("Successfully restored best state")
            except Exception as e:
                print(f"Warning: Failed to restore best state: {e}")
    
    def _validate_theoretical_step(
        self, 
        pert_gen: Generator, 
        target_samples: torch.Tensor,
        congestion_info: Optional[Dict] = None
    ) -> Dict[str, float]:
        """
        Validate theoretical consistency of the perturbation step.
        
        Args:
            pert_gen (Generator): Current generator.
            target_samples (torch.Tensor): Target samples.
            congestion_info (Optional[Dict]): Congestion information.
        
        Returns:
            Dict[str, float]: Validation metrics.
        """
        if not self.config.get('theoretical_validation', True):
            return {}
        
        try:
            # Generate samples for validation
            noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
            with torch.no_grad():
                gen_samples = pert_gen(noise)
            
            # Compute density information for validation
            density_info = compute_spatial_density(gen_samples)
            
            # Use provided congestion info or compute fresh
            if congestion_info is None and self.critic is not None:
                congestion_info = compute_traffic_flow(
                    self.critic, pert_gen, noise, 
                    density_info['density_at_samples'],
                    self.config.get('lambda_congestion', 1.0)
                )
            
            if congestion_info is not None:
                # Validate theoretical consistency
                validation_results = validate_theoretical_consistency(
                    congestion_info, density_info, gen_samples, target_samples
                )
                
                # Update congestion tracker with validation results
                if hasattr(self, 'congestion_tracker') and self.congestion_tracker:
                    validation_info = {
                        'theoretical_consistency': validation_results.get('overall_consistency', 0.0),
                        'mass_conservation_error': validation_results.get('coverage_error', 0.0)
                    }
                    self.congestion_tracker.update(validation_info)
                
                return validation_results
            
        except Exception as e:
            print(f"Warning: Theoretical validation failed: {e}")
        
        return {}
    
    @abstractmethod
    def _compute_loss_and_grad(self, pert_gen: Generator, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute loss and gradients specific to the perturbation method.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, gradients)
        """
        pass
    
    @abstractmethod
    def _validate_and_adapt(self, pert_gen: Generator, *args, **kwargs) -> Tuple[float, float]:
        """
        Validate perturbation and adapt learning rate.
        
        Returns:
            Tuple[float, float]: (current_loss, improvement)
        """
        pass
    
    @abstractmethod
    def perturb(self, *args, **kwargs) -> Generator:
        """
        Perform the perturbation process.
        
        Returns:
            Generator: Perturbed generator model.
        """
        pass


class CTWeightPerturberTargetGiven(CTWeightPerturber):
    """
    Enhanced Weight Perturber for target-given perturbation with full theoretical integration.
    
    This class now incorporates:
    - Enhanced congestion tracking with mass conservation
    - Theoretical validation at each step
    - Reduced over-conservative constraints
    - Improved theoretical justification for weight space perturbation
    
    Args:
        generator (Generator): Pre-trained generator model.
        target_samples (torch.Tensor): Samples from the target distribution.
        config (Optional[Dict]): Configuration dictionary. If None, loads from default.yaml.
        critic (Optional[nn.Module]): Pre-trained critic for traffic flow computation.
        enable_congestion_tracking (bool): Whether to enable congestion tracking.
    """
    
    def __init__(self, generator: Generator, target_samples: torch.Tensor, 
                 config: Optional[Dict] = None, critic: Optional[nn.Module] = None,
                 enable_congestion_tracking: bool = True):
        super().__init__(generator, config, enable_congestion_tracking)
        self.target_samples = target_samples.to(self.device)
        self.critic = critic
        
        # Validate target samples
        if self.target_samples.numel() == 0:
            raise ValueError("Target samples must not be empty.")
    
    def _compute_loss_and_grad_with_congestion(self, pert_gen: Generator) -> Tuple[torch.Tensor, torch.Tensor, Dict]:
        """
        Compute loss and gradients with full congestion tracking and validation.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor, Dict]: (loss, grads, congestion_info)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        
        # Use loss function with congestion tracking
        loss, grads, congestion_info = global_w2_loss_and_grad_with_congestion(
            pert_gen, 
            self.target_samples, 
            noise,
            critic=self.critic,
            lambda_congestion=self.config.get('lambda_congestion', 1.0),
            lambda_sobolev=self.config.get('lambda_sobolev', 0.1),
            track_congestion=True,
            use_direct_w2=True,
            w2_weight=1.0,
            map_weight=0.5,
            mass_conservation_weight=1.0,
        )
        return loss, grads, congestion_info
    
    def _compute_loss_and_grad(self, pert_gen: Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute improved global W2 loss and flattened gradients.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        
        # Use improved loss function
        loss, grads, _ = global_w2_loss_and_grad_with_congestion(
            pert_gen, 
            self.target_samples, 
            noise,
            critic=self.critic,
            track_congestion=False,
            use_direct_w2=True,
            w2_weight=1.0,
            map_weight=0.5,
            mass_conservation_weight=1.0,
        )
        return loss, grads
    
    def _validate_and_adapt(self, pert_gen: Generator, eta: float, w2_hist: List[float],
                            patience: int, verbose: bool, step: int) -> Tuple[float, float]:
        """
        Validate perturbation and compute improvement.
        
        Args:
            pert_gen (Generator): Current perturbed generator.
            eta (float): Current learning rate.
            w2_hist (List[float]): History of W2 distances.
            patience (int): Patience for adaptation.
            verbose (bool): Print flag.
            step (int): Current step.
        
        Returns:
            Tuple[float, float]: (w2_pert, improvement)
        """
        noise_eval = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        
        with torch.no_grad():
            orig_out = self.generator(noise_eval)
            pert_out = pert_gen(noise_eval)
        
        # Use reasonable blur for stable evaluation
        try:
            w2_orig = compute_wasserstein_distance(orig_out, self.target_samples, p=2, blur=0.1)
            w2_pert = compute_wasserstein_distance(pert_out, self.target_samples, p=2, blur=0.1)
        except:
            # Fallback: simple MSE distance
            w2_orig = ((orig_out.unsqueeze(1) - self.target_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
            w2_pert = ((pert_out.unsqueeze(1) - self.target_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
        
        # Compute improvement compared to original
        improvement = w2_orig.item() - w2_pert.item()
        w2_hist.append(w2_pert.item())
        
        return w2_pert.item(), improvement
    
    def perturb(self, steps: int = 50, eta_init: float = 0.08, clip_norm: float = 0.6,
                momentum: float = 0.88, patience: int = 10, verbose: bool = True) -> Generator:
        """
        Perform the enhanced perturbation process with theoretical validation.
        
        Args:
            steps (int): Number of perturbation steps. Defaults to 50.
            eta_init (float): Initial learning rate. Defaults to 0.08.
            clip_norm (float): Gradient clipping norm. Defaults to 0.6.
            momentum (float): Momentum factor. Defaults to 0.88.
            patience (int): Maximum patience. Defaults to 10.
            verbose (bool): Print progress. Defaults to True.
        
        Returns:
            Generator: Perturbed generator model.
        """
        try:
            data_dim = self.target_samples.shape[1]
            pert_gen = self._create_generator_copy(data_dim)
            
            # Initialize perturbation state
            theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
            
            if theta_prev.numel() == 0:
                raise RuntimeError("Generated generator has no parameters.")
            
            delta_theta_prev = torch.zeros_like(theta_prev)
            eta = eta_init
            w2_hist = []
            loss_hist = []
            best_vec = None
            best_w2 = float('inf')
            no_improvement_count = 0
            consecutive_rollbacks = 0
            
            for step in range(steps):
                try:
                    # Generate samples for theoretical validation
                    noise_samples = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
                    with torch.no_grad():
                        gen_samples = pert_gen(noise_samples)
                    
                    # Compute loss and gradients with congestion tracking
                    if self.enable_congestion_tracking and self.critic is not None:
                        loss, grads, congestion_info = self._compute_loss_and_grad_with_congestion(pert_gen)
                        
                        # Update congestion tracker
                        if congestion_info:
                            self.congestion_tracker.update(congestion_info)
                        
                        # Compute delta_theta with full theoretical justification
                        delta_theta = self._compute_delta_theta_with_congestion(
                            grads, eta, clip_norm, momentum, delta_theta_prev, 
                            congestion_info, self.target_samples, gen_samples
                        )
                        
                        # Validate theoretical consistency
                        validation_results = self._validate_theoretical_step(
                            pert_gen, self.target_samples, congestion_info
                        )
                        
                    else:
                        loss, grads = self._compute_loss_and_grad(pert_gen)
                        delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                        validation_results = {}
                    
                    # Apply update
                    theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                    delta_theta_prev = delta_theta.clone()
                    
                    # Validate and get improvement
                    w2_pert, improvement = self._validate_and_adapt(pert_gen, eta, w2_hist, patience, verbose, step)
                    
                    # Track loss history
                    loss_hist.append(loss.item())
                    
                    # Update best state
                    best_w2, best_vec = self._update_best_state(w2_pert, pert_gen, best_w2, best_vec)
                    
                    # Adapt learning rate
                    eta, no_improvement_count = self._adapt_learning_rate(eta, improvement, step, no_improvement_count, loss_hist)
                    
                    # Check for rollback condition with theoretical validation
                    if self._check_rollback_condition_with_congestion(w2_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at step {step}")
                        
                        self._restore_best_state(pert_gen, best_vec)
                        # Reset parameters after rollback
                        eta = max(eta * 0.95, self.config.get('eta_min', 5e-6))
                        no_improvement_count = 0
                        consecutive_rollbacks += 1
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                        
                        # If too many rollbacks, break early
                        if consecutive_rollbacks >= 3:
                            if verbose:
                                print(f"Too many rollbacks, stopping early")
                            break
                    else:
                        consecutive_rollbacks = 0
                    
                    if verbose:
                        log_msg = f"[{step:2d}] W2(Pert, Target)={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                        if self.enable_congestion_tracking and 'congestion_info' in locals() and congestion_info:
                            log_msg += f" Congestion={congestion_info.get('congestion_cost', torch.tensor(0)).item():.2f}"
                        if validation_results:
                            log_msg += f" Consistency={validation_results.get('overall_consistency', 0.0):.2f}"
                        print(log_msg)
                    
                    # Early stopping conditions
                    if no_improvement_count >= patience:
                        if verbose:
                            print(f"Early stopping at step {step}")
                        break
                    
                    # Very good convergence
                    if w2_pert < 1e-3:
                        if verbose:
                            print(f"Excellent convergence at step {step}")
                        break
                
                except Exception as e:
                    print(f"Error in step {step}: {e}")
                    if step == 0:
                        raise
                    break
            
            # Final restore to best state
            self._restore_best_state(pert_gen, best_vec)
            
            return pert_gen
            
        except Exception as e:
            print(f"Error in perturbation: {e}")
            # Return a copy as fallback
            data_dim = self.target_samples.shape[1] if hasattr(self, 'target_samples') else 2
            return self._create_generator_copy(data_dim)


class CTWeightPerturberTargetNotGiven(CTWeightPerturber):
    """
    Enhanced Weight Perturber for evidence-based perturbation with theoretical improvements.
    
    This class now incorporates enhanced theoretical validation and reduced conservatism.
    
    Args:
        generator (Generator): Pre-trained generator model.
        evidence_list (List[torch.Tensor]): List of evidence domain samples.
        centers (List[np.ndarray]): Centers of evidence domains.
        config (Optional[Dict]): Configuration dictionary. If None, loads from default.yaml.
        critics (Optional[List[nn.Module]]): Pre-trained critics for each evidence domain.
        enable_congestion_tracking (bool): Whether to enable congestion tracking.
    """
    
    def __init__(self, generator: Generator, evidence_list: List[torch.Tensor],
                 centers: List[np.ndarray], config: Optional[Dict] = None,
                 critics: Optional[List[nn.Module]] = None,
                 enable_congestion_tracking: bool = True):
        super().__init__(generator, config, enable_congestion_tracking)
        self.evidence_list = [ev.to(self.device) for ev in evidence_list]
        self.centers = centers
        self.critics = critics if critics is not None else []
        
        # Less conservative eval_batch_size
        self.eval_batch_size = self.config.get('eval_batch_size', 600)
        
        if len(self.evidence_list) == 0:
            raise ValueError("Evidence list must not be empty.")
        
        # Initialize multi-marginal congestion trackers if enabled
        if self.enable_congestion_tracking:
            self.multi_congestion_trackers = [
                CongestionTracker(lambda_param=self.config.get('lambda_congestion', 1.0))
                for _ in self.evidence_list
            ]
    
    def _estimate_virtual_target_with_congestion(
        self, evidence_list: List[torch.Tensor], epoch: int
    ) -> torch.Tensor:
        """
        Enhanced virtual target estimation with theoretical congestion awareness.
        
        Args:
            evidence_list (List[torch.Tensor]): Evidence domains.
            epoch (int): Current epoch for adaptation.
            
        Returns:
            torch.Tensor: Virtual target samples.
        """
        # Less conservative bandwidth adaptation
        base_bandwidth = 0.25  # Reduced base bandwidth for better adaptation
        
        if self.enable_congestion_tracking and self.multi_congestion_trackers:
            congestion_values = [
                tracker.get_average_congestion() 
                for tracker in self.multi_congestion_trackers
                if len(tracker.history['congestion_cost']) > 0
            ]
            if congestion_values:
                avg_congestion = np.mean(congestion_values)
                # More responsive bandwidth adjustment
                bandwidth = base_bandwidth * (1.0 + 0.15 * avg_congestion)
            else:
                bandwidth = base_bandwidth
        else:
            bandwidth = base_bandwidth
        
        # Less conservative time-based annealing
        bandwidth += 0.05 * torch.exp(torch.tensor(-epoch / 15.0)).item()
        bandwidth = max(bandwidth, 0.15)  # Lower minimum for better adaptation
        bandwidth = min(bandwidth, 0.5)   # Reasonable maximum
        
        try:
            virtuals = virtual_target_sampler(
                evidence_list, 
                bandwidth=bandwidth, 
                num_samples=self.eval_batch_size, 
                device=self.device
            )
            return virtuals
        except Exception as e:
            print(f"Warning: Virtual target sampling failed: {e}, using fallback")
            # Fallback: simple concatenation and subsampling
            all_evidence = torch.cat(evidence_list, dim=0)
            if len(all_evidence) >= self.eval_batch_size:
                indices = torch.randperm(len(all_evidence))[:self.eval_batch_size]
                return all_evidence[indices]
            else:
                # Repeat samples if not enough
                repeat_factor = (self.eval_batch_size // len(all_evidence)) + 1
                repeated = all_evidence.repeat(repeat_factor, 1)
                return repeated[:self.eval_batch_size]
    
    def _compute_multi_marginal_congestion(self, pert_gen: Generator, 
                                          noise_samples: torch.Tensor) -> Dict[str, List[Dict]]:
        """
        Enhanced multi-marginal congestion computation with better theoretical integration.
        
        Args:
            pert_gen (Generator): Current generator.
            noise_samples (torch.Tensor): Noise input.
            
        Returns:
            Dict containing multi-marginal congestion information.
        """
        multi_congestion_info = {'domains': []}
        
        with torch.no_grad():
            gen_samples = pert_gen(noise_samples)
        
        for i, (evidence, critic) in enumerate(zip(self.evidence_list, self.critics)):
            if critic is None:
                continue
                
            try:
                # Compute spatial density for this evidence domain
                all_samples = torch.cat([gen_samples, evidence], dim=0)
                density_info = compute_spatial_density(all_samples, bandwidth=0.2)
                sigma_gen = density_info['density_at_samples'][:gen_samples.shape[0]]
                
                # Compute traffic flow for this domain
                flow_info = compute_traffic_flow(
                    critic, pert_gen, noise_samples, sigma_gen,
                    lambda_param=self.config.get('lambda_congestion', 1.0)
                )
                
                # Compute congestion cost with less restrictive bounds
                raw_congestion_cost = congestion_cost_function(
                    flow_info['traffic_intensity'], sigma_gen,
                    lambda_param=self.config.get('lambda_congestion', 1.0)
                )
                
                congestion_cost = raw_congestion_cost.mean()
                
                # Only handle extreme numerical issues
                if torch.isnan(congestion_cost) or torch.isinf(congestion_cost):
                    congestion_cost = torch.zeros(1, device=congestion_cost.device)[0]
                elif congestion_cost < 0:
                    congestion_cost = torch.abs(congestion_cost)
                    
                domain_info = {
                    'domain_id': i,
                    'spatial_density': sigma_gen,
                    'traffic_flow': flow_info['traffic_flow'],
                    'traffic_intensity': flow_info['traffic_intensity'],
                    'congestion_cost': congestion_cost,
                    'gradient_norm': flow_info['gradient_norm']
                }
                
                multi_congestion_info['domains'].append(domain_info)
                
                # Update domain-specific tracker
                if self.enable_congestion_tracking and i < len(self.multi_congestion_trackers):
                    self.multi_congestion_trackers[i].update({
                        'traffic_intensity': flow_info['traffic_intensity'],
                        'congestion_cost': congestion_cost,
                        'spatial_density': sigma_gen
                    })
            
            except Exception as e:
                print(f"Warning: Congestion computation failed for domain {i}: {e}")
                # Create dummy domain info
                domain_info = {
                    'domain_id': i,
                    'spatial_density': torch.zeros(gen_samples.shape[0], device=gen_samples.device),
                    'traffic_flow': torch.zeros_like(gen_samples),
                    'traffic_intensity': torch.zeros(gen_samples.shape[0], device=gen_samples.device),
                    'congestion_cost': torch.tensor(0.0, device=gen_samples.device),
                    'gradient_norm': torch.zeros(gen_samples.shape[0], device=gen_samples.device)
                }
                if multi_congestion_info:
                    multi_congestion_info['domains'].append(domain_info)
        
        return multi_congestion_info
    
    def _average_multi_marginal_congestion(self, multi_congestion_info: Dict) -> Dict:
        """
        Average congestion information across domains with improved stability.
        
        Args:
            multi_congestion_info (Dict): Multi-domain congestion info.
            
        Returns:
            Dict: Averaged congestion information.
        """
        if 'domains' not in multi_congestion_info or not multi_congestion_info['domains']:
            return {}
        
        domains = multi_congestion_info['domains']
        avg_info = {}
        
        try:
            # Average traffic intensity with mean (more responsive than median)
            all_intensities = [d['traffic_intensity'] for d in domains]
            if all_intensities:
                stacked_intensities = torch.stack(all_intensities)
                avg_info['traffic_intensity'] = torch.mean(stacked_intensities, dim=0)
            
            # Average spatial density with mean
            all_densities = [d['spatial_density'] for d in domains]
            if all_densities:
                stacked_densities = torch.stack(all_densities)
                avg_info['spatial_density'] = torch.mean(stacked_densities, dim=0)
            
            # Average congestion cost with mean
            all_costs = [d['congestion_cost'] for d in domains]
            if all_costs:
                cost_tensor = torch.stack(all_costs)
                avg_info['congestion_cost'] = torch.mean(cost_tensor)
                
        except Exception as e:
            print(f"Warning: Failed to average congestion info: {e}")
            return {}
        
        return avg_info
    
    def _compute_loss_and_grad(self, pert_gen: Generator, virtual_samples: torch.Tensor,
                               lambda_entropy: float, lambda_virtual: float, lambda_multi: float, mass_conservation_weight: float
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-marginal OT loss computation with improved stability and bounds.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        gen_out = pert_gen(noise)
        
        try:
            loss = multi_marginal_ot_loss(
                gen_out, self.evidence_list, virtual_samples,
                blur=0.15,  # Reasonable blur for stability
                lambda_virtual=min(lambda_virtual, 1.0),    # Less restrictive cap
                lambda_multi=min(lambda_multi, 1.0),       # Less restrictive cap
                lambda_entropy=min(lambda_entropy, 0.02),   # Less restrictive cap
                mass_conservation_weight=mass_conservation_weight
            )
            
            # Less aggressive clamping
            loss = torch.clamp(loss, max=5000.0)
            
        except Exception as e:
            print(f"Warning: OT loss computation failed: {e}")
            # Fallback: weighted MSE loss to virtual targets
            try:
                # Compute distances to all evidence domains
                evidence_losses = []
                for evidence in self.evidence_list:
                    # Distance to each evidence domain
                    evidence_dist = ((gen_out.unsqueeze(1) - evidence.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
                    evidence_losses.append(evidence_dist)
                
                # Distance to virtual targets
                virtual_dist = ((gen_out.unsqueeze(1) - virtual_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
                
                # Combine losses
                evidence_loss = torch.stack(evidence_losses).mean()
                loss = lambda_virtual * virtual_dist + lambda_multi * evidence_loss
            except:
                # Final fallback
                loss = ((gen_out.unsqueeze(1) - virtual_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
        
        pert_gen.zero_grad()
        try:
            loss.backward()
            grads = torch.cat([p.grad.view(-1) for p in pert_gen.parameters() if p.grad is not None])
            if grads.numel() == 0:
                total_params = sum(p.numel() for p in pert_gen.parameters())
                device = next(pert_gen.parameters()).device
                grads = torch.zeros(total_params, device=device)
        except Exception as e:
            print(f"Warning: Gradient computation failed: {e}")
            total_params = sum(p.numel() for p in pert_gen.parameters())
            device = next(pert_gen.parameters()).device
            grads = torch.zeros(total_params, device=device)
            
        return loss, grads
    
    def _validate_and_adapt(self, pert_gen: Generator, virtual_samples: torch.Tensor, eta: float,
                            ot_hist: List[float], patience: int, verbose: bool, epoch: int) -> Tuple[float, float]:
        """
        Validation with improved stability and responsiveness.
        
        Args:
            pert_gen (Generator): Current perturbed generator.
            virtual_samples (torch.Tensor): Current virtual target samples.
            eta (float): Current learning rate.
            ot_hist (List[float]): History of OT losses.
            patience (int): Patience for adaptation.
            verbose (bool): Print flag.
            epoch (int): Current epoch.
        
        Returns:
            Tuple[float, float]: (ot_pert, improvement)
        """
        noise_eval = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        
        with torch.no_grad():
            orig_out = self.generator(noise_eval)
            pert_out = pert_gen(noise_eval)

        try:
            ot_orig = multi_marginal_ot_loss(
                orig_out, self.evidence_list, virtual_samples,
                blur=0.15,
                lambda_virtual=self.config.get('lambda_virtual', 0.8),
                lambda_multi=self.config.get('lambda_multi', 1.0),
                lambda_entropy=self.config.get('lambda_entropy', 0.012),
                mass_conservation_weight=self.config.get('mass_conservation_weight', 1.0),
            ).item()

            ot_pert = multi_marginal_ot_loss(
                pert_out, self.evidence_list, virtual_samples,
                blur=0.15,
                lambda_virtual=self.config.get('lambda_virtual', 0.8),
                lambda_multi=self.config.get('lambda_multi', 1.0),
                lambda_entropy=self.config.get('lambda_entropy', 0.012),
                mass_conservation_weight=self.config.get('mass_conservation_weight', 1.0),
            ).item()
            
        except Exception as e:
            print(f"Warning: OT validation failed: {e}")
            # Fallback with evidence domain consideration
            try:
                # Compute distances to evidence domains
                orig_evidence_dists = []
                pert_evidence_dists = []
                
                for evidence in self.evidence_list:
                    orig_dist = ((orig_out.unsqueeze(1) - evidence.unsqueeze(0))**2).sum(-1).min(1)[0].mean().item()
                    pert_dist = ((pert_out.unsqueeze(1) - evidence.unsqueeze(0))**2).sum(-1).min(1)[0].mean().item()
                    orig_evidence_dists.append(orig_dist)
                    pert_evidence_dists.append(pert_dist)
                
                ot_orig = np.mean(orig_evidence_dists)
                ot_pert = np.mean(pert_evidence_dists)
                
            except:
                # Final simple fallback
                ot_orig = ((orig_out.unsqueeze(1) - virtual_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean().item()
                ot_pert = ((pert_out.unsqueeze(1) - virtual_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean().item()
        
        # Compute improvement compared to original
        improvement = ot_orig - ot_pert
        ot_hist.append(ot_pert)
        
        return ot_pert, improvement
    
    def perturb(self, epochs: int = 50, eta_init: float = 0.08, clip_norm: float = 0.6,
                momentum: float = 0.88, patience: int = 10, lambda_entropy: float = 0.012,
                lambda_virtual: float = 0.8, lambda_multi: float = 1.0, verbose: bool = True) -> Generator:
        """
        Perform the enhanced perturbation process with improved theoretical integration.
        
        Args:
            epochs (int): Number of epochs. Defaults to 50.
            eta_init (float): Initial learning rate. Defaults to 0.08.
            clip_norm (float): Gradient clipping norm. Defaults to 0.6.
            momentum (float): Momentum factor. Defaults to 0.88.
            patience (int): Maximum patience. Defaults to 10.
            lambda_entropy (float): Entropy regularization. Defaults to 0.012.
            lambda_virtual (float): Virtual target coefficient. Defaults to 0.8.
            lambda_multi (float): Multi-marginal coefficient. Defaults to 1.0.
            verbose (bool): Print progress. Defaults to True.
        
        Returns:
            Generator: Perturbed generator model.
        """
        try:
            data_dim = self.evidence_list[0].shape[1]
            pert_gen = self._create_generator_copy(data_dim)
            
            # Initialize perturbation state
            theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
            
            if theta_prev.numel() == 0:
                raise RuntimeError("Generated generator has no parameters.")
                
            delta_theta_prev = torch.zeros_like(theta_prev)
            eta = eta_init
            ot_hist = []
            best_vec = None
            best_ot = float('inf')
            no_improvement_count = 0
            consecutive_rollbacks = 0
            
            for epoch in range(epochs):
                try:
                    # Estimate virtual target with congestion awareness
                    virtual_samples = self._estimate_virtual_target_with_congestion(
                        self.evidence_list, epoch
                    )
                    
                    # Generate noise for this epoch
                    noise_samples = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
                    
                    # Compute multi-marginal congestion if enabled
                    multi_congestion_info = None
                    if self.enable_congestion_tracking and self.critics:
                        multi_congestion_info = self._compute_multi_marginal_congestion(pert_gen, noise_samples)
                    
                    # Compute multi-marginal OT loss and gradients
                    loss, grads = self._compute_loss_and_grad(
                        pert_gen, virtual_samples, lambda_entropy, lambda_virtual, lambda_multi
                    )
                    
                    # Compute delta_theta with enhanced multi-marginal congestion awareness
                    if multi_congestion_info and multi_congestion_info['domains']:
                        avg_congestion_info = self._average_multi_marginal_congestion(multi_congestion_info)
                        if avg_congestion_info:
                            with torch.no_grad():
                                gen_samples = pert_gen(noise_samples)
                            delta_theta = self._compute_delta_theta_with_congestion(
                                grads, eta, clip_norm, momentum, delta_theta_prev, 
                                avg_congestion_info, virtual_samples, gen_samples
                            )
                        else:
                            delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                    else:
                        delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                    
                    # Apply update
                    theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                    delta_theta_prev = delta_theta.clone()
                    
                    # Validate and get improvement
                    ot_pert, improvement = self._validate_and_adapt(
                        pert_gen, virtual_samples, eta, ot_hist, patience, verbose, epoch
                    )
                    
                    # Update best state
                    best_ot, best_vec = self._update_best_state(ot_pert, pert_gen, best_ot, best_vec)
                    
                    # Adapt learning rate
                    eta, no_improvement_count = self._adapt_learning_rate(
                        eta, improvement, epoch, no_improvement_count, ot_hist
                    )
                    
                    # Check for rollback condition with enhanced theoretical validation
                    if self._check_rollback_condition_with_congestion(ot_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at epoch {epoch}")
                        self._restore_best_state(pert_gen, best_vec)
                        eta = max(eta * 0.95, self.config.get('eta_min', 5e-6))
                        no_improvement_count = 0
                        consecutive_rollbacks += 1
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                        
                        # Less conservative rollback handling
                        if consecutive_rollbacks >= 3:
                            if verbose:
                                print(f"Too many rollbacks, stopping early")
                            break
                    else:
                        consecutive_rollbacks = 0
                    
                    if verbose:
                        log_msg = f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                        if multi_congestion_info and multi_congestion_info['domains']:
                            total_congestion = sum(d['congestion_cost'].item() for d in multi_congestion_info['domains'])
                            log_msg += f" Total_Congestion={total_congestion:.2f}"
                        print(log_msg)
                    
                    # Early stopping conditions
                    if no_improvement_count >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch}")
                        break
                        
                    # Check for good convergence
                    if ot_pert < 0.05:  # More aggressive convergence target
                        if verbose:
                            print(f"Good convergence achieved at epoch {epoch}")
                        break
                        
                except Exception as e:
                    print(f"Error in epoch {epoch}: {e}")
                    if epoch == 0:
                        raise
                    break
            
            # Final restore to best state
            self._restore_best_state(pert_gen, best_vec)
            
            return pert_gen
            
        except Exception as e:
            print(f"Error in perturbation: {e}")
            # Return a copy as fallback
            data_dim = self.evidence_list[0].shape[1] if hasattr(self, 'evidence_list') and self.evidence_list else 2
            return self._create_generator_copy(data_dim)


# Convenience aliases
CongestionAwareWeightPerturberTargetGiven = CTWeightPerturberTargetGiven
CongestionAwareWeightPerturberTargetNotGiven = CTWeightPerturberTargetNotGiven