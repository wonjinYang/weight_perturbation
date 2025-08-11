"""
Perturbation classes with full congested transport integration.

This module provides the perturbation classes that integrate the theoretical components: 
spatial density, traffic flow, Sobolev regularization, and congestion tracking.
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
    congestion_cost_function
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
        self.eval_batch_size = self.config.get('eval_batch_size', 400)  # Reduced default
        
        # Initialize congestion tracking
        if self.enable_congestion_tracking:
            self.congestion_tracker = CongestionTracker(
                lambda_param=self.config.get('lambda_congestion', 0.05)  # Reduced default
            )
        
        # Initialize loss function with stable parameters
        self.loss_function = CongestionAwareLossFunction(
            lambda_congestion=min(self.config.get('lambda_congestion', 0.05), 0.1),    # Reduced
            lambda_sobolev=min(self.config.get('lambda_sobolev', 0.005), 0.01),       # Reduced
            lambda_entropy=min(self.config.get('lambda_entropy', 0.003), 0.01)        # Reduced
        )
        
        # Initialize common parameters
        self._initialize_common_params()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration with stability."""
        return {
            'noise_dim': 2,
            'eval_batch_size': 400,      # Reduced
            'eta_init': 0.008,           # Significantly reduced
            'eta_min': 1e-6,             # Lower minimum
            'eta_max': 0.05,             # Lower maximum
            'eta_decay_factor': 0.98,    # Less aggressive
            'eta_boost_factor': 1.01,    # Very conservative
            'clip_norm': 0.1,            # Strong clipping
            'momentum': 0.85,            # Reduced momentum
            'patience': 12,              # Increased patience
            'rollback_patience': 6,      # Increased rollback patience
            'improvement_threshold': 1e-5,   # More lenient
            'lambda_congestion': 0.03,   # Reduced
            'lambda_sobolev': 0.003,     # Reduced
            'lambda_entropy': 0.003,     # Reduced
            'congestion_threshold': 0.3, # Higher threshold
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
    
    def _compute_delta_theta_with_congestion(self, grads: torch.Tensor, eta: float, clip_norm: float,
                                            momentum: float, prev_delta: torch.Tensor,
                                            congestion_info: Optional[Dict] = None) -> torch.Tensor:
        """
        Delta_theta computation with congestion-aware scaling and improved stability.
        
        This incorporates the theoretical H''(x,i) second-order information
        to adaptively scale the perturbation based on local congestion.
        
        Args:
            grads (torch.Tensor): Flattened gradients.
            eta (float): Learning rate.
            clip_norm (float): Clipping norm (congestion bound).
            momentum (float): Momentum factor.
            prev_delta (torch.Tensor): Previous delta_theta.
            congestion_info (Optional[Dict]): Congestion metrics for adaptive scaling.
        
        Returns:
            torch.Tensor: Computed delta_theta.
        """
        # Gradient preprocessing
        grads = torch.clamp(grads, min=-2.0, max=2.0)  # Stronger clamping
        
        # Check for problematic gradients
        if torch.isnan(grads).any() or torch.isinf(grads).any():
            print("Warning: NaN or inf detected in gradients, using zero gradients")
            grads = torch.zeros_like(grads)
        
        # Apply adaptive gradient scaling based on magnitude
        grad_norm = grads.norm()
        if grad_norm > 10.0:  # Very large gradients
            grads = grads * (10.0 / grad_norm)
            print(f"Warning: Large gradient norm {grad_norm:.2f}, scaled down")
        
        delta_theta = -eta * grads
        
        # Apply congestion-based scaling if available (with conservation)
        if congestion_info is not None and 'traffic_intensity' in congestion_info:
            try:
                # More conservative congestion-based adaptation
                traffic_intensity = congestion_info['traffic_intensity'].mean()
                sigma_mean = congestion_info['spatial_density'].mean()
                lambda_param = min(self.config.get('lambda_congestion', 0.05), 0.1)
                
                # Second derivative computation with stability
                h_second = 1.0 / (lambda_param * sigma_mean + 1e-5)
                h_second = min(float(h_second), 50.0)  # Cap influence
                
                # Much more conservative adaptive scaling
                congestion_factor = 0.05 * h_second * traffic_intensity  # Reduced from 0.1
                congestion_scale = 1.0 / (1.0 + congestion_factor)
                congestion_scale = max(0.2, min(congestion_scale, 1.5))  # Tighter bounds
                
                delta_theta = delta_theta * congestion_scale
                
                # Additional stability check for congestion adaptation
                if traffic_intensity > 10.0:  # Very high congestion
                    delta_theta = delta_theta * 0.5  # Additional dampening
                    
            except Exception as e:
                print(f"Warning: Congestion adaptation failed: {e}")
        
        # Clipping with parameter-count aware scaling
        norm = delta_theta.norm()
        param_count = len(grads)
        
        # More conservative parameter scaling
        param_scale = max(1.0, min(param_count / 50000.0, 1.5))  # Less aggressive, cap at 1.5x
        max_norm = clip_norm * param_scale
        
        if norm > max_norm:
            delta_theta = delta_theta * (max_norm / (norm + 1e-8))
        
        # Momentum with stability checks
        prev_norm = prev_delta.norm()
        if prev_norm > 5.0:  # If previous delta is large, reduce momentum significantly
            momentum = momentum * 0.3
        elif prev_norm > 2.0:
            momentum = momentum * 0.7
        
        # Apply momentum
        delta_theta = momentum * prev_delta + (1 - momentum) * delta_theta
        
        # Final comprehensive clamping
        delta_theta = torch.clamp(delta_theta, min=-0.5, max=0.5)  # Stronger final bounds
        
        # Additional check for explosion prevention
        final_norm = delta_theta.norm()
        if final_norm > 1.0:
            delta_theta = delta_theta * (0.8 / final_norm)  # Conservative scaling
        
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
        
        # Check for excessively large updates
        delta_norm = delta_theta.norm()
        if delta_norm > 2.0:
            print(f"Warning: Very large delta_theta norm {delta_norm:.2f}, scaling down")
            delta_theta = delta_theta * (1.0 / delta_norm)
        
        theta_new = theta_prev + delta_theta
        
        # Parameter checking
        if torch.isnan(theta_new).any() or torch.isinf(theta_new).any():
            print("Warning: NaN or inf detected in new parameters, keeping old parameters")
            theta_new = theta_prev.clone()
        
        # Check for parameter explosion
        param_norm = theta_new.norm()
        if param_norm > 100.0:  # Very large parameters
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
        Adaptive learning rate with more conservative adjustments and stability monitoring.
        
        Args:
            eta (float): Current learning rate.
            improvement (float): Current improvement value.
            step (int): Current step/epoch.
            no_improvement_count (int): Consecutive steps without improvement.
            loss_history (List[float]): Recent loss history for trend analysis.
            
        Returns:
            Tuple[float, int]: (new_eta, updated_no_improvement_count)
        """
        eta_min = max(self.config.get('eta_min', 1e-6), 1e-7)      # Lower minimum
        eta_max = min(self.config.get('eta_max', 0.05), 0.1)       # Lower maximum
        eta_decay_factor = max(self.config.get('eta_decay_factor', 0.98), 0.95)   # Conservative
        eta_boost_factor = min(self.config.get('eta_boost_factor', 1.01), 1.03)   # Very conservative
        improvement_threshold = self.config.get('improvement_threshold', 1e-5)
        
        # Adaptive threshold based on loss magnitude
        if loss_history and loss_history[-1] > 100.0:
            improvement_threshold *= 50  # Much more lenient for large losses
        elif loss_history and loss_history[-1] > 10.0:
            improvement_threshold *= 10  # More lenient for moderate losses
        
        # Trend analysis
        is_stagnating = False
        is_oscillating = False
        if len(loss_history) >= 6:
            recent_losses = loss_history[-6:]
            # Check for stagnation
            recent_var = np.var(recent_losses)
            is_stagnating = recent_var < improvement_threshold / 100
            
            # Check for oscillation (rapid alternating between up and down)
            diffs = np.diff(recent_losses)
            if len(diffs) >= 4:
                sign_changes = np.sum(np.diff(np.sign(diffs)) != 0)
                is_oscillating = sign_changes >= 3  # Many sign changes indicate oscillation
        
        # Learning rate adaptation logic
        if improvement > improvement_threshold:
            # Good improvement: slightly boost learning rate
            new_eta = min(eta * eta_boost_factor, eta_max)
            new_no_improvement_count = 0
            
        elif improvement < -improvement_threshold * 10:  # Much more lenient degradation threshold
            # Significant degradation: decay learning rate
            decay_factor = eta_decay_factor if step > 5 else 0.95  # Less aggressive early on
            new_eta = max(eta * decay_factor, eta_min)
            new_no_improvement_count = no_improvement_count + 1
            
        elif is_oscillating:
            # Oscillation detected: reduce learning rate to stabilize
            new_eta = max(eta * 0.9, eta_min)
            new_no_improvement_count = no_improvement_count + 1
            if step % 5 == 0:  # Periodic logging
                print(f"Oscillation detected, reducing learning rate to {new_eta:.6f}")
                
        elif is_stagnating:
            # Stagnation: small reduction
            new_eta = max(eta * 0.99, eta_min)
            new_no_improvement_count = no_improvement_count + 1
            
        else:
            # Marginal change: very gentle decay
            new_eta = max(eta * 0.999, eta_min)  # Very minimal decay
            new_no_improvement_count = no_improvement_count + 1
        
        # Prevent learning rate from being reduced too aggressively early on
        if step < 15 and new_eta < eta * 0.5:
            new_eta = eta * 0.8  # Less aggressive reduction in early stages
            
        # Safety bounds
        new_eta = max(eta_min, min(new_eta, eta_max))
        
        return new_eta, new_no_improvement_count
    
    def _check_rollback_condition_with_congestion(self, loss_hist: List[float], 
                                                 no_improvement_count: int) -> bool:
        """
        Rollback condition checking with congestion monitoring.
        
        Args:
            loss_hist (List[float]): History of loss values.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            bool: True if rollback should be triggered.
        """
        # Check standard rollback conditions with higher thresholds
        should_rollback = self._check_rollback_condition(loss_hist, no_improvement_count)
        
        # Additional congestion check
        if self.enable_congestion_tracking and self.congestion_tracker:
            congestion_threshold = max(self.config.get('congestion_threshold', 0.3), 0.4)  # Higher threshold
            if self.congestion_tracker.check_congestion_increase(congestion_threshold):
                print("Rollback triggered due to excessive congestion increase")
                return True
                
        return should_rollback
    
    def _check_rollback_condition(self, loss_hist: List[float], 
                                 no_improvement_count: int) -> bool:
        """
        Rollback condition checking with more conservative triggers.
        
        Args:
            loss_hist (List[float]): History of loss values.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            bool: True if rollback should be triggered.
        """
        rollback_patience = max(self.config.get('rollback_patience', 6), 8)  # Higher patience
        
        # Trigger rollback if no improvement for many consecutive steps
        if no_improvement_count >= rollback_patience * 2.5:  # Even more patient
            return True
            
        # Check for consistent severe loss increase over longer period
        if len(loss_hist) >= rollback_patience + 8:  # Longer window
            recent_losses = loss_hist[-(rollback_patience + 8):]
            # Check if loss has been consistently increasing (more lenient threshold)
            increasing_count = sum(1 for i in range(1, len(recent_losses)) 
                                 if recent_losses[i] >= recent_losses[i-1])
            if increasing_count >= len(recent_losses) * 0.9:  # 90% must be increasing
                return True
        
        # Check for sudden severe spike in loss (more lenient)
        if len(loss_hist) >= 8:
            recent_avg = sum(loss_hist[-4:]) / 4
            prev_avg = sum(loss_hist[-8:-4]) / 4
            if recent_avg > prev_avg * 5.0:  # 5x increase (was 3x)
                return True
        
        # Additional check for loss explosion
        if len(loss_hist) >= 3:
            if loss_hist[-1] > 1000.0 and loss_hist[-1] > loss_hist[-3] * 10:
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
        # More lenient improvement requirement to prevent getting stuck
        improvement_required = max(0.001, best_loss * 0.0001)  # 0.01% improvement required
        
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
    Weight Perturber for target-given perturbation with congestion tracking.
    
    This class now incorporates the full theoretical framework including:
    - Spatial density estimation σ(x)
    - Traffic flow computation w_Q
    - Sobolev regularization
    - Congestion cost tracking
    - Continuity equation verification
    
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
        Compute loss and gradients with full congestion tracking.
        
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
            lambda_congestion=min(self.config.get('lambda_congestion', 0.05), 0.1),
            lambda_sobolev=min(self.config.get('lambda_sobolev', 0.005), 0.01),
            track_congestion=True,
            use_direct_w2=True,
            w2_weight=0.9,  # Increased W2 weight for stability
            map_weight=0.1   # Reduced mapping weight
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
        
        # Use improved loss function with hybrid approach
        loss, grads, _ = global_w2_loss_and_grad_with_congestion(
            pert_gen, 
            self.target_samples, 
            noise,
            critic=self.critic,
            track_congestion=False,
            use_direct_w2=True,
            w2_weight=0.9,
            map_weight=0.1
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
        
        # Use larger blur for more stable evaluation
        try:
            w2_orig = compute_wasserstein_distance(orig_out, self.target_samples, p=2, blur=0.15)
            w2_pert = compute_wasserstein_distance(pert_out, self.target_samples, p=2, blur=0.15)
        except:
            # Fallback: simple MSE distance
            w2_orig = ((orig_out.unsqueeze(1) - self.target_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
            w2_pert = ((pert_out.unsqueeze(1) - self.target_samples.unsqueeze(0))**2).sum(-1).min(1)[0].mean()
        
        # Compute improvement compared to original
        improvement = w2_orig.item() - w2_pert.item()
        w2_hist.append(w2_pert.item())
        
        return w2_pert.item(), improvement
    
    def perturb(self, steps: int = 50, eta_init: float = 0.008, clip_norm: float = 0.1,
                momentum: float = 0.85, patience: int = 12, verbose: bool = True) -> Generator:
        """
        Perform the perturbation process with superior stability.
        
        Args:
            steps (int): Number of perturbation steps. Defaults to 50.
            eta_init (float): Initial learning rate. Defaults to 0.008.
            clip_norm (float): Gradient clipping norm. Defaults to 0.1.
            momentum (float): Momentum factor. Defaults to 0.85.
            patience (int): Maximum patience. Defaults to 12.
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
                    # Compute loss and gradients with congestion tracking
                    if self.enable_congestion_tracking and self.critic is not None:
                        loss, grads, congestion_info = self._compute_loss_and_grad_with_congestion(pert_gen)
                        
                        # Update congestion tracker
                        if congestion_info:
                            self.congestion_tracker.update(congestion_info)
                        
                        # Compute delta_theta with congestion awareness
                        delta_theta = self._compute_delta_theta_with_congestion(
                            grads, eta, clip_norm, momentum, delta_theta_prev, congestion_info
                        )
                    else:
                        loss, grads = self._compute_loss_and_grad(pert_gen)
                        delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                    
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
                    
                    # Check for rollback condition
                    if self._check_rollback_condition_with_congestion(w2_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at step {step}")
                        
                        self._restore_best_state(pert_gen, best_vec)
                        # Reset parameters after rollback
                        eta = max(eta * 0.9, self.config.get('eta_min', 1e-6))
                        no_improvement_count = 0
                        consecutive_rollbacks += 1
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                        
                        # If too many rollbacks, break early
                        if consecutive_rollbacks >= 3:  # Reduced from 5
                            if verbose:
                                print(f"Too many rollbacks, stopping early")
                            break
                    else:
                        consecutive_rollbacks = 0
                    
                    if verbose:
                        log_msg = f"[{step:2d}] W2(Pert, Target)={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                        if self.enable_congestion_tracking and 'congestion_info' in locals() and congestion_info:
                            log_msg += f" Congestion={congestion_info.get('congestion_cost', torch.tensor(0)).item():.2f}"
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
    Weight Perturber for evidence-based perturbation with multi-marginal congestion tracking.
    
    This class now incorporates:
    - Multi-marginal traffic flow computation
    - Evidence-weighted spatial density
    - Multi-marginal continuity equation
    - Adaptive virtual target estimation with congestion awareness
    
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
        
        # Override eval_batch_size for Section 3 with more conservative value
        self.eval_batch_size = min(self.config.get('eval_batch_size', 400), 300)  # Cap at 300
        
        if len(self.evidence_list) == 0:
            raise ValueError("Evidence list must not be empty.")
        
        # Initialize multi-marginal congestion trackers if enabled
        if self.enable_congestion_tracking:
            self.multi_congestion_trackers = [
                CongestionTracker(lambda_param=min(self.config.get('lambda_congestion', 0.05), 0.1))
                for _ in self.evidence_list
            ]
    
    def _estimate_virtual_target_with_congestion(
        self, evidence_list: List[torch.Tensor], epoch: int
    ) -> torch.Tensor:
        """
        Virtual target estimation with congestion awareness and superior stability.
        
        Args:
            evidence_list (List[torch.Tensor]): Evidence domains.
            epoch (int): Current epoch for adaptation.
            
        Returns:
            torch.Tensor: Virtual target samples.
        """
        # Much more conservative bandwidth adaptation
        base_bandwidth = 0.3  # Increased base bandwidth for stability
        
        if self.enable_congestion_tracking and self.multi_congestion_trackers:
            congestion_values = [
                tracker.get_average_congestion() 
                for tracker in self.multi_congestion_trackers
                if len(tracker.history['congestion_cost']) > 0
            ]
            if congestion_values:
                avg_congestion = np.mean(congestion_values)
                # Much more conservative bandwidth adjustment
                bandwidth = base_bandwidth * (1.0 + 0.1 * avg_congestion)  # Reduced factor from 0.2
            else:
                bandwidth = base_bandwidth
        else:
            bandwidth = base_bandwidth
        
        # Very conservative time-based annealing
        bandwidth += 0.1 * torch.exp(torch.tensor(-epoch / 20.0)).item()  # Slower decay, larger addition
        bandwidth = max(bandwidth, 0.2)  # Higher minimum bandwidth
        
        # Additional stability: cap the bandwidth
        bandwidth = min(bandwidth, 0.6)  # Maximum bandwidth cap
        
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
        Multi-marginal congestion computation with superior stability.
        
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
                # Compute spatial density for this evidence domain with stability
                all_samples = torch.cat([gen_samples, evidence], dim=0)
                density_info = compute_spatial_density(all_samples, bandwidth=0.25)  # Increased bandwidth
                sigma_gen = density_info['density_at_samples'][:gen_samples.shape[0]]
                
                # Compute traffic flow for this domain with much reduced lambda
                flow_info = compute_traffic_flow(
                    critic, pert_gen, noise_samples, sigma_gen,
                    lambda_param=min(self.config.get('lambda_congestion', 0.05), 0.08)  # Much reduced
                )
                
                 # Compute congestion cost without restrictive clamping  
                raw_congestion_cost = congestion_cost_function(
                    flow_info['traffic_intensity'], sigma_gen,
                    lambda_param=min(self.config.get('lambda_congestion', 0.05), 0.08)
                )
                
                # Apply mean reduction
                congestion_cost = raw_congestion_cost.mean()
                
                # Only clamp for extreme numerical issues, allow wide range of values
                if torch.isnan(congestion_cost) or torch.isinf(congestion_cost):
                    congestion_cost = torch.zeros(1, device=congestion_cost.device)[0]
                elif congestion_cost < 0:
                    congestion_cost = torch.abs(congestion_cost)  # Ensure positive
                # Remove the artificial upper limit to allow natural cost variations
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
                multi_congestion_info['domains'].append(domain_info)
        
        return multi_congestion_info
    
    def _average_multi_marginal_congestion(self, multi_congestion_info: Dict) -> Dict:
        """
        Average congestion information across domains with stability.
        
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
            # Average traffic intensity with stability
            all_intensities = [d['traffic_intensity'] for d in domains]
            if all_intensities:
                stacked_intensities = torch.stack(all_intensities)
                # Use median instead of mean for more stability
                avg_info['traffic_intensity'] = torch.median(stacked_intensities, dim=0)[0]
            
            # Average spatial density with stability
            all_densities = [d['spatial_density'] for d in domains]
            if all_densities:
                stacked_densities = torch.stack(all_densities)
                avg_info['spatial_density'] = torch.median(stacked_densities, dim=0)[0]
            
            # Average congestion cost with stability
            all_costs = [d['congestion_cost'] for d in domains]
            if all_costs:
                cost_tensor = torch.stack(all_costs)
                avg_info['congestion_cost'] = torch.median(cost_tensor)
                
        except Exception as e:
            print(f"Warning: Failed to average congestion info: {e}")
            return {}
        
        return avg_info
    
    def _compute_loss_and_grad(self, pert_gen: Generator, virtual_samples: torch.Tensor,
                               lambda_entropy: float, lambda_virtual: float, lambda_multi: float
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Multi-marginal OT loss computation with superior stability.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        gen_out = pert_gen(noise)
        
        try:
            loss = multi_marginal_ot_loss(
                gen_out, self.evidence_list, virtual_samples,
                blur=0.2,  # Larger blur for stability
                lambda_virtual=min(lambda_virtual, 0.8),    # Cap at 0.8
                lambda_multi=min(lambda_multi, 0.8),       # Cap at 0.8
                lambda_entropy=min(lambda_entropy, 0.01)   # Cap at 0.01
            )
            
            # Additional stability: clamp loss
            loss = torch.clamp(loss, max=1000.0)
            
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
        Validation with superior stability.
        
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
                blur=0.2,  # Larger blur for stability
                lambda_virtual=min(self.config.get('lambda_virtual', 0.5), 0.8),
                lambda_multi=min(self.config.get('lambda_multi', 0.7), 0.8),
                lambda_entropy=min(self.config.get('lambda_entropy', 0.003), 0.01)
            ).item()

            ot_pert = multi_marginal_ot_loss(
                pert_out, self.evidence_list, virtual_samples,
                blur=0.2,
                lambda_virtual=min(self.config.get('lambda_virtual', 0.5), 0.8),
                lambda_multi=min(self.config.get('lambda_multi', 0.7), 0.8),
                lambda_entropy=min(self.config.get('lambda_entropy', 0.003), 0.01)
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
    
    def perturb(self, epochs: int = 30, eta_init: float = 0.008, clip_norm: float = 0.1,
                momentum: float = 0.85, patience: int = 12, lambda_entropy: float = 0.003,
                lambda_virtual: float = 0.5, lambda_multi: float = 0.7, verbose: bool = True) -> Generator:
        """
        Perform the perturbation process with superior stability.
        
        Args:
            epochs (int): Number of epochs. Defaults to 30.
            eta_init (float): Initial learning rate. Defaults to 0.008.
            clip_norm (float): Gradient clipping norm. Defaults to 0.1.
            momentum (float): Momentum factor. Defaults to 0.85.
            patience (int): Maximum patience. Defaults to 12.
            lambda_entropy (float): Entropy regularization. Defaults to 0.003.
            lambda_virtual (float): Virtual target coefficient. Defaults to 0.5.
            lambda_multi (float): Multi-marginal coefficient. Defaults to 0.7.
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
                    
                    # Compute delta_theta with multi-marginal congestion awareness
                    if multi_congestion_info and multi_congestion_info['domains']:
                        avg_congestion_info = self._average_multi_marginal_congestion(multi_congestion_info)
                        if avg_congestion_info:
                            delta_theta = self._compute_delta_theta_with_congestion(
                                grads, eta, clip_norm, momentum, delta_theta_prev, avg_congestion_info
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
                    
                    # Check for rollback condition
                    if self._check_rollback_condition_with_congestion(ot_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at epoch {epoch}")
                        self._restore_best_state(pert_gen, best_vec)
                        eta = max(eta * 0.9, self.config.get('eta_min', 1e-6))
                        no_improvement_count = 0
                        consecutive_rollbacks += 1
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                        
                        # More conservative rollback handling
                        if consecutive_rollbacks >= 3:  # Reduced from 5
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
                        
                    # Check for very good convergence
                    if ot_pert < 0.1:
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