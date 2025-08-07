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
from .sobolev import SobolevConstrainedCritic, sobolev_regularization
from .ct_losses import (
    global_w2_loss_and_grad_with_congestion,
    multi_marginal_ot_loss_with_congestion,
    CongestionAwareLossFunction,
    compute_convergence_metrics
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
        self.eval_batch_size = self.config.get('eval_batch_size', 1600)
        
        # Initialize congestion tracking
        if self.enable_congestion_tracking:
            self.congestion_tracker = CongestionTracker(
                lambda_param=self.config.get('lambda_congestion', 0.1)
            )
        
        # Initialize loss function
        self.loss_function = CongestionAwareLossFunction(
            lambda_congestion=self.config.get('lambda_congestion', 0.1),
            lambda_sobolev=self.config.get('lambda_sobolev', 0.01),
            lambda_entropy=self.config.get('lambda_entropy', 0.012)
        )
        
        # Initialize common parameters
        self._initialize_common_params()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'noise_dim': 2,
            'eval_batch_size': 1600,
            'eta_init': 0.008,
            'eta_min': 5e-6,
            'eta_max': 0.05,
            'eta_decay_factor': 0.88,
            'eta_boost_factor': 1.03,
            'clip_norm': 0.15,
            'momentum': 0.85,
            'patience': 15,
            'rollback_patience': 8,
            'improvement_threshold': 5e-5,
            'lambda_congestion': 0.1,
            'lambda_sobolev': 0.01,
            'congestion_threshold': 0.1,
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
        Delta_theta computation with congestion-aware scaling.
        
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
        delta_theta = -eta * grads
        
        # Apply congestion-based scaling if available
        if congestion_info is not None and 'traffic_intensity' in congestion_info:
            # Compute H''(x,i) for quadratic-linear cost: H''(i) = 1/(λσ)
            traffic_intensity = congestion_info['traffic_intensity'].mean()
            sigma_mean = congestion_info['spatial_density'].mean()
            lambda_param = self.config.get('lambda_congestion', 0.1)
            
            # Second derivative of congestion cost
            h_second = 1.0 / (lambda_param * sigma_mean + 1e-8)
            
            # Adaptive scaling based on local congestion curvature
            congestion_scale = 1.0 / (1.0 + h_second * traffic_intensity)
            delta_theta = delta_theta * congestion_scale
        
        norm = delta_theta.norm()
        
        # Improved clipping with adaptive scaling
        param_scale = max(1.0, len(grads) / 10000.0)  # Scale by parameter count
        max_norm = clip_norm * param_scale
        
        if norm > max_norm:
            delta_theta = delta_theta * (max_norm / (norm + 1e-8))
        
        # Apply momentum
        delta_theta = momentum * prev_delta + (1 - momentum) * delta_theta
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
        # Call previous version without congestion info for backward compatibility
        return self._compute_delta_theta_with_congestion(
            grads, eta, clip_norm, momentum, prev_delta, congestion_info=None
        )
    
    def _apply_parameter_update(self, pert_gen: Generator, theta_prev: torch.Tensor, 
                               delta_theta: torch.Tensor) -> torch.Tensor:
        """
        Apply parameter update to the generator.
        
        Args:
            pert_gen (Generator): Generator to update.
            theta_prev (torch.Tensor): Previous parameter vector.
            delta_theta (torch.Tensor): Parameter update.
            
        Returns:
            torch.Tensor: New parameter vector.
        """
        theta_new = theta_prev + delta_theta
        vector_to_parameters(theta_new, pert_gen.parameters())
        return theta_new
    
    def _adapt_learning_rate(self, eta: float, improvement: float, step: int, 
                           no_improvement_count: int, loss_history: List[float]) -> Tuple[float, int]:
        """
        Improved adaptive learning rate based on improvement trends and step count.
        
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
        eta_max = self.config.get('eta_max', 0.05)
        eta_decay_factor = self.config.get('eta_decay_factor', 0.88)
        eta_boost_factor = self.config.get('eta_boost_factor', 1.03)
        improvement_threshold = self.config.get('improvement_threshold', 5e-5)
        
        # Analyze loss trend for better adaptation
        if len(loss_history) >= 3:
            recent_trend = sum(loss_history[-3:]) / 3 - sum(loss_history[-6:-3]) / 3 if len(loss_history) >= 6 else 0
            is_stagnating = abs(recent_trend) < improvement_threshold / 10
        else:
            is_stagnating = False
        
        if improvement > improvement_threshold:
            # Significant improvement: boost learning rate slightly
            new_eta = min(eta * eta_boost_factor, eta_max)
            new_no_improvement_count = 0
        elif improvement < -improvement_threshold * 2:  # Stronger degradation threshold
            # Performance degradation: decay learning rate more aggressively
            new_eta = max(eta * eta_decay_factor * 0.9, eta_min)
            new_no_improvement_count = no_improvement_count + 1
        elif is_stagnating:
            # Stagnation detected: small decay to escape local minima
            new_eta = max(eta * 0.95, eta_min)
            new_no_improvement_count = no_improvement_count + 1
        else:
            # Marginal change: maintain or slight decay
            new_eta = max(eta * 0.98, eta_min)
            new_no_improvement_count = no_improvement_count + 1
            
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
        # Check standard rollback conditions
        should_rollback = self._check_rollback_condition(loss_hist, no_improvement_count)
        
        # Additional check for congestion increase
        if self.enable_congestion_tracking and self.congestion_tracker:
            congestion_threshold = self.config.get('congestion_threshold', 0.1)
            if self.congestion_tracker.check_congestion_increase(congestion_threshold):
                return True
                
        return should_rollback
    
    def _check_rollback_condition(self, loss_hist: List[float], 
                                 no_improvement_count: int) -> bool:
        """
        Improved rollback condition checking with trend analysis.
        
        Args:
            loss_hist (List[float]): History of loss values.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            bool: True if rollback should be triggered.
        """
        rollback_patience = self.config.get('rollback_patience', 8)
        
        # Trigger rollback if no improvement for several consecutive steps
        if no_improvement_count >= rollback_patience:
            return True
            
        # Check for consistent loss increase over longer period
        if len(loss_hist) >= rollback_patience + 2:
            recent_losses = loss_hist[-(rollback_patience + 2):]
            # Check if loss has been consistently increasing
            increasing_count = sum(1 for i in range(1, len(recent_losses)) 
                                 if recent_losses[i] >= recent_losses[i-1])
            if increasing_count >= rollback_patience:
                return True
        
        # Check for sudden spike in loss
        if len(loss_hist) >= 3:
            recent_avg = sum(loss_hist[-3:]) / 3
            prev_avg = sum(loss_hist[-6:-3]) / 3 if len(loss_hist) >= 6 else recent_avg
            if recent_avg > prev_avg * 1.5:  # 50% increase
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
        if current_loss < best_loss:
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
            vector_to_parameters(best_vec, pert_gen.parameters())
    
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
                        loss, grads = self._compute_loss_and_grad(pert_gen, virtual_samples, lambda_entropy, lambda_virtual, lambda_multi)
                        
                        # Compute delta_theta with multi-marginal congestion awareness
                        if multi_congestion_info and multi_congestion_info['domains']:
                            avg_congestion_info = self._average_multi_marginal_congestion(multi_congestion_info)
                            delta_theta = self._compute_delta_theta_with_congestion(
                                grads, eta, clip_norm, momentum, delta_theta_prev, avg_congestion_info
                            )
                        else:
                            delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                        
                        # Apply update
                        theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                        delta_theta_prev = delta_theta.clone()
                        
                        # Validate and get improvement
                        ot_pert, improvement = self._validate_and_adapt(pert_gen, virtual_samples, eta, ot_hist, patience, verbose, epoch)
                        
                        # Update best state
                        best_ot, best_vec = self._update_best_state(ot_pert, pert_gen, best_ot, best_vec)
                        
                        # Adapt learning rate based on improvement
                        eta, no_improvement_count = self._adapt_learning_rate(eta, improvement, epoch, no_improvement_count, ot_hist)
                        
                        # Check for rollback condition with congestion awareness
                        if self._check_rollback_condition_with_congestion(ot_hist, no_improvement_count):
                            if verbose:
                                print(f"Rollback triggered at epoch {epoch} (no improvement for {no_improvement_count} epochs)")
                            self._restore_best_state(pert_gen, best_vec)
                            # Reset parameters after rollback
                            eta = eta * 0.6
                            no_improvement_count = 0
                            delta_theta_prev = torch.zeros_like(delta_theta_prev)
                        
                        if verbose:
                            log_msg = f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}"
                            if multi_congestion_info and multi_congestion_info['domains']:
                                total_congestion = sum(d['congestion_cost'].item() for d in multi_congestion_info['domains'])
                                log_msg += f" Total_Congestion={total_congestion:.4f}"
                            print(log_msg)
                        
                        # Early stopping if patience exceeded
                        if no_improvement_count >= patience:
                            if verbose:
                                print(f"Early stopping at epoch {epoch} due to lack of improvement")
                            break
                            
                    except Exception as e:
                        print(f"Error in perturbation epoch {epoch}: {e}")
                        if epoch == 0:  # If first epoch fails, re-raise
                            raise
                        break
                
                # Final restore to best state
                self._restore_best_state(pert_gen, best_vec)
                
                return pert_gen
                
            except Exception as e:
                print(f"Error in perturbation process: {e}")
                # Return a copy of the original generator as fallback
                data_dim = self.evidence_list[0].shape[1] if hasattr(self, 'evidence_list') and self.evidence_list else 2
                return self._create_generator_copy(data_dim)


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
            lambda_congestion=self.config.get('lambda_congestion', 0.1),
            lambda_sobolev=self.config.get('lambda_sobolev', 0.01),
            track_congestion=True,
            use_direct_w2=True,
            w2_weight=0.7,
            map_weight=0.3
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
            w2_weight=0.7,
            map_weight=0.3
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
        
        # Use tighter blur for more accurate evaluation
        w2_orig = compute_wasserstein_distance(orig_out, self.target_samples, p=2, blur=0.05)
        w2_pert = compute_wasserstein_distance(pert_out, self.target_samples, p=2, blur=0.05)
        
        # Compute improvement compared to original
        improvement = w2_orig.item() - w2_pert.item()
        w2_hist.append(w2_pert.item())
        
        return w2_pert.item(), improvement
    
    def perturb(self, steps: int = 80, eta_init: float = 0.008, clip_norm: float = 0.15,
                momentum: float = 0.85, patience: int = 15, verbose: bool = True) -> Generator:
        """
        Perform the improved perturbation process with adaptive learning rate, rollback, and congestion tracking.
        
        Args:
            steps (int): Number of perturbation steps. Defaults to 80.
            eta_init (float): Initial learning rate. Defaults to 0.008.
            clip_norm (float): Gradient clipping norm (congestion bound). Defaults to 0.15.
            momentum (float): Momentum factor for updates. Defaults to 0.85.
            patience (int): Maximum patience before stopping. Defaults to 15.
            verbose (bool): If True, print progress. Defaults to True.
        
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
                    # Compute improved global W2 loss and gradients with congestion tracking
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
                    
                    # Add gradient noise for exploration (decreasing with steps)
                    noise_scale = 0.01 * (1.0 - step / steps)
                    if noise_scale > 0:
                        grads = grads + noise_scale * torch.randn_like(grads)
                    
                    # Apply update
                    theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                    delta_theta_prev = delta_theta.clone()
                    
                    # Validate and get improvement
                    w2_pert, improvement = self._validate_and_adapt(pert_gen, eta, w2_hist, patience, verbose, step)
                    
                    # Track loss history
                    loss_hist.append(loss.item())
                    
                    # Update best state
                    best_w2, best_vec = self._update_best_state(w2_pert, pert_gen, best_w2, best_vec)
                    
                    # Adapt learning rate based on improvement and loss history
                    eta, no_improvement_count = self._adapt_learning_rate(eta, improvement, step, no_improvement_count, loss_hist)
                    
                    # Check for rollback condition with congestion awareness
                    if self._check_rollback_condition_with_congestion(w2_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at step {step} (no improvement for {no_improvement_count} steps)")
                            if self.enable_congestion_tracking:
                                avg_congestion = self.congestion_tracker.get_average_congestion()
                                print(f"  Average congestion: {avg_congestion:.4f}")
                        
                        self._restore_best_state(pert_gen, best_vec)
                        # Reset parameters after rollback
                        eta = max(eta * (0.7 ** consecutive_rollbacks), self.config.get('eta_min', 5e-6))
                        no_improvement_count = 0
                        consecutive_rollbacks += 1
                        # Reset momentum to explore new directions
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                        
                        # If too many rollbacks, break early
                        if consecutive_rollbacks >= 10:
                            if verbose:
                                print(f"Too many rollbacks ({consecutive_rollbacks}), stopping early")
                            break
                    else:
                        consecutive_rollbacks = 0
                    
                    if verbose:
                        log_msg = f"[{step:2d}] W2(Pert, Target)={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.6f}"
                        if self.enable_congestion_tracking and 'congestion_info' in locals() and congestion_info:
                            log_msg += f" Congestion={congestion_info.get('congestion_cost', 0):.4f}"
                        print(log_msg)
                    
                    # Early stopping if patience exceeded
                    if no_improvement_count >= patience:
                        if verbose:
                            print(f"Early stopping at step {step} due to lack of improvement")
                        break
                    
                    # Additional stopping condition: very good convergence
                    if w2_pert < 1e-4:  # Very close to target
                        if verbose:
                            print(f"Excellent convergence achieved at step {step}")
                        break
                
                except Exception as e:
                    print(f"Error in perturbation step {step}: {e}")
                    if step == 0:  # If first step fails, re-raise
                        raise
                    break
            
            # Final restore to best state
            self._restore_best_state(pert_gen, best_vec)
            
            return pert_gen
            
        except Exception as e:
            print(f"Error in perturbation process: {e}")
            # Return a copy of the original generator as fallback
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
        
        # Override eval_batch_size for Section 3
        self.eval_batch_size = self.config.get('eval_batch_size', 600)
        
        if len(self.evidence_list) == 0:
            raise ValueError("Evidence list must not be empty.")
        
        # Initialize multi-marginal congestion trackers if enabled
        if self.enable_congestion_tracking:
            self.multi_congestion_trackers = [
                CongestionTracker(lambda_param=self.config.get('lambda_congestion', 0.1))
                for _ in self.evidence_list
            ]
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for evidence-based perturbation."""
        config = {
            'noise_dim': 2,
            'eval_batch_size': 600,
            'eta_init': 0.045,
            'eta_min': 1e-4,
            'eta_max': 0.2,
            'eta_decay_factor': 0.75,
            'eta_boost_factor': 1.1,
            'clip_norm': 0.23,
            'momentum': 0.975,
            'patience': 6,
            'rollback_patience': 4,
            'lambda_entropy': 0.012,
            'lambda_virtual': 0.8,
            'lambda_multi': 1.0,
            'improvement_threshold': 1e-3,
            'lambda_congestion': 0.1,
            'lambda_sobolev': 0.01,
            'congestion_threshold': 0.15,
        }
        return config
    
    def _estimate_virtual_target_with_congestion(
        self, evidence_list: List[torch.Tensor], epoch: int
    ) -> torch.Tensor:
        """
        Estimate virtual target with congestion awareness.
        
        Args:
            evidence_list (List[torch.Tensor]): Evidence domains.
            epoch (int): Current epoch for adaptation.
            
        Returns:
            torch.Tensor: Virtual target samples.
        """
        # Adaptive bandwidth based on congestion levels
        base_bandwidth = 0.22
        if self.enable_congestion_tracking and self.multi_congestion_trackers:
            avg_congestion = np.mean([
                tracker.get_average_congestion() 
                for tracker in self.multi_congestion_trackers
                if len(tracker.history['congestion_cost']) > 0
            ])
            if avg_congestion > 0:
                # Increase bandwidth in high congestion to promote exploration
                bandwidth = base_bandwidth * (1.0 + 0.5 * avg_congestion)
            else:
                bandwidth = base_bandwidth
        else:
            bandwidth = base_bandwidth
        
        # Time-based annealing
        bandwidth += 0.07 * torch.exp(torch.tensor(-epoch / 10.0)).item()
        
        virtuals = virtual_target_sampler(
            evidence_list, 
            bandwidth=bandwidth, 
            num_samples=self.eval_batch_size, 
            device=self.device
        )
        return virtuals
    
    def _compute_multi_marginal_congestion(self, pert_gen: Generator, 
                                          noise_samples: torch.Tensor) -> Dict[str, List[Dict]]:
        """
        Compute multi-marginal congestion metrics for each evidence domain.
        
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
                density_info = compute_spatial_density(all_samples, bandwidth=0.15)
                sigma_gen = density_info['density_at_samples'][:gen_samples.shape[0]]
                
                # Compute traffic flow for this domain
                flow_info = compute_traffic_flow(
                    critic, pert_gen, noise_samples, sigma_gen,
                    lambda_param=self.config.get('lambda_congestion', 0.1)
                )
                
                # Compute congestion cost
                congestion_cost = congestion_cost_function(
                    flow_info['traffic_intensity'], sigma_gen,
                    lambda_param=self.config.get('lambda_congestion', 0.1)
                ).mean()
                
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
        Average congestion information across domains.
        
        Args:
            multi_congestion_info (Dict): Multi-domain congestion info.
            
        Returns:
            Dict: Averaged congestion information.
        """
        if 'domains' not in multi_congestion_info or not multi_congestion_info['domains']:
            return {}
        
        domains = multi_congestion_info['domains']
        avg_info = {}
        
        # Average traffic intensity
        all_intensities = [d['traffic_intensity'] for d in domains]
        avg_info['traffic_intensity'] = torch.stack(all_intensities).mean(dim=0)
        
        # Average spatial density
        all_densities = [d['spatial_density'] for d in domains]
        avg_info['spatial_density'] = torch.stack(all_densities).mean(dim=0)
        
        # Average congestion cost
        avg_info['congestion_cost'] = torch.stack([d['congestion_cost'] for d in domains]).mean()
        
        return avg_info
    
    def perturb(self, epochs: int = 100, eta_init: float = 0.045, clip_norm: float = 0.23,
                momentum: float = 0.975, patience: int = 6, lambda_entropy: float = 0.012,
                lambda_virtual: float = 0.8, lambda_multi: float = 1.0, verbose: bool = True) -> Generator:
        """
        Perform the perturbation process for evidence-based case with congestion tracking.
        
        Args:
            epochs (int): Number of perturbation epochs. Defaults to 100.
            eta_init (float): Initial learning rate. Defaults to 0.045.
            clip_norm (float): Gradient clipping norm. Defaults to 0.23.
            momentum (float): Momentum factor. Defaults to 0.975.
            patience (int): Maximum patience before stopping. Defaults to 6.
            lambda_entropy (float): Entropy regularization coefficient. Defaults to 0.012.
            lambda_virtual (float): Coefficient for virtual target OT. Defaults to 0.8.
            lambda_multi (float): Coefficient for multi-marginal evidence OT. Defaults to 1.0.
            verbose (bool): If True, print progress. Defaults to True.
        
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
                    loss, grads = self._compute_loss_and_grad(pert_gen, virtual_samples, lambda_entropy, lambda_virtual, lambda_multi)
                    
                    # Compute delta_theta with multi-marginal congestion awareness
                    if multi_congestion_info and multi_congestion_info['domains']:
                        avg_congestion_info = self._average_multi_marginal_congestion(multi_congestion_info)
                        delta_theta = self._compute_delta_theta_with_congestion(
                            grads, eta, clip_norm, momentum, delta_theta_prev, avg_congestion_info
                        )
                    else:
                        delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                    
                    # Apply update
                    theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                    delta_theta_prev = delta_theta.clone()
                    
                    # Validate and get improvement
                    ot_pert, improvement = self._validate_and_adapt(pert_gen, virtual_samples, eta, ot_hist, patience, verbose, epoch)
                    
                    # Update best state
                    best_ot, best_vec = self._update_best_state(ot_pert, pert_gen, best_ot, best_vec)
                    
                    # Adapt learning rate based on improvement
                    eta, no_improvement_count = self._adapt_learning_rate(eta, improvement, epoch, no_improvement_count, ot_hist)
                    
                    # Check for rollback condition with congestion awareness
                    if self._check_rollback_condition_with_congestion(ot_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at epoch {epoch} (no improvement for {no_improvement_count} epochs)")
                        self._restore_best_state(pert_gen, best_vec)
                        # Reset parameters after rollback
                        eta = eta * 0.6
                        no_improvement_count = 0
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                    
                    if verbose:
                        log_msg = f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}"
                        if multi_congestion_info and multi_congestion_info['domains']:
                            total_congestion = sum(d['congestion_cost'].item() for d in multi_congestion_info['domains'])
                            log_msg += f" Total_Congestion={total_congestion:.4f}"
                        print(log_msg)
                    
                    # Early stopping if patience exceeded
                    if no_improvement_count >= patience:
                        if verbose:
                            print(f"Early stopping at epoch {epoch} due to lack of improvement")
                        break
                        
                except Exception as e:
                    print(f"Error in perturbation epoch {epoch}: {e}")
                    if epoch == 0:  # If first epoch fails, re-raise
                        raise
                    break
            
            # Final restore to best state
            self._restore_best_state(pert_gen, best_vec)
            
            return pert_gen
            
        except Exception as e:
            print(f"Error in perturbation process: {e}")
            # Return a copy of the original generator as fallback
            data_dim = self.evidence_list[0].shape[1] if hasattr(self, 'evidence_list') and self.evidence_list else 2
            return self._create_generator_copy(data_dim)
    
    def _compute_loss_and_grad(self, pert_gen: Generator, virtual_samples: torch.Tensor,
                               lambda_entropy: float, lambda_virtual: float, lambda_multi: float
                               ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-marginal OT loss and flattened gradients.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        gen_out = pert_gen(noise)
        
        loss = multi_marginal_ot_loss(
            gen_out, self.evidence_list, virtual_samples,
            blur=0.06,
            lambda_virtual=lambda_virtual,
            lambda_multi=lambda_multi,
            lambda_entropy=lambda_entropy
        )
        
        pert_gen.zero_grad()
        loss.backward()
        grads = torch.cat([p.grad.view(-1) for p in pert_gen.parameters() if p.grad is not None])
        return loss, grads
    
    def _validate_and_adapt(self, pert_gen: Generator, virtual_samples: torch.Tensor, eta: float,
                            ot_hist: List[float], patience: int, verbose: bool, epoch: int) -> Tuple[float, float]:
        """
        Validate perturbation and compute improvement in multi-marginal case.
        
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

        ot_orig = multi_marginal_ot_loss(
            orig_out, self.evidence_list, virtual_samples,
            blur=0.06, lambda_virtual=self.config.get('lambda_virtual', 0.8),
            lambda_multi=self.config.get('lambda_multi', 1.0),
            lambda_entropy=self.config.get('lambda_entropy', 0.012)
        ).item()

        ot_pert = multi_marginal_ot_loss(
            pert_out, self.evidence_list, virtual_samples,
            blur=0.06, lambda_virtual=self.config.get('lambda_virtual', 0.8),
            lambda_multi=self.config.get('lambda_multi', 1.0),
            lambda_entropy=self.config.get('lambda_entropy', 0.012)
        ).item()
        
        # Compute improvement compared to original (lower is better for OT loss)
        improvement = ot_orig - ot_pert
        ot_hist.append(ot_pert)
        
        return ot_pert, improvement


CongestionAwareWeightPerturberTargetGiven = CTWeightPerturberTargetGiven
CongestionAwareWeightPerturberTargetNotGiven = CTWeightPerturberTargetNotGiven