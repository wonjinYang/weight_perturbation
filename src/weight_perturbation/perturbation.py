import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from abc import ABC, abstractmethod
from geomloss import SamplesLoss

from .models import Generator
from .samplers import virtual_target_sampler
from .losses import global_w2_loss_and_grad, multi_marginal_ot_loss, compute_wasserstein_distance
from .utils import parameters_to_vector, vector_to_parameters, load_config

class WeightPerturber(ABC):
    """
    Abstract base class for Weight Perturbation strategies.
    
    This class defines the common interface and shared functionality for all weight perturbation
    methods. It handles parameter management, device consistency, configuration loading,
    and provides abstract methods for specific perturbation strategies.
    
    Args:
        generator (Generator): Pre-trained generator model.
        config (Optional[Dict]): Configuration dictionary. If None, loads from default.yaml.
    """
    
    def __init__(self, generator: Generator, config: Optional[Dict] = None):
        self.generator = generator
        self.device = next(generator.parameters()).device
        
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
        
        # Initialize common parameters
        self._initialize_common_params()
    
    def _get_default_config(self) -> Dict:
        """Get default configuration if loading fails."""
        return {
            'noise_dim': 2,
            'eval_batch_size': 1600,
            'eta_init': 0.017,
            'eta_min': 1e-5,  # 최소 학습률
            'eta_max': 0.1,   # 최대 학습률
            'eta_decay_factor': 0.8,  # 성능 악화시 decay factor
            'eta_boost_factor': 1.2,  # 성능 개선시 boost factor
            'clip_norm': 0.12,
            'momentum': 0.91,
            'patience': 7,
            'rollback_patience': 3,  # rollback을 위한 patience
            'improvement_threshold': 1e-4,  # 개선으로 간주하는 최소 임계값
        }
    
    def _initialize_common_params(self) -> None:
        """Initialize common parameters used across all perturbation methods."""
        pass
    
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
        delta_theta = -eta * grads
        norm = delta_theta.norm()
        max_norm = clip_norm * len(grads)  # Scale by approximate parameter count
        if norm > max_norm:
            delta_theta = delta_theta * (max_norm / (norm + 1e-8))
        delta_theta = momentum * prev_delta + (1 - momentum) * delta_theta
        return delta_theta
    
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
                           no_improvement_count: int) -> Tuple[float, int]:
        """
        Adapt learning rate based on improvement and step count.
        
        Args:
            eta (float): Current learning rate.
            improvement (float): Current improvement value.
            step (int): Current step/epoch.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            Tuple[float, int]: (new_eta, updated_no_improvement_count)
        """
        eta_min = self.config.get('eta_min', 1e-5)
        eta_max = self.config.get('eta_max', 0.1)
        eta_decay_factor = self.config.get('eta_decay_factor', 0.8)
        eta_boost_factor = self.config.get('eta_boost_factor', 1.05)
        improvement_threshold = self.config.get('improvement_threshold', 1e-4)
        
        if improvement > improvement_threshold:
            # Significant improvement: boost learning rate slightly
            new_eta = min(eta * eta_boost_factor, eta_max)
            new_no_improvement_count = 0
        elif improvement < -improvement_threshold:
            # Performance degradation: decay learning rate
            new_eta = max(eta * eta_decay_factor, eta_min)
            new_no_improvement_count = no_improvement_count + 1
        else:
            # Marginal change: slight decay to maintain stability
            new_eta = max(eta * 0.99, eta_min)
            new_no_improvement_count = no_improvement_count + 1
            
        return new_eta, new_no_improvement_count
    
    def _check_rollback_condition(self, loss_hist: List[float], 
                                 no_improvement_count: int) -> bool:
        """
        Check if rollback to best state should be triggered.
        
        Args:
            loss_hist (List[float]): History of loss values.
            no_improvement_count (int): Consecutive steps without improvement.
            
        Returns:
            bool: True if rollback should be triggered.
        """
        rollback_patience = self.config.get('rollback_patience', 3)
        
        # Trigger rollback if no improvement for several consecutive steps
        if no_improvement_count >= rollback_patience:
            return True
            
        # Also check if loss has been consistently increasing
        if len(loss_hist) >= rollback_patience:
            recent_losses = loss_hist[-rollback_patience:]
            if all(recent_losses[i] >= recent_losses[i-1] for i in range(1, len(recent_losses))):
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
        pass

class WeightPerturberTargetGiven(WeightPerturber):
    """
    Weight Perturber for target-given perturbation using global W2 gradient flow.
    
    This class performs weight perturbation on a pre-trained generator to align its output
    distribution with a given target distribution. It uses congested transport principles
    with gradient clipping, momentum, adaptive learning rate, and rollback mechanism
    for stability.
    
    Args:
        generator (Generator): Pre-trained generator model.
        target_samples (torch.Tensor): Samples from the target distribution.
        config (Optional[Dict]): Configuration dictionary. If None, loads from default.yaml.
    """
    
    def __init__(self, generator: Generator, target_samples: torch.Tensor, config: Optional[Dict] = None):
        super().__init__(generator, config)
        self.target_samples = target_samples.to(self.device)
        
        # Validate target samples
        if self.target_samples.numel() == 0:
            raise ValueError("Target samples must not be empty.")
    
    def perturb(self, steps: int = 24, eta_init: float = 0.017, clip_norm: float = 0.12,
                momentum: float = 0.91, patience: int = 7, verbose: bool = True) -> Generator:
        """
        Perform the perturbation process with adaptive learning rate and rollback.
        
        Args:
            steps (int): Number of perturbation steps. Defaults to 24.
            eta_init (float): Initial learning rate. Defaults to 0.017.
            clip_norm (float): Gradient clipping norm (congestion bound). Defaults to 0.12.
            momentum (float): Momentum factor for updates. Defaults to 0.91.
            patience (int): Maximum patience before stopping. Defaults to 7.
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
            best_vec = None
            best_w2 = float('inf')
            no_improvement_count = 0
            
            for step in range(steps):
                try:
                    # Compute global W2 loss and gradients
                    loss, grads = self._compute_loss_and_grad(pert_gen)
                    
                    # Compute delta_theta with momentum and clipping
                    delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                    
                    # Apply update
                    theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                    delta_theta_prev = delta_theta.clone()
                    
                    # Validate and get improvement
                    w2_pert, improvement = self._validate_and_adapt(pert_gen, eta, w2_hist, patience, verbose, step)
                    
                    # Update best state
                    best_w2, best_vec = self._update_best_state(w2_pert, pert_gen, best_w2, best_vec)
                    
                    # Adapt learning rate based on improvement
                    eta, no_improvement_count = self._adapt_learning_rate(eta, improvement, step, no_improvement_count)
                    
                    # Check for rollback condition
                    if self._check_rollback_condition(w2_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at step {step} (no improvement for {no_improvement_count} steps)")
                        self._restore_best_state(pert_gen, best_vec)
                        # Reset eta and counters after rollback
                        eta = eta_init * 0.5  # Start with reduced learning rate
                        no_improvement_count = 0
                        # Reset momentum
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                    
                    if verbose:
                        print(f"[{step:2d}] W2(Pert, Target)={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}")
                    
                    # Early stopping if patience exceeded
                    if no_improvement_count >= patience:
                        if verbose:
                            print(f"Early stopping at step {step} due to lack of improvement")
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
    
    def _compute_loss_and_grad(self, pert_gen: Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global W2 loss and flattened gradients.
        
        Returns:
            Tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        loss, grads = global_w2_loss_and_grad(pert_gen, self.target_samples, noise)
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
        
        w2_orig = compute_wasserstein_distance(orig_out, self.target_samples)
        w2_pert = compute_wasserstein_distance(pert_out, self.target_samples)
        
        # Compute improvement compared to original
        improvement = w2_orig.item() - w2_pert.item()
        w2_hist.append(w2_pert.item())
        
        return w2_pert.item(), improvement

class WeightPerturberTargetNotGiven(WeightPerturber):
    """
    Weight Perturber for evidence-based perturbation using multi-marginal OT.
    
    This class performs weight perturbation on a pre-trained generator using evidence domains
    to estimate a virtual target, then applies multi-marginal OT with entropy regularization.
    
    Args:
        generator (Generator): Pre-trained generator model.
        evidence_list (List[torch.Tensor]): List of evidence domain samples.
        centers (List[np.ndarray]): Centers of evidence domains.
        config (Optional[Dict]): Configuration dictionary. If None, loads from default.yaml.
    """
    
    def __init__(self, generator: Generator, evidence_list: List[torch.Tensor],
                 centers: List[np.ndarray], config: Optional[Dict] = None):
        super().__init__(generator, config)
        self.evidence_list = [ev.to(self.device) for ev in evidence_list]
        self.centers = centers
        
        # Override eval_batch_size for Section 3
        self.eval_batch_size = self.config.get('eval_batch_size', 600)
        
        if len(self.evidence_list) == 0:
            raise ValueError("Evidence list must not be empty.")
    
    def _get_default_config(self) -> Dict:
        """Get default configuration for evidence-based perturbation."""
        return {
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
        }
    
    def perturb(self, epochs: int = 100, eta_init: float = 0.045, clip_norm: float = 0.23,
                momentum: float = 0.975, patience: int = 6, lambda_entropy: float = 0.012,
                lambda_virtual: float = 0.8, lambda_multi: float = 1.0, verbose: bool = True) -> Generator:
        """
        Perform the perturbation process for evidence-based case with adaptive learning rate and rollback.
        
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
                    # Estimate virtual target
                    virtual_samples = self._estimate_virtual_target(self.evidence_list, epoch)
                    
                    # Compute multi-marginal OT loss and gradients
                    loss, grads = self._compute_loss_and_grad(pert_gen, virtual_samples, lambda_entropy, lambda_virtual, lambda_multi)
                    
                    # Compute delta_theta
                    delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
                    
                    # Apply update
                    theta_prev = self._apply_parameter_update(pert_gen, theta_prev, delta_theta)
                    delta_theta_prev = delta_theta.clone()
                    
                    # Validate and get improvement
                    ot_pert, improvement = self._validate_and_adapt(pert_gen, virtual_samples, eta, ot_hist, patience, verbose, epoch)
                    
                    # Update best state
                    best_ot, best_vec = self._update_best_state(ot_pert, pert_gen, best_ot, best_vec)
                    
                    # Adapt learning rate based on improvement
                    eta, no_improvement_count = self._adapt_learning_rate(eta, improvement, epoch, no_improvement_count)
                    
                    # Check for rollback condition
                    if self._check_rollback_condition(ot_hist, no_improvement_count):
                        if verbose:
                            print(f"Rollback triggered at epoch {epoch} (no improvement for {no_improvement_count} epochs)")
                        self._restore_best_state(pert_gen, best_vec)
                        # Reset eta and counters after rollback
                        eta = eta_init * 0.6  # Start with reduced learning rate
                        no_improvement_count = 0
                        # Reset momentum
                        delta_theta_prev = torch.zeros_like(delta_theta_prev)
                    
                    if verbose:
                        print(f"[{epoch:2d}] OT(Pert, Evidence)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}")
                    
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
    
    def _estimate_virtual_target(self, evidence_list: List[torch.Tensor], epoch: int,
                                bandwidth_base: float = 0.22) -> torch.Tensor:
        """
        Estimate virtual target using adaptive KDE and multi-domain weighting.
        
        Args:
            evidence_list (List[torch.Tensor]): Evidence domains.
            epoch (int): Current epoch for bandwidth adaptation.
            bandwidth_base (float): Base bandwidth. Defaults to 0.22.
        
        Returns:
            torch.Tensor: Virtual target samples.
        """
        bandwidth = bandwidth_base + 0.07 * torch.exp(torch.tensor(-epoch / 10.0, device=self.device))
        virtuals = virtual_target_sampler(
            evidence_list, 
            bandwidth=bandwidth.item(), 
            num_samples=self.eval_batch_size, 
            device=self.device
        )
        return virtuals
    
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
        # loss = multi_marginal_ot_loss(gen_out, self.evidence_list, blur=0.06, entropy_lambda=lambda_entropy)
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


# Backward compatibility aliases
WeightPerturberSection2 = WeightPerturberTargetGiven
WeightPerturberSection3 = WeightPerturberTargetNotGiven