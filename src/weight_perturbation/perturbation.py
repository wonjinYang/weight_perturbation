import numpy as np
import torch
import torch.nn as nn
from typing import List, Optional, Dict, Tuple
from geomloss import SamplesLoss

from .models import Generator
from .samplers import sample_target_data, sample_evidence_domains, kde_sampler, virtual_target_sampler
from .losses import global_w2_loss_and_grad, multi_marginal_ot_loss
from .utils import parameters_to_vector, vector_to_parameters, load_config, plot_distributions

class WeightPerturberSection2:
    """
    Weight Perturber for Section 2: Target-given perturbation using global W2 gradient flow.
    
    This class performs weight perturbation on a pre-trained generator to align its output
    distribution with a given target distribution. It uses congested transport principles
    with gradient clipping, momentum, adaptive learning rate, and early stopping/rollback
    for stability.
    
    Args:
        generator (Generator): Pre-trained generator model.
        target_samples (torch.Tensor): Samples from the target distribution.
        config (Optional[Dict]): Configuration dictionary. If None, loads from default.yaml.
    """
    def __init__(self, generator: Generator, target_samples: torch.Tensor, config: Optional[Dict] = None):
        self.generator = generator
        self.target_samples = target_samples.to(generator.model[0].weight.device)  # Ensure device consistency
        self.device = next(generator.parameters()).device
        self.config = load_config() if config is None else config
        self.noise_dim = self.config.get('noise_dim', 2)
        self.eval_batch_size = self.config.get('eval_batch_size', 1600)

    def perturb(self, steps: int = 24, eta_init: float = 0.017, clip_norm: float = 0.12,
                momentum: float = 0.91, patience: int = 7, verbose: bool = True) -> Generator:
        """
        Perform the perturbation process.
        
        Args:
            steps (int): Number of perturbation steps. Defaults to 24.
            eta_init (float): Initial learning rate. Defaults to 0.017.
            clip_norm (float): Gradient clipping norm (congestion bound). Defaults to 0.12.
            momentum (float): Momentum factor for updates. Defaults to 0.91.
            patience (int): Patience for early stopping if W2 increases continuously. Defaults to 7.
            verbose (bool): If True, print progress. Defaults to True.
        
        Returns:
            Generator: Perturbed generator model.
        
        Raises:
            ValueError: If target_samples are insufficient or mismatched dimensions.
        
        Example:
            >>> perturber = WeightPerturberSection2(generator, target_samples)
            >>> perturbed_gen = perturber.perturb(steps=20)
        """
        if self.target_samples.numel() == 0:
            raise ValueError("Target samples must not be empty.")
        if self.target_samples.shape[1] != self.generator.model[-1].out_features:
            raise ValueError("Target samples dimension must match generator output dimension.")
        
        pert_gen = type(self.generator)().to(self.device)  # Create a copy
        pert_gen.load_state_dict(self.generator.state_dict())
        
        theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
        delta_theta_prev = torch.zeros_like(theta_prev)
        eta = eta_init
        w2_hist = []
        best_vec = None
        best_w2 = float('inf')
        stopped = False
        
        for step in range(steps):
            # Compute global W2 loss and gradients
            loss, grads = self._compute_loss_and_grad(pert_gen)
            
            # Compute delta_theta with momentum and clipping
            delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
            
            # Apply update
            theta_new = theta_prev + delta_theta
            vector_to_parameters(theta_new, pert_gen.parameters())
            delta_theta_prev = delta_theta.clone()
            theta_prev = theta_new.clone()
            
            # Validate and adapt
            w2_pert, improvement = self._validate_and_adapt(pert_gen, eta, w2_hist, patience, verbose, step)
            
            # Update best
            if w2_pert < best_w2:
                best_w2 = w2_pert
                best_vec = parameters_to_vector(pert_gen.parameters()).clone()
            
            # Early stopping check
            if len(w2_hist) > patience and all(w2_hist[-i-1] < w2_hist[-i-2] for i in range(1, patience)):
                if verbose:
                    print(f"Early stop/rollback at step {step} due to continuous W2 increase.")
                stopped = True
            
            if verbose:
                print(f"[{step:2d}] W2(Pert, Target)={w2_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}")
            
            if stopped:
                break
        
        # Restore best state
        if best_vec is not None:
            vector_to_parameters(best_vec, pert_gen.parameters())
        
        return pert_gen

    def _compute_loss_and_grad(self, pert_gen: Generator) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute global W2 loss and flattened gradients.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        loss, grads = global_w2_loss_and_grad(pert_gen, self.target_samples, noise)
        return loss, grads

    def _compute_delta_theta(self, grads: torch.Tensor, eta: float, clip_norm: float,
                             momentum: float, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute weight update delta_theta with momentum and clipping.
        
        Args:
            grads (torch.Tensor): Flattened gradients.
            eta (float): Learning rate.
            clip_norm (float): Clipping norm.
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

    def _validate_and_adapt(self, pert_gen: Generator, eta: float, w2_hist: List[float],
                            patience: int, verbose: bool, step: int) -> Tuple[float, float]:
        """
        Validate perturbation, adapt eta, and check for rollback.
        
        Args:
            pert_gen (Generator): Current perturbed generator.
            eta (float): Current learning rate.
            w2_hist (List[float]): History of W2 distances.
            patience (int): Patience for adaptation.
            verbose (bool): Print flag.
            step (int): Current step.
        
        Returns:
            tuple[float, float]: (w2_pert, improvement)
        """
        noise_eval = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        orig_out = self.generator(noise_eval).detach()
        pert_out = pert_gen(noise_eval).detach()
        w2_orig = compute_wasserstein_distance(orig_out, self.target_samples)
        w2_pert = compute_wasserstein_distance(pert_out, self.target_samples)
        improvement = w2_orig - w2_pert
        w2_hist.append(w2_pert.item())
        
        if step > 1 and improvement < 0:
            eta *= 0.54  # Reduce eta on degradation
        
        return w2_pert.item(), improvement.item()

class WeightPerturberSection3:
    """
    Weight Perturber for Section 3: Evidence-based perturbation using multi-marginal OT.
    
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
        self.generator = generator
        self.evidence_list = [ev.to(generator.model[0].weight.device) for ev in evidence_list]
        self.centers = centers
        self.device = next(generator.parameters()).device
        self.config = load_config() if config is None else config
        self.noise_dim = self.config.get('noise_dim', 2)
        self.eval_batch_size = self.config.get('eval_batch_size', 600)
        if len(self.evidence_list) == 0:
            raise ValueError("Evidence list must not be empty.")

    def perturb(self, epochs: int = 100, eta_init: float = 0.045, clip_norm: float = 0.23,
                momentum: float = 0.975, patience: int = 6, lambda_entropy: float = 0.012,
                verbose: bool = True) -> Generator:
        """
        Perform the perturbation process for evidence-based case.
        
        Args:
            epochs (int): Number of perturbation epochs. Defaults to 100.
            eta_init (float): Initial learning rate. Defaults to 0.045.
            clip_norm (float): Gradient clipping norm. Defaults to 0.23.
            momentum (float): Momentum factor. Defaults to 0.975.
            patience (int): Patience for early stopping. Defaults to 6.
            lambda_entropy (float): Entropy regularization coefficient. Defaults to 0.012.
            verbose (bool): If True, print progress. Defaults to True.
        
        Returns:
            Generator: Perturbed generator model.
        """
        pert_gen = type(self.generator)().to(self.device)
        pert_gen.load_state_dict(self.generator.state_dict())
        
        theta_prev = parameters_to_vector(pert_gen.parameters()).clone()
        delta_theta_prev = torch.zeros_like(theta_prev)
        eta = eta_init
        ot_hist = []
        best_vec = None
        best_ot = float('inf')
        stopped = False
        
        for epoch in range(epochs):
            # Estimate virtual target
            virtual_samples = self._estimate_virtual_target(self.evidence_list, epoch)
            
            # Compute multi-marginal OT loss and gradients
            loss, grads = self._compute_loss_and_grad(pert_gen, virtual_samples, lambda_entropy)
            
            # Compute delta_theta
            delta_theta = self._compute_delta_theta(grads, eta, clip_norm, momentum, delta_theta_prev)
            
            # Apply update
            theta_new = theta_prev + delta_theta
            vector_to_parameters(theta_new, pert_gen.parameters())
            delta_theta_prev = delta_theta.clone()
            theta_prev = theta_new.clone()
            
            # Validate and adapt
            ot_pert, improvement = self._validate_and_adapt(pert_gen, virtual_samples, eta, ot_hist, patience, verbose, epoch)
            
            # Update best
            if ot_pert < best_ot:
                best_ot = ot_pert
                best_vec = parameters_to_vector(pert_gen.parameters()).clone()
            
            # Early stopping check
            if len(ot_hist) > patience and all(ot_hist[-i-1] < ot_hist[-i-2] for i in range(1, patience)):
                if verbose:
                    print(f"Early stop/rollback at epoch {epoch} due to continuous OT loss increase.")
                stopped = True
            
            if verbose:
                print(f"[{epoch:2d}] OT(Pert, Virtual)={ot_pert:.4f} Improvement={improvement:.4f} eta={eta:.4f}")
            
            if stopped:
                break
        
        # Restore best state
        if best_vec is not None:
            vector_to_parameters(best_vec, pert_gen.parameters())
        
        return pert_gen

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
        virtuals = virtual_target_sampler(evidence_list, bandwidth=bandwidth.item(), num_samples=self.eval_batch_size, device=self.device.str)
        return virtuals

    def _compute_loss_and_grad(self, pert_gen: Generator, virtual_samples: torch.Tensor,
                               lambda_entropy: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute multi-marginal OT loss and flattened gradients.
        
        Returns:
            tuple[torch.Tensor, torch.Tensor]: (loss, grads)
        """
        pert_gen.train()
        noise = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        gen_out = pert_gen(noise)
        loss = multi_marginal_ot_loss(gen_out, self.evidence_list, blur=0.06, entropy_lambda=lambda_entropy)
        pert_gen.zero_grad()
        loss.backward()
        grads = torch.cat([p.grad.view(-1) for p in pert_gen.parameters() if p.grad is not None])
        return loss, grads

    def _compute_delta_theta(self, grads: torch.Tensor, eta: float, clip_norm: float,
                             momentum: float, prev_delta: torch.Tensor) -> torch.Tensor:
        """
        Compute weight update delta_theta with momentum and clipping for multi-marginal case.
        
        Args:
            grads (torch.Tensor): Flattened gradients.
            eta (float): Learning rate.
            clip_norm (float): Clipping norm.
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

    def _validate_and_adapt(self, pert_gen: Generator, virtual_samples: torch.Tensor, eta: float,
                            ot_hist: List[float], patience: int, verbose: bool, epoch: int) -> Tuple[float, float]:
        """
        Validate perturbation, adapt eta, and check for rollback in multi-marginal case.
        
        Args:
            pert_gen (Generator): Current perturbed generator.
            virtual_samples (torch.Tensor): Current virtual target samples.
            eta (float): Current learning rate.
            ot_hist (List[float]): History of OT losses.
            patience (int): Patience for adaptation.
            verbose (bool): Print flag.
            epoch (int): Current epoch.
        
        Returns:
            tuple[float, float]: (ot_pert, improvement)
        """
        noise_eval = torch.randn(self.eval_batch_size, self.noise_dim, device=self.device)
        orig_out = self.generator(noise_eval).detach()
        pert_out = pert_gen(noise_eval).detach()
        ot_orig = multi_marginal_ot_loss(orig_out, self.evidence_list).item()
        ot_pert = multi_marginal_ot_loss(pert_out, self.evidence_list).item()
        improvement = ot_orig - ot_pert
        ot_hist.append(ot_pert)
        
        if epoch > 1 and improvement < 0:
            eta *= 0.72  # Reduce eta on degradation
        
        return ot_pert, improvement
