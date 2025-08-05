import torch
import torch.nn.functional as F
from geomloss import SamplesLoss
from typing import Callable, List, Optional, Tuple

def compute_wasserstein_distance(
    samples1: torch.Tensor,
    samples2: torch.Tensor,
    p: int = 2,
    blur: float = 0.07
) -> torch.Tensor:
    """
    Compute the Wasserstein-p distance between two sets of samples using Sinkhorn divergence.
    
    This function uses the geomloss library to approximate the Wasserstein distance
    via entropic regularization (Sinkhorn algorithm). It's efficient for empirical distributions.
    
    Args:
        samples1 (torch.Tensor): First set of samples, shape (n_samples1, data_dim).
        samples2 (torch.Tensor): Second set of samples, shape (n_samples2, data_dim).
        p (int): Power of the Wasserstein distance (e.g., 1 for W1, 2 for W2). Defaults to 2.
        blur (float): Entropic regularization parameter (smaller values give sharper approximations). Defaults to 0.07.
    
    Returns:
        torch.Tensor: Scalar Wasserstein-p distance.
    
    Raises:
        ValueError: If samples have incompatible shapes or dimensions.
    
    Example:
        >>> s1 = torch.randn(100, 2)
        >>> s2 = torch.randn(100, 2)
        >>> dist = compute_wasserstein_distance(s1, s2, p=2)
        >>> dist.shape
        torch.Size([])
    """
    if samples1.dim() != 2 or samples2.dim() != 2:
        raise ValueError("Samples must be 2D tensors (n_samples, data_dim).")
    if samples1.shape[1] != samples2.shape[1]:
        raise ValueError("Samples must have the same data dimension.")
    
    wasserstein = SamplesLoss(loss="sinkhorn", p=p, blur=blur)
    return wasserstein(samples1, samples2)

def barycentric_ot_map(
    source_samples: torch.Tensor,
    target_samples: torch.Tensor,
    cost_p: int = 2,
    reg: float = 0.01
) -> torch.Tensor:
    """
    Compute the barycentric optimal transport map from source to target samples.
    
    This function approximates the OT map using a simple nearest-neighbor matching
    based on cost minimization (e.g., Euclidean distance). For more accurate barycentric
    projection, advanced libraries like POT can be integrated, but this provides a fast approximation.
    
    Args:
        source_samples (torch.Tensor): Source samples, shape (n_source, data_dim).
        target_samples (torch.Tensor): Target samples, shape (n_target, data_dim).
        cost_p (int): Power for the cost function (e.g., 2 for squared Euclidean). Defaults to 2.
        reg (float): Small regularization to avoid division by zero or instability. Defaults to 0.01.
    
    Returns:
        torch.Tensor: Mapped source samples to target space, shape (n_source, data_dim).
    
    Raises:
        ValueError: If samples have incompatible shapes or dimensions.
    
    Note:
        This is a simplified implementation; for entropic OT maps, consider using POT's ot.mapping.
    
    Example:
        >>> source = torch.randn(50, 2)
        >>> target = torch.randn(50, 2)
        >>> mapped = barycentric_ot_map(source, target)
        >>> mapped.shape
        torch.Size([50, 2])
    """
    if source_samples.dim() != 2 or target_samples.dim() != 2:
        raise ValueError("Samples must be 2D tensors (n_samples, data_dim).")
    if source_samples.shape[1] != target_samples.shape[1]:
        raise ValueError("Samples must have the same data dimension.")
    
    # Compute cost matrix (e.g., squared Euclidean for p=2)
    cost_matrix = torch.cdist(source_samples, target_samples, p=cost_p)
    
    # Improved softmin for approximate barycentric mapping with better temperature control
    temperature = reg * 0.5  # More aggressive temperature for sharper mapping
    weights = F.softmin(cost_matrix / (temperature + 1e-8), dim=1)
    
    # Barycentric projection: weighted average of target points
    mapped = torch.matmul(weights, target_samples)
    
    return mapped

def global_w2_loss_and_grad(
    generator: torch.nn.Module,
    target_samples: torch.Tensor,
    noise_samples: torch.Tensor,
    map_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = barycentric_ot_map,
    use_direct_w2: bool = True,
    w2_weight: float = 0.7,
    map_weight: float = 0.3
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the global W2 loss and gradients for Section 2 perturbation with improved loss formulation.
    
    This function generates samples from the generator, computes both direct W2 loss and 
    barycentric OT map loss, combining them for better convergence properties.
    
    Args:
        generator (torch.nn.Module): Pre-trained generator model.
        target_samples (torch.Tensor): Target distribution samples, shape (n_target, data_dim).
        noise_samples (torch.Tensor): Input noise for generator, shape (batch_size, noise_dim).
        map_fn (Callable): Function to compute OT map (defaults to barycentric_ot_map).
        use_direct_w2 (bool): Whether to include direct W2 loss. Defaults to True.
        w2_weight (float): Weight for direct W2 loss. Defaults to 0.7.
        map_weight (float): Weight for mapping loss. Defaults to 0.3.
    
    Returns:
        Tuple[torch.Tensor, torch.Tensor]: (loss scalar, flattened gradients tensor).
    
    Raises:
        ValueError: If shapes are incompatible or generator output mismatches target dim.
    
    Example:
        >>> gen = Generator(2, 2, 256)
        >>> target = torch.randn(100, 2)
        >>> noise = torch.randn(100, 2)
        >>> loss, grads = global_w2_loss_and_grad(gen, target, noise)
    """
    generator.train()
    
    # Enable gradients for parameters
    for param in generator.parameters():
        param.requires_grad_(True)
    
    gen_out = generator(noise_samples)
    
    if gen_out.shape[1] != target_samples.shape[1]:
        raise ValueError("Generator output dimension must match target dimension.")
    
    total_loss = 0.0
    
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
    
    # Add regularization to prevent mode collapse
    if gen_out.shape[0] > 1:
        # Diversity regularization: encourage spread in generated samples
        gen_cov = torch.cov(gen_out.t())
        diversity_loss = -torch.logdet(gen_cov + 1e-6 * torch.eye(gen_out.shape[1], device=gen_out.device))
        total_loss += 0.01 * diversity_loss
    
    # Compute gradients
    generator.zero_grad()
    
    try:
        total_loss.backward()
        grads = torch.cat([p.grad.view(-1) for p in generator.parameters() if p.grad is not None])
        
        if grads.numel() == 0:
            # Fallback: create dummy gradients if none computed
            total_params = sum(p.numel() for p in generator.parameters())
            grads = torch.zeros(total_params, device=total_loss.device)
            
    except RuntimeError as e:
        print(f"Warning: Gradient computation failed: {e}")
        # Fallback: create dummy gradients
        total_params = sum(p.numel() for p in generator.parameters())
        grads = torch.zeros(total_params, device=total_loss.device)
    
    return total_loss.detach(), grads

def multi_marginal_ot_loss(
    generator_outputs: torch.Tensor,
    evidence_list: List[torch.Tensor],
    virtual_targets: torch.Tensor,
    weights: Optional[List[float]] = None,
    blur: float = 0.06,
    lambda_virtual: float = 0.8,
    lambda_multi: float = 1.0,
    lambda_entropy: float = 0.012
) -> torch.Tensor:
    """
    Compute the multi-marginal OT loss with entropy regularization for Section 3.
    
    This function approximates multi-marginal Wasserstein loss by averaging pairwise
    Sinkhorn losses between generator outputs and each evidence domain, with optional
    weights and entropy regularization to prevent mode collapse.
    
    Args:
        generator_outputs (torch.Tensor): Generated samples, shape (batch_size, data_dim).
        evidence_list (List[torch.Tensor]): List of evidence tensors, each (n_ev, data_dim).
        virtual_targets (torch.Tensor): Virtual target samples for additional regularization.
        weights (Optional[List[float]]): Weights for each evidence domain. Defaults to uniform.
        blur (float): Entropic blur for Sinkhorn. Defaults to 0.06.
        lambda_virtual (float): Coefficient for virtual target OT loss regularization. Defaults to 0.8.
        lambda_multi (float): Coefficient for multi-marginal evidence OT loss regularization. Defaults to 1.0.
        lambda_entropy (float): Coefficient for entropy regularization. Defaults to 0.012.

    Returns:
        torch.Tensor: Scalar multi-marginal OT loss.
    
    Raises:
        ValueError: If evidence_list is empty or weights mismatch.
    
    Note:
        For exact multi-marginal OT, more advanced solvers (e.g., via POT) could be used,
        but this provides an efficient approximation.
    
    Example:
        >>> gen_out = torch.randn(100, 2)
        >>> ev_list = [torch.randn(50, 2) for _ in range(3)]
        >>> vt = torch.randn(100, 2)
        >>> loss = multi_marginal_ot_loss(gen_out, ev_list, vt)
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

    # Multi-marginal evidence loss
    loss_multi = 0.0
    for i, evidence in enumerate(evidence_list):
        if generator_outputs.shape[1] != evidence.shape[1]:
            raise ValueError("All tensors must have the same data dimension.")
        pairwise_loss = sinkhorn(generator_outputs, evidence)
        loss_multi += weights[i] * pairwise_loss
    
    # Entropy regularization (e.g., via covariance determinant for diversity)
    data_dim = generator_outputs.shape[1]
    cov = torch.cov(generator_outputs.t()) + torch.eye(data_dim, device=generator_outputs.device) * 1e-5
    eigenvals = torch.linalg.eigvals(cov).real
    eigenvals = torch.clamp(eigenvals, min=1e-8)
    log_det = torch.sum(torch.log(eigenvals))
    entropy_reg = -lambda_entropy * log_det

    total_loss = lambda_virtual * loss_virtual + lambda_multi * loss_multi + entropy_reg
    return total_loss