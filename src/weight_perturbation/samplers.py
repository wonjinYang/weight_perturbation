import torch
import numpy as np
from typing import List, Optional, Union, Tuple

def sample_real_data(
    batch_size: int,
    means: Optional[List[Union[list, tuple, np.ndarray, torch.Tensor]]] = None,
    std: float = 0.4,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Sample from a real data distribution, such as multiple Gaussian clusters.
    
    This function generates samples from a mixture of Gaussians centered at specified means.
    If means are not provided, default to four clusters in 2D space.
    
    Args:
        batch_size (int): Number of samples to generate.
        means (Optional[List[Union[list, tuple, np.ndarray, torch.Tensor]]]): List of mean vectors for each cluster.
            If None, use default 2D means: [[2,0], [-2,0], [0,2], [0,-2]].
        std (float): Standard deviation for each Gaussian cluster. Defaults to 0.4.
        device (Union[str, torch.device]): Device to generate tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: Generated samples of shape (batch_size, data_dim).
    
    Raises:
        ValueError: If means are provided but inconsistent in dimension or type.
    
    Example:
        >>> samples = sample_real_data(100)
        >>> samples.shape
        torch.Size([100, 2])
    """
    device = torch.device(device)
    
    if means is None:
        means = [torch.tensor([2.0, 0.0]), torch.tensor([-2.0, 0.0]),
                 torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0])]
    else:
        if len(means) == 0:
            raise ValueError("Means list cannot be empty.")
        means = [torch.tensor(m, dtype=torch.float32) for m in means]
    
    num_clusters = len(means)
    data_dim = means[0].shape[0]
    if any(m.shape[0] != data_dim for m in means):
        raise ValueError("All means must have the same dimension.")
    
    samples_per_cluster = batch_size // num_clusters
    remainder = batch_size % num_clusters
    
    samples = []
    for i, mean in enumerate(means):
        n = samples_per_cluster + (1 if i < remainder else 0)
        cluster_samples = mean.to(device) + std * torch.randn(n, data_dim, device=device)
        samples.append(cluster_samples)
    
    return torch.cat(samples, dim=0)

def sample_target_data(
    batch_size: int,
    shift: Optional[Union[list, tuple, np.ndarray, torch.Tensor]] = None,
    means: Optional[List[Union[list, tuple, np.ndarray, torch.Tensor]]] = None,
    std: float = 0.4,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Sample from a target distribution, which is a shifted version of the real data clusters.
    
    This function generates samples similar to sample_real_data but applies a shift to the means.
    
    Args:
        batch_size (int): Number of samples to generate.
        shift (Optional[Union[list, tuple, np.ndarray, torch.Tensor]]): Shift vector to apply to all means.
            If None, use default shift [1.8, 1.8] for 2D.
        means (Optional[List[Union[list, tuple, np.ndarray, torch.Tensor]]]): Base means before shifting.
            If None, use default real data means.
        std (float): Standard deviation for each Gaussian cluster. Defaults to 0.4.
        device (Union[str, torch.device]): Device to generate tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: Generated target samples of shape (batch_size, data_dim).
    
    Raises:
        ValueError: If shift dimension mismatches means or other inconsistencies.
    
    Example:
        >>> target_samples = sample_target_data(100, shift=[1.0, 1.0])
        >>> target_samples.shape
        torch.Size([100, 2])
    """
    device = torch.device(device)
    
    if means is None:
        means = [torch.tensor([2.0, 0.0]), torch.tensor([-2.0, 0.0]),
                 torch.tensor([0.0, 2.0]), torch.tensor([0.0, -2.0])]
    else:
        if len(means) == 0:
            raise ValueError("Means list cannot be empty.")
        means = [torch.tensor(m, dtype=torch.float32) for m in means]
    
    if shift is None:
        shift = torch.tensor([1.8, 1.8], dtype=torch.float32)
    else:
        shift = torch.tensor(shift, dtype=torch.float32)
    
    data_dim = means[0].shape[0]
    if shift.shape[0] != data_dim:
        raise ValueError("Shift must match the dimension of means.")
    
    shifted_means = [m + shift for m in means]
    
    return sample_real_data(batch_size, means=shifted_means, std=std, device=device)

def sample_evidence_domains(
    num_domains: int = 3,
    samples_per_domain: int = 35,
    random_shift: float = 3.4,
    std: float = 0.7,
    device: Union[str, torch.device] = 'cpu'
) -> Tuple[List[torch.Tensor], List[np.ndarray]]:
    """
    Sample multiple evidence domains arranged in a circular pattern.
    
    Each domain is a cluster of samples around a center point placed circularly.
    
    Args:
        num_domains (int): Number of evidence domains to generate. Defaults to 3.
        samples_per_domain (int): Number of samples per domain. Defaults to 35.
        random_shift (float): Radius of the circular placement. Defaults to 3.4.
        std (float): Standard deviation for sampling around each center. Defaults to 0.7.
        device (Union[str, torch.device]): Device to generate tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
    
    Returns:
        Tuple[List[torch.Tensor], List[np.ndarray]]: List of domain samples (tensors) and list of centers (numpy arrays).
    
    Raises:
        ValueError: If num_domains or samples_per_domain are invalid.
    
    Example:
        >>> domains, centers = sample_evidence_domains(3, 35)
        >>> len(domains)
        3
    """
    if num_domains <= 0:
        raise ValueError("num_domains must be positive.")
    if samples_per_domain <= 0:
        raise ValueError("samples_per_domain must be positive.")
    
    device = torch.device(device)
    centers = []
    samples = []
    angles = np.linspace(0, 2 * np.pi, num_domains, endpoint=False)
    
    for a in angles:
        center = np.array([np.cos(a), np.sin(a)]) * random_shift
        centers.append(center)
        c_torch = torch.tensor(center, device=device, dtype=torch.float32)
        domain_samples = c_torch + std * torch.randn(samples_per_domain, c_torch.shape[0], device=device)
        samples.append(domain_samples)
    
    return samples, centers

def kde_sampler(
    evidence: torch.Tensor,
    bandwidth: float = 0.22,
    num_samples: int = 160,
    adaptive: bool = False,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Kernel Density Estimation (KDE) sampler using Gaussian kernels.
    
    Samples new points by selecting random evidence points as means and adding Gaussian noise.
    If adaptive, bandwidth is adjusted based on local density (simple heuristic).
    
    Args:
        evidence (torch.Tensor): Input evidence samples of shape (n_points, data_dim).
        bandwidth (float): Bandwidth for Gaussian kernel. Defaults to 0.22.
        num_samples (int): Number of samples to generate. Defaults to 160.
        adaptive (bool): If True, adapt bandwidth based on local variance. Defaults to False.
        device (Union[str, torch.device]): Device to generate tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: Generated samples of shape (num_samples, data_dim).
    
    Raises:
        ValueError: If evidence is empty or num_samples is invalid.
    
    Example:
        >>> evidence = torch.randn(50, 2)
        >>> samples = kde_sampler(evidence, num_samples=100)
        >>> samples.shape
        torch.Size([100, 2])
    """
    if evidence.numel() == 0:
        raise ValueError("Evidence tensor must not be empty.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    
    device = torch.device(device)
    evidence = evidence.to(device)
    
    n_points, data_dim = evidence.shape
    idx = torch.randint(0, n_points, (num_samples,), device=device)
    means = evidence[idx]
    
    if adaptive:
        # Simple adaptive: scale bandwidth by local std (heuristic)
        local_std = torch.std(means, dim=0, keepdim=True) + 1e-5
        bandwidth = bandwidth * local_std.mean().item()
    
    samples = means + bandwidth * torch.randn(num_samples, data_dim, device=device)
    return samples

def virtual_target_sampler(
    evidence_list: List[torch.Tensor],
    weights: Optional[List[float]] = None,
    bandwidth: float = 0.22,
    num_samples: int = 600,
    temperature: float = 1.0,
    device: Union[str, torch.device] = 'cpu'
) -> torch.Tensor:
    """
    Estimate a virtual target distribution via broadened KDE from multiple evidence domains.
    
    Weights domains and samples using KDE, with temperature for softening selection.
    
    Args:
        evidence_list (List[torch.Tensor]): List of evidence tensors, each (n_points, data_dim).
        weights (Optional[List[float]]): Weights for each domain. If None, uniform weights.
        bandwidth (float): Bandwidth for KDE sampling. Defaults to 0.22.
        num_samples (int): Total number of virtual samples to generate. Defaults to 600.
        temperature (float): Temperature for softmax-based domain selection. Defaults to 1.0.
        device (Union[str, torch.device]): Device to generate tensors on ('cpu' or 'cuda'). Defaults to 'cpu'.
    
    Returns:
        torch.Tensor: Virtual target samples of shape (num_samples, data_dim).
    
    Raises:
        ValueError: If evidence_list is empty or weights mismatch.
    
    Example:
        >>> evidence_list = [torch.randn(35, 2) for _ in range(3)]
        >>> virtuals = virtual_target_sampler(evidence_list, num_samples=600)
        >>> virtuals.shape
        torch.Size([600, 2])
    """
    num_domains = len(evidence_list)
    if num_domains == 0:
        raise ValueError("Evidence list must not be empty.")
    if num_samples <= 0:
        raise ValueError("num_samples must be positive.")
    
    device = torch.device(device)
    
    if weights is None:
        weights = [1.0 / num_domains] * num_domains
    elif len(weights) != num_domains:
        raise ValueError("Weights must match the number of domains.")
    
    # Softmax weights with temperature for selection probabilities
    weights_tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    probs = torch.softmax(weights_tensor / temperature, dim=0).cpu().numpy()
    
    choices = np.random.choice(num_domains, size=num_samples, p=probs)
    chunks = [np.sum(choices == i) for i in range(num_domains)]
    
    res = []
    for i, n in enumerate(chunks):
        if n > 0:
            domain_samples = kde_sampler(evidence_list[i], bandwidth=bandwidth, num_samples=n, device=device)
            res.append(domain_samples)
    
    if not res:  # Empty result case
        # Fallback: sample uniformly from the first domain
        res.append(kde_sampler(evidence_list[0], bandwidth=bandwidth, num_samples=num_samples, device=device))
    
    virtuals = torch.cat(res, dim=0)
    return virtuals