import torch
import os
import yaml
from typing import Iterable, Optional, List, Dict, Any
import numpy as np

# Handle matplotlib backend before importing pyplot
import matplotlib
# Set backend for headless environments
if os.environ.get('MPLBACKEND'):
    matplotlib.use(os.environ.get('MPLBACKEND'))
elif 'DISPLAY' not in os.environ:
    matplotlib.use('Agg')  # Use non-interactive backend for headless environments

import matplotlib.pyplot as plt

def parameters_to_vector(parameters: Iterable[torch.Tensor]) -> torch.Tensor:
    """
    Flatten an iterable of tensors (model parameters) into a single 1D vector.
    
    This function concatenates all parameters into a flat vector, which is useful
    for operations like vectorized updates in optimization or perturbation.
    
    Args:
        parameters (Iterable[torch.Tensor]): Iterable of tensors to flatten.
    
    Returns:
        torch.Tensor: 1D tensor containing all flattened parameters.
    
    Raises:
        ValueError: If parameters are empty or not all on the same device.
    
    Example:
        >>> params = list(model.parameters())
        >>> vec = parameters_to_vector(params)
        >>> vec.shape
        torch.Size([total_num_elements])
    """
    # Convert to list to handle generators properly
    param_list = list(parameters)
    
    if not param_list:
        raise ValueError("Parameters iterable must not be empty.")
    
    device = param_list[0].device
    if any(p.device != device for p in param_list):
        raise ValueError("All parameters must be on the same device.")
    
    # Filter out parameters with no elements and flatten
    vec = []
    for p in param_list:
        if p.numel() > 0:  # Only include parameters with elements
            vec.append(p.contiguous().view(-1))
    
    if not vec:
        raise ValueError("No valid parameters found (all parameters are empty).")
    
    return torch.cat(vec)

def vector_to_parameters(vec: torch.Tensor, parameters: Iterable[torch.Tensor]) -> None:
    """
    Restore a flat 1D vector back into the shapes of the given parameters.
    
    This function copies the values from the vector into the parameters in-place.
    It is useful for updating model parameters after vectorized operations.
    
    Args:
        vec (torch.Tensor): 1D tensor containing flattened parameters.
        parameters (Iterable[torch.Tensor]): Iterable of tensors to update in-place.
    
    Raises:
        ValueError: If the total size of parameters does not match vec's size,
                    or if vec and parameters are on different devices.
    
    Example:
        >>> params = list(model.parameters())
        >>> vec = parameters_to_vector(params)
        >>> vector_to_parameters(vec + 0.1, params)  # Perturbs parameters in-place
    """
    # Convert to list to handle generators properly
    param_list = list(parameters)
    
    if not param_list:
        raise ValueError("Parameters iterable must not be empty.")
    
    device = param_list[0].device
    if vec.device != device:
        raise ValueError("Vector and parameters must be on the same device.")
    
    pointer = 0
    for param in param_list:
        numel = param.numel()
        if numel > 0:  # Only process parameters with elements
            if pointer + numel > vec.numel():
                raise ValueError("Vector size does not match total parameters size.")
            param.data.copy_(vec[pointer : pointer + numel].view_as(param).data)
            pointer += numel
    
    if pointer != vec.numel():
        raise ValueError("Vector size does not match total parameters size.")

def plot_distributions(
    original: torch.Tensor,
    perturbed: torch.Tensor,
    target_or_virtual: torch.Tensor,
    evidence: Optional[List[torch.Tensor]] = None,
    title: str = "",
    save_path: Optional[str] = None,
    show: bool = False
) -> None:
    """
    Plot 2D distributions for visualization of original, perturbed, target/virtual, and optional evidence.
    
    This function creates a scatter plot to compare distributions. Assumes data is 2D (e.g., toy datasets).
    Evidence is plotted as separate clusters if provided.
    
    Args:
        original (torch.Tensor): Original generated samples (N x 2).
        perturbed (torch.Tensor): Perturbed generated samples (M x 2).
        target_or_virtual (torch.Tensor): Target or virtual target samples (P x 2).
        evidence (Optional[List[torch.Tensor]]): List of evidence domain tensors, each (Q x 2). Defaults to None.
        title (str): Plot title. Defaults to "".
        save_path (Optional[str]): Path to save the figure. If None, does not save. Defaults to None.
        show (bool): If True, display the plot. Defaults to False.
    
    Raises:
        ValueError: If any tensor is not 2D or has wrong dimension.
    
    Example:
        >>> plot_distributions(orig_samples, pert_samples, target_samples, evidence_list, title="Comparison")
    """
    tensors = [original, perturbed, target_or_virtual]
    if evidence:
        tensors.extend(evidence)
    
    for t in tensors:
        if t.dim() != 2 or t.shape[1] != 2:
            raise ValueError("All tensors must be 2D with shape (n_samples, 2).")
    
    # Convert to numpy for plotting
    orig_np = original.cpu().numpy()
    pert_np = perturbed.cpu().numpy()
    targ_np = target_or_virtual.cpu().numpy()
    
    plt.figure(figsize=(10, 8))
    plt.scatter(orig_np[:, 0], orig_np[:, 1], c='blue', label='Original', alpha=0.3, s=20)
    plt.scatter(pert_np[:, 0], pert_np[:, 1], c='red', label='Perturbed', alpha=0.3, s=20)
    plt.scatter(targ_np[:, 0], targ_np[:, 1], c='green', label='Target/Virtual', alpha=0.3, s=20)
    
    if evidence:
        colors = plt.cm.tab10(np.linspace(0, 1, len(evidence)))
        for i, ev in enumerate(evidence):
            ev_np = ev.cpu().numpy()
            plt.scatter(ev_np[:, 0], ev_np[:, 1], c=[colors[i]], label=f'Evidence {i+1}', alpha=0.5, s=30, marker='x')
    
    plt.title(title)
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        try:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        except Exception as e:
            print(f"Warning: Could not save plot to {save_path}: {e}")
    
    if show and 'DISPLAY' in os.environ:
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {e}")
    else:
        plt.close()

def load_config(config_path: str = 'configs/default.yaml') -> Dict[str, Any]:
    """
    Load hyperparameters from a YAML configuration file.
    
    This function reads a YAML file and returns a dictionary of configurations.
    It supports nested structures and provides default handling if file not found.
    
    Args:
        config_path (str): Path to the YAML config file. Defaults to 'configs/default.yaml'.
    
    Returns:
        Dict[str, Any]: Dictionary of loaded configurations.
    
    Raises:
        FileNotFoundError: If the config file does not exist.
        yaml.YAMLError: If the YAML file is invalid.
    
    Example:
        >>> config = load_config('my_config.yaml')
        >>> print(config['eta_init'])
        0.017
    """
    # Try to find config file in different locations
    possible_paths = [
        config_path,
        f'src/{config_path}',
        f'./{config_path}',
        os.path.join(os.path.dirname(__file__), '..', '..', config_path)
    ]
    
    config_file = None
    for path in possible_paths:
        if os.path.exists(path):
            config_file = path
            break
    
    if config_file is None:
        raise FileNotFoundError(f"Config file not found. Tried paths: {possible_paths}")
    
    with open(config_file, 'r') as f:
        try:
            config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise yaml.YAMLError(f"Invalid YAML in {config_file}: {e}")
    
    if config is None:
        config = {}
    
    # Default values if not present (example defaults)
    defaults = {
        'noise_dim': 2,
        'data_dim': 2,
        'hidden_dim': 256,
        'eta_init': 0.045,
        'clip_norm': 0.4,
        'momentum': 0.95,
        'patience': 15,
        'lambda_entropy': 0.012,
        'eval_batch_size': 600,
    }
    
    for key, val in defaults.items():
        config.setdefault(key, val)
    
    return config

def compute_device() -> torch.device:
    """
    Determine the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: The selected device.
    
    Example:
        >>> device = compute_device()
        >>> print(device)
        cuda
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility across torch, numpy, and random.
    
    Args:
        seed (int): Seed value. Defaults to 42.
    
    Example:
        >>> set_seed(123)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False