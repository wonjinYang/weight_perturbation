import torch
import torch.nn as nn

class Generator(nn.Module):
    """
    Generator model for the Weight Perturbation library.
    
    This is a simple feedforward neural network that maps noise vectors to data space.
    It uses multiple hidden layers with LeakyReLU activations for non-linearity.
    
    Args:
        noise_dim (int): Dimension of the input noise vector.
        data_dim (int): Dimension of the output data.
        hidden_dim (int): Number of units in the hidden layers.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.LeakyReLU(0.2).
    
    Raises:
        ValueError: If dimensions are invalid (zero or negative).
    """
    def __init__(self, noise_dim: int, data_dim: int, hidden_dim: int, activation=None):
        super(Generator, self).__init__()
        
        # Validate dimensions
        if noise_dim <= 0:
            raise ValueError("noise_dim must be positive")
        if data_dim <= 0:
            raise ValueError("data_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        
        # Store dimensions for debugging
        self.noise_dim = noise_dim
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, data_dim)
        )
        
        # Verify parameters were created
        param_count = sum(p.numel() for p in self.parameters())
        if param_count == 0:
            raise RuntimeError(f"Generator created with no parameters! dims: {noise_dim}, {data_dim}, {hidden_dim}")
        
        # Initialize parameters to ensure they're not empty
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for Linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the generator.
        
        Args:
            z (torch.Tensor): Input noise tensor of shape (batch_size, noise_dim).
        
        Returns:
            torch.Tensor: Generated samples of shape (batch_size, data_dim).
        """
        return self.model(z)

class Critic(nn.Module):
    """
    Critic model for the Weight Perturbation library.
    
    This is a feedforward neural network that evaluates the "realness" of input data.
    It uses multiple hidden layers with LeakyReLU activations.
    
    Args:
        data_dim (int): Dimension of the input data.
        hidden_dim (int): Number of units in the hidden layers.
        activation (nn.Module, optional): Activation function to use. Defaults to nn.LeakyReLU(0.2).
    
    Raises:
        ValueError: If dimensions are invalid (zero or negative).
    """
    def __init__(self, data_dim: int, hidden_dim: int, activation=None):
        super(Critic, self).__init__()
        
        # Validate dimensions
        if data_dim <= 0:
            raise ValueError("data_dim must be positive")
        if hidden_dim <= 0:
            raise ValueError("hidden_dim must be positive")
        
        if activation is None:
            activation = nn.LeakyReLU(0.2)
        
        # Store dimensions for debugging
        self.data_dim = data_dim
        self.hidden_dim = hidden_dim
        
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )
        
        # Verify parameters were created
        param_count = sum(p.numel() for p in self.parameters())
        if param_count == 0:
            raise RuntimeError(f"Critic created with no parameters! dims: {data_dim}, {hidden_dim}")
        
        # Initialize parameters
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights for Linear layers."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, 0.0, 0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic.
        
        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, data_dim).
        
        Returns:
            torch.Tensor: Critic scores of shape (batch_size, 1).
        """
        return self.model(x)