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
    """
    def __init__(self, noise_dim: int, data_dim: int, hidden_dim: int, activation=nn.LeakyReLU(0.2)):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(noise_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, data_dim)
        )

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
    """
    def __init__(self, data_dim: int, hidden_dim: int, activation=nn.LeakyReLU(0.2)):
        super(Critic, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, hidden_dim),
            activation,
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the critic.
        
        Args:
            x (torch.Tensor): Input data tensor of shape (batch_size, data_dim).
        
        Returns:
            torch.Tensor: Critic scores of shape (batch_size, 1).
        """
        return self.model(x)
