import unittest
import torch
import torch.nn as nn

from weight_perturbation.models import Generator, Critic

class TestGenerator(unittest.TestCase):
    """
    Unit tests for the Generator class in models.py.
    
    These tests cover initialization, forward pass, output shapes, and error handling.
    """

    def setUp(self):
        """
        Set up common parameters for tests.
        """
        self.noise_dim = 2
        self.data_dim = 2
        self.hidden_dim = 64  # Use 64 to match actual implementation
        self.batch_size = 4

    def test_initialization_default_activation(self):
        """
        Test Generator initialization with default activation (LeakyReLU(0.2)).
        """
        gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim)
        self.assertIsInstance(gen, nn.Module)
        self.assertIsInstance(gen.model, nn.Sequential)
        # Generator has 3 linear layers + 3 activations = 7 modules (no activation after last layer)
        self.assertEqual(len(gen.model), 7)
        self.assertIsInstance(gen.model[1], nn.LeakyReLU)
        self.assertEqual(gen.model[1].negative_slope, 0.2)

    def test_initialization_custom_activation(self):
        """
        Test Generator initialization with a custom activation function (e.g., ReLU).
        """
        custom_activation = nn.ReLU()
        gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim, activation=custom_activation)
        self.assertIsInstance(gen.model[1], nn.ReLU)
        self.assertIsInstance(gen.model[3], nn.ReLU)
        self.assertIsInstance(gen.model[5], nn.ReLU)

    def test_forward_pass(self):
        """
        Test the forward pass of the Generator with valid input.
        """
        gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim)
        noise = torch.randn(self.batch_size, self.noise_dim)
        output = gen(noise)
        self.assertEqual(output.shape, (self.batch_size, self.data_dim))
        self.assertTrue(torch.all(torch.isfinite(output)))  # Ensure no NaNs or Infs

    def test_forward_pass_different_dimensions(self):
        """
        Test forward pass with non-default dimensions.
        """
        custom_noise_dim = 4
        custom_data_dim = 3
        gen = Generator(custom_noise_dim, custom_data_dim, self.hidden_dim)
        noise = torch.randn(self.batch_size, custom_noise_dim)
        output = gen(noise)
        self.assertEqual(output.shape, (self.batch_size, custom_data_dim))

    def test_invalid_dimensions(self):
        """
        Test initialization with invalid dimensions (e.g., zero or negative).
        """
        with self.assertRaises(ValueError):
            Generator(0, self.data_dim, self.hidden_dim)  # noise_dim=0
        with self.assertRaises(ValueError):
            Generator(self.noise_dim, -1, self.hidden_dim)  # data_dim negative
        with self.assertRaises(ValueError):
            Generator(self.noise_dim, self.data_dim, 0)  # hidden_dim=0

    def test_parameter_count(self):
        """
        Test the number of trainable parameters in the Generator.
        """
        gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim)
        param_count = sum(p.numel() for p in gen.parameters() if p.requires_grad)
        
        # Expected for current architecture (3 hidden layers):
        # Layer 1: (2*64 + 64) = 192
        # Layer 2: (64*64 + 64) = 4160  
        # Layer 3: (64*64 + 64) = 4160
        # Layer 4: (64*2 + 2) = 130
        # Total: 192 + 4160 + 4160 + 130 = 8642
        expected = (self.noise_dim * self.hidden_dim + self.hidden_dim) + \
                  2 * (self.hidden_dim * self.hidden_dim + self.hidden_dim) + \
                  (self.hidden_dim * self.data_dim + self.data_dim)
        self.assertEqual(param_count, expected)

    def test_gradient_flow(self):
        """
        Test that gradients can flow through the Generator (requires_grad=True).
        """
        gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim)
        noise = torch.randn(self.batch_size, self.noise_dim, requires_grad=True)
        output = gen(noise)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(noise.grad)
        self.assertTrue(torch.all(torch.isfinite(noise.grad)))

class TestCritic(unittest.TestCase):
    """
    Unit tests for the Critic class in models.py.
    
    These tests cover initialization, forward pass, output/outputs, and error handling.
    """

    def setUp(self):
        """
        Set up common parameters for tests.
        """
        self.data_dim = 2
        self.hidden_dim = 64  # Use 64 to match actual implementation
        self.batch_size = 4

    def test_initialization_default_activation(self):
        """
        Test Critic initialization with default activation (LeakyReLU(0.2)).
        """
        crit = Critic(self.data_dim, self.hidden_dim)
        self.assertIsInstance(crit, nn.Module)
        self.assertIsInstance(crit.model, nn.Sequential)
        # Critic has 3 linear layers + 3 activations = 7 modules (no activation after last layer) 
        self.assertEqual(len(crit.model), 7)
        self.assertIsInstance(crit.model[1], nn.LeakyReLU)
        self.assertEqual(crit.model[1].negative_slope, 0.2)

    def test_initialization_custom_activation(self):
        """
        Test Critic initialization with a custom activation function (e.g., ReLU).
        """
        custom_activation = nn.ReLU()
        crit = Critic(self.data_dim, self.hidden_dim, activation=custom_activation)
        self.assertIsInstance(crit.model[1], nn.ReLU)
        self.assertIsInstance(crit.model[3], nn.ReLU)
        self.assertIsInstance(crit.model[5], nn.ReLU)

    def test_forward_pass(self):
        """
        Test the forward pass of the Critic with valid input.
        """
        crit = Critic(self.data_dim, self.hidden_dim)
        data = torch.randn(self.batch_size, self.data_dim)
        output = crit(data)
        self.assertEqual(output.shape, (self.batch_size, 1))
        self.assertTrue(torch.all(torch.isfinite(output)))

    def test_forward_pass_different_dimensions(self):
        """
        Test forward pass with non-default dimensions.
        """
        custom_data_dim = 3
        crit = Critic(custom_data_dim, self.hidden_dim)
        data = torch.randn(self.batch_size, custom_data_dim)
        output = crit(data)
        self.assertEqual(output.shape, (self.batch_size, 1))

    def test_invalid_dimensions(self):
        """
        Test initialization with invalid dimensions (e.g., zero or negative).
        """
        with self.assertRaises(ValueError):
            Critic(0, self.hidden_dim)  # data_dim=0
        with self.assertRaises(ValueError):
            Critic(self.data_dim, -1)  # hidden_dim negative

    def test_parameter_count(self):
        """
        Test the number of trainable parameters in the Critic.
        """
        crit = Critic(self.data_dim, self.hidden_dim)
        param_count = sum(p.numel() for p in crit.parameters() if p.requires_grad)
        
        # Expected for current architecture (3 hidden layers):
        # Layer 1: (2*64 + 64) = 192
        # Layer 2: (64*64 + 64) = 4160
        # Layer 3: (64*64 + 64) = 4160
        # Layer 4: (64*1 + 1) = 65
        # Total: 192 + 4160 + 4160 + 65 = 8577
        expected = (self.data_dim * self.hidden_dim + self.hidden_dim) + \
                  2 * (self.hidden_dim * self.hidden_dim + self.hidden_dim) + \
                  (self.hidden_dim * 1 + 1)
        self.assertEqual(param_count, expected)

    def test_gradient_flow(self):
        """
        Test that gradients can flow through the Critic (requires_grad=True).
        """
        crit = Critic(self.data_dim, self.hidden_dim)
        data = torch.randn(self.batch_size, self.data_dim, requires_grad=True)
        output = crit(data)
        loss = output.sum()
        loss.backward()
        self.assertIsNotNone(data.grad)
        self.assertTrue(torch.all(torch.isfinite(data.grad)))

if __name__ == '__main__':
    unittest.main()