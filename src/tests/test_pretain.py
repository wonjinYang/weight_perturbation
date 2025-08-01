import unittest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from typing import Callable

from weight_perturbation.pretrain import compute_gradient_penalty, pretrain_wgan_gp
from weight_perturbation.models import Generator, Critic
from weight_perturbation.samplers import sample_real_data

class TestComputeGradientPenalty(unittest.TestCase):
    """
    Unit tests for the compute_gradient_penalty function in pretrain.py.
    
    These tests cover basic computation, edge cases, and error handling.
    """

    def setUp(self):
        """
        Set up common parameters and models for tests.
        """
        self.data_dim = 2
        self.hidden_dim = 256
        self.batch_size = 8  # Small batch for fast tests
        self.device = torch.device('cpu')  # Use CPU for consistency
        self.critic = Critic(self.data_dim, self.hidden_dim).to(self.device)

    def test_gradient_penalty_identical_samples(self):
        """
        Test gradient penalty when real and fake samples are identical (should be zero).
        """
        real_samples = torch.randn(self.batch_size, self.data_dim, device=self.device)
        fake_samples = real_samples.clone()  # Identical
        gp = compute_gradient_penalty(self.critic, real_samples, fake_samples, self.device)
        self.assertIsInstance(gp, torch.Tensor)
        self.assertEqual(gp.dim(), 0)  # Scalar
        self.assertTrue(torch.isfinite(gp))
        self.assertAlmostEqual(gp.item(), 0.0, places=5)  # Should be exactly 0 for identical

    def test_gradient_penalty_different_samples(self):
        """
        Test gradient penalty with different real and fake samples (should be positive).
        """
        real_samples = torch.randn(self.batch_size, self.data_dim, device=self.device)
        fake_samples = real_samples + 1.0  # Shifted
        gp = compute_gradient_penalty(self.critic, real_samples, fake_samples, self.device)
        self.assertGreater(gp.item(), 0.0)  # Penalty should be positive if norm !=1

    def test_gradient_penalty_with_grad_flow(self):
        """
        Test that gradients are computed correctly (requires_grad=True).
        """
        real_samples = torch.randn(self.batch_size, self.data_dim, device=self.device, requires_grad=True)
        fake_samples = torch.randn(self.batch_size, self.data_dim, device=self.device, requires_grad=True)
        gp = compute_gradient_penalty(self.critic, real_samples, fake_samples, self.device)
        gp.backward()  # Should not raise error
        self.assertIsNotNone(real_samples.grad)
        self.assertIsNotNone(fake_samples.grad)
        self.assertTrue(torch.all(torch.isfinite(real_samples.grad)))
        self.assertTrue(torch.all(torch.isfinite(fake_samples.grad)))

    def test_gradient_penalty_shape_mismatch(self):
        """
        Test with mismatched shapes (should raise ValueError).
        """
        real_samples = torch.randn(self.batch_size, self.data_dim, device=self.device)
        fake_samples = torch.randn(self.batch_size, self.data_dim + 1, device=self.device)  # Mismatch
        with self.assertRaises(ValueError):
            compute_gradient_penalty(self.critic, real_samples, fake_samples, self.device)

    def test_gradient_penalty_empty_batch(self):
        """
        Test with empty batch (should handle or raise appropriately).
        """
        real_samples = torch.empty(0, self.data_dim, device=self.device)
        fake_samples = torch.empty(0, self.data_dim, device=self.device)
        gp = compute_gradient_penalty(self.critic, real_samples, fake_samples, self.device)
        self.assertEqual(gp.item(), 0.0)  # Empty case should give 0 or handle gracefully

class TestPretrainWganGp(unittest.TestCase):
    """
    Unit tests for the pretrain_wgan_gp function in pretrain.py.
    
    These tests cover training process, output models, loss progression, and error handling.
    Note: Training is tested with small epochs and batches for speed.
    """

    def setUp(self):
        """
        Set up common parameters and models for tests.
        """
        self.noise_dim = 2
        self.data_dim = 2
        self.hidden_dim = 64  # Small for fast tests
        self.batch_size = 4
        self.epochs = 2  # Small number for unit test
        self.device = 'cpu'
        self.generator = Generator(self.noise_dim, self.data_dim, self.hidden_dim)
        self.critic = Critic(self.data_dim, self.hidden_dim)
        self.real_sampler = lambda batch_size, **kwargs: sample_real_data(
            batch_size, means=None, std=0.4, device=self.device
        )

    def test_pretrain_basic(self):
        """
        Test basic pretraining: Check if models are returned and parameters updated.
        """
        trained_gen, trained_crit = pretrain_wgan_gp(
            self.generator, self.critic, self.real_sampler,
            epochs=self.epochs, batch_size=self.batch_size, lr=1e-3,
            gp_lambda=10.0, critic_iters=2, noise_dim=self.noise_dim,
            device=self.device, verbose=False
        )
        self.assertIsInstance(trained_gen, nn.Module)
        self.assertIsInstance(trained_crit, nn.Module)
        
        # Check if parameters changed (training happened)
        orig_gen_params = parameters_to_vector(self.generator.parameters())
        trained_gen_params = parameters_to_vector(trained_gen.parameters())
        self.assertFalse(torch.allclose(orig_gen_params, trained_gen_params))
        
        orig_crit_params = parameters_to_vector(self.critic.parameters())
        trained_crit_params = parameters_to_vector(trained_crit.parameters())
        self.assertFalse(torch.allclose(orig_crit_params, trained_crit_params))

    def test_pretrain_loss_progression(self):
        """
        Test if losses are computed and training progresses (e.g., losses decrease).
        """
        # Run training and capture losses if verbose=True, but since verbose=False, we can patch print or check indirectly
        trained_gen, trained_crit = pretrain_wgan_gp(
            self.generator, self.critic, self.real_sampler,
            epochs=self.epochs, batch_size=self.batch_size, lr=1e-3,
            gp_lambda=10.0, critic_iters=2, noise_dim=self.noise_dim,
            device=self.device, verbose=True  # Enable for testing
        )
        # Indirect check: After training, critic should distinguish real vs fake better
        real_samples = self.real_sampler(self.batch_size).to(self.device)
        noise = torch.randn(self.batch_size, self.noise_dim, device=self.device)
        fake_samples = trained_gen(noise).detach()
        
        crit_real = trained_crit(real_samples).mean().item()
        crit_fake = trained_crit(fake_samples).mean().item()
        self.assertGreater(crit_real, crit_fake)  # Critic should score real higher than fake

    def test_pretrain_device_consistency(self):
        """
        Test if models are moved to the specified device.
        """
        trained_gen, trained_crit = pretrain_wgan_gp(
            self.generator, self.critic, self.real_sampler,
            epochs=1, batch_size=self.batch_size, device=self.device, verbose=False
        )
        self.assertEqual(next(trained_gen.parameters()).device.type, self.device)
        self.assertEqual(next(trained_crit.parameters()).device.type, self.device)

    def test_pretrain_errors(self):
        """
        Test pretrain_wgan_gp with invalid inputs.
        """
        # Mismatched device
        self.generator.to('cuda' if self.device == 'cpu' else 'cpu')
        with self.assertRaises(ValueError):
            pretrain_wgan_gp(
                self.generator, self.critic, self.real_sampler,
                epochs=1, batch_size=self.batch_size, device=self.device, verbose=False
            )
        
        # Invalid sampler output
        def invalid_sampler(batch_size, **kwargs):
            return torch.randn(batch_size, self.data_dim + 1, device=self.device)  # Wrong dim
        with self.assertRaises(ValueError):
            pretrain_wgan_gp(
                self.generator, self.critic, invalid_sampler,
                epochs=1, batch_size=self.batch_size, device=self.device, verbose=False
            )

    def test_pretrain_with_custom_params(self):
        """
        Test pretraining with custom hyperparameters.
        """
        custom_lr = 5e-4
        custom_betas = (0.0, 0.9)
        custom_gp_lambda = 5.0
        custom_critic_iters = 3
        
        trained_gen, trained_crit = pretrain_wgan_gp(
            self.generator, self.critic, self.real_sampler,
            epochs=self.epochs, batch_size=self.batch_size, lr=custom_lr,
            betas=custom_betas, gp_lambda=custom_gp_lambda, critic_iters=custom_critic_iters,
            noise_dim=self.noise_dim, device=self.device, verbose=False
        )
        # Check if optimizers were set correctly (indirectly via training)
        self.assertIsInstance(trained_gen, nn.Module)
        self.assertIsInstance(trained_crit, nn.Module)

def parameters_to_vector(parameters):
    return torch.cat([p.view(-1) for p in parameters])

if __name__ == '__main__':
    unittest.main()
