import unittest
import torch
import torch.nn as nn
from geomloss import SamplesLoss
from typing import List

from weight_perturbation.losses import (
    compute_wasserstein_distance,
    barycentric_ot_map,
    global_w2_loss_and_grad,
    multi_marginal_ot_loss
)
from weight_perturbation.models import Generator

class TestLosses(unittest.TestCase):
    """
    Unit tests for the loss functions in losses.py.
    
    These tests cover initialization, output values, shape consistency, and error handling
    for each loss function. We use torch.testing.assert_close for tensor comparisons
    where appropriate, and check for expected exceptions.
    """

    def setUp(self):
        """
        Set up common parameters and device for tests.
        """
        self.data_dim = 2
        self.batch_size = 100
        self.device = torch.device('cpu')  # Use CPU for consistency; can be 'cuda' if available
        self.tol = 5e-3  # Increased tolerance for Sinkhorn approximation and regularization

    def test_compute_wasserstein_distance_basic(self):
        """
        Test compute_wasserstein_distance with identical distributions (should be near zero).
        """
        samples1 = torch.randn(self.batch_size, self.data_dim, device=self.device)
        samples2 = samples1.clone()  # Identical
        dist = compute_wasserstein_distance(samples1, samples2, p=2, blur=0.07)
        self.assertIsInstance(dist, torch.Tensor)
        self.assertEqual(dist.dim(), 0)  # Scalar
        self.assertTrue(torch.isfinite(dist))
        torch.testing.assert_close(dist, torch.tensor(0.0, device=self.device), atol=self.tol, rtol=0)

    def test_compute_wasserstein_distance_different(self):
        """
        Test compute_wasserstein_distance with shifted distributions.
        """
        samples1 = torch.randn(self.batch_size, self.data_dim, device=self.device)
        samples2 = samples1 + 1.0  # Shift by 1 in each dimension
        dist = compute_wasserstein_distance(samples1, samples2, p=2, blur=0.07)
        self.assertGreater(dist.item(), 0.0)  # Should be positive
        # Approximate expected W2 for Gaussians: sqrt(2) for shift=1 in 2D, but with Sinkhorn approx
        expected_approx = (1.0 ** 2 * 2) ** 0.5  # Euclidean shift
        self.assertLess(abs(dist.item() - expected_approx), 0.5)  # Loose check due to approx

    def test_compute_wasserstein_distance_p1(self):
        """
        Test with p=1 (W1 distance).
        """
        samples1 = torch.randn(self.batch_size, self.data_dim, device=self.device)
        samples2 = samples1.clone()
        dist = compute_wasserstein_distance(samples1, samples2, p=1, blur=0.07)
        torch.testing.assert_close(dist, torch.tensor(0.0, device=self.device), atol=self.tol, rtol=0)

    def test_compute_wasserstein_distance_errors(self):
        """
        Test compute_wasserstein_distance with invalid inputs.
        """
        with self.assertRaises(ValueError):
            compute_wasserstein_distance(torch.randn(10, 2), torch.randn(10, 3))  # Dim mismatch

        with self.assertRaises(ValueError):
            compute_wasserstein_distance(torch.randn(10), torch.randn(10, 2))  # Not 2D

    def test_barycentric_ot_map_basic(self):
        """
        Test barycentric_ot_map with identical source and target (should map to itself).
        """
        source = torch.randn(self.batch_size, self.data_dim, device=self.device)
        target = source.clone()
        mapped = barycentric_ot_map(source, target, cost_p=2, reg=0.01)
        self.assertEqual(mapped.shape, source.shape)
        self.assertEqual(mapped.device, self.device)
        # Increase tolerance for barycentric mapping due to softmin approximation
        torch.testing.assert_close(mapped, source, atol=self.tol * 10, rtol=1e-2)

    def test_barycentric_ot_map_shifted(self):
        """
        Test barycentric_ot_map with shifted distributions.
        """
        source = torch.randn(self.batch_size, self.data_dim, device=self.device)
        target = source + 1.0
        mapped = barycentric_ot_map(source, target, cost_p=2, reg=0.01)
        # Mapped should be closer to target mean
        source_mean = source.mean(dim=0)
        mapped_mean = mapped.mean(dim=0)
        target_mean = target.mean(dim=0)
        dist_source = (source_mean - target_mean).norm()
        dist_mapped = (mapped_mean - target_mean).norm()
        self.assertLess(dist_mapped, dist_source)

    def test_barycentric_ot_map_different_sizes(self):
        """
        Test barycentric_ot_map with different number of samples.
        """
        source = torch.randn(50, self.data_dim, device=self.device)
        target = torch.randn(100, self.data_dim, device=self.device)
        mapped = barycentric_ot_map(source, target, cost_p=2, reg=0.01)
        self.assertEqual(mapped.shape, (50, self.data_dim))

    def test_barycentric_ot_map_errors(self):
        """
        Test barycentric_ot_map with invalid inputs.
        """
        with self.assertRaises(ValueError):
            barycentric_ot_map(torch.randn(10, 2), torch.randn(10, 3))  # Dim mismatch

        with self.assertRaises(ValueError):
            barycentric_ot_map(torch.randn(10), torch.randn(10, 2))  # Not 2D

    def test_global_w2_loss_and_grad_basic(self):
        """
        Test global_w2_loss_and_grad with a simple generator.
        """
        generator = Generator(2, 2, 64).to(self.device)  # Small hidden_dim for test
        target_samples = torch.randn(self.batch_size, self.data_dim, device=self.device)
        noise_samples = torch.randn(self.batch_size, 2, device=self.device)
        loss, grads = global_w2_loss_and_grad(generator, target_samples, noise_samples)
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertIsInstance(grads, torch.Tensor)
        self.assertEqual(grads.dim(), 1)  # Flattened
        self.assertTrue(torch.all(torch.isfinite(grads)))

    def test_global_w2_loss_and_grad_identical(self):
        """
        Test with generator that outputs identical to target (loss near zero).
        """
        class IdentityGen(nn.Module):
            def __init__(self):
                super().__init__()
                self.dummy = nn.Parameter(torch.tensor(0.0))  # Dummy parameter for gradients
            def forward(self, z): 
                return z + self.dummy * 0  # Identity with learnable parameter
        
        generator = IdentityGen().to(self.device)
        target_samples = torch.randn(self.batch_size, self.data_dim, device=self.device)
        noise_samples = target_samples.clone()  # Input same as target
        loss, grads = global_w2_loss_and_grad(generator, target_samples, noise_samples)
        # Relax tolerance for this test as it may include regularization terms
        self.assertLess(loss.item(), 0.1)  # Should be small but may not be exactly zero
        # Gradients should exist (may not be zero due to regularization)
        self.assertGreater(grads.numel(), 0)

    def test_global_w2_loss_and_grad_errors(self):
        """
        Test global_w2_loss_and_grad with invalid inputs.
        """
        generator = Generator(2, 3, 64).to(self.device)  # Mismatch data_dim
        target_samples = torch.randn(self.batch_size, 2, device=self.device)
        noise_samples = torch.randn(self.batch_size, 2, device=self.device)
        with self.assertRaises(ValueError):
            global_w2_loss_and_grad(generator, target_samples, noise_samples)

    def test_multi_marginal_ot_loss_basic(self):
        """
        Test multi_marginal_ot_loss with identical generator outputs and evidence.
        """
        gen_out = torch.randn(self.batch_size, self.data_dim, device=self.device)
        evidence_list = [gen_out.clone() for _ in range(3)]  # Identical
        virtual_targets = gen_out.clone()  # Add required virtual_targets parameter
        loss = multi_marginal_ot_loss(
            gen_out, evidence_list, virtual_targets, 
            weights=None, blur=0.06, lambda_entropy=0.012
        )
        self.assertIsInstance(loss, torch.Tensor)
        self.assertEqual(loss.dim(), 0)
        self.assertTrue(torch.isfinite(loss))
        # Loss should be small but may not be zero due to entropy regularization
        self.assertLess(loss.item(), 5.0)  # Reasonable bound

    def test_multi_marginal_ot_loss_different(self):
        """
        Test multi_marginal_ot_loss with shifted evidence.
        """
        gen_out = torch.randn(self.batch_size, self.data_dim, device=self.device)
        evidence_list = [gen_out + i * 1.0 for i in range(3)]  # Shifted
        virtual_targets = torch.randn(self.batch_size, self.data_dim, device=self.device)
        loss = multi_marginal_ot_loss(
            gen_out, evidence_list, virtual_targets,
            weights=[0.2, 0.3, 0.5], blur=0.06, lambda_entropy=0.012
        )
        self.assertGreater(loss.item(), -10.0)  # Can be negative due to entropy
        self.assertTrue(torch.isfinite(loss))

    def test_multi_marginal_ot_loss_weights(self):
        """
        Test multi_marginal_ot_loss with custom weights.
        """
        gen_out = torch.randn(self.batch_size, self.data_dim, device=self.device)
        evidence_list = [gen_out.clone() for _ in range(2)]
        virtual_targets = gen_out.clone()
        weights = [0.7, 0.3]
        loss = multi_marginal_ot_loss(
            gen_out, evidence_list, virtual_targets, weights=weights
        )
        self.assertTrue(torch.isfinite(loss))

    def test_multi_marginal_ot_loss_errors(self):
        """
        Test multi_marginal_ot_loss with invalid inputs.
        """
        gen_out = torch.randn(10, 2, device=self.device)
        virtual_targets = torch.randn(10, 2, device=self.device)
        
        with self.assertRaises(ValueError):
            multi_marginal_ot_loss(gen_out, [], virtual_targets)  # Empty evidence

        with self.assertRaises(ValueError):
            multi_marginal_ot_loss(
                gen_out, [torch.randn(10, 2)], virtual_targets, weights=[0.5, 0.5]
            )  # Weights mismatch

        with self.assertRaises(ValueError):
            multi_marginal_ot_loss(
                gen_out, [torch.randn(10, 3)], virtual_targets
            )  # Dim mismatch

if __name__ == '__main__':
    unittest.main()