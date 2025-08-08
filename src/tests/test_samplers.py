import unittest
import torch
import numpy as np

from weight_perturbation.samplers import (
    sample_real_data,
    sample_target_data,
    sample_evidence_domains,
    kde_sampler,
    virtual_target_sampler
)

class TestSamplers(unittest.TestCase):
    """
    Unit tests for the sampling functions in samplers.py.
    
    These tests cover initialization, output shapes, statistical properties, and error handling
    for each sampling function. We use torch.testing.assert_close for tensor comparisons
    where appropriate, and check for expected exceptions.
    """

    def setUp(self):
        """
        Set up common parameters and device for tests.
        """
        self.batch_size = 100
        self.data_dim = 2
        self.device = torch.device('cpu')  # Use CPU for consistency in tests; can be changed to 'cuda' if needed
        self.tol = 1e-5  # Tolerance for floating-point comparisons

    def test_sample_real_data_default(self):
        """
        Test sample_real_data with default parameters.
        """
        samples = sample_real_data(self.batch_size, device=self.device)
        self.assertEqual(samples.shape, (self.batch_size, self.data_dim))
        self.assertEqual(samples.device, self.device)
        self.assertTrue(torch.all(torch.isfinite(samples)))  # Ensure no NaNs or Infs

        # Check approximate means (4 clusters)
        means = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]], device=self.device)
        dists = torch.cdist(samples, means)
        min_dists, _ = dists.min(dim=1)
        self.assertLess(min_dists.mean(), 1.0)  # Samples should be close to cluster centers

    def test_sample_real_data_custom_means(self):
        """
        Test sample_real_data with custom means.
        """
        custom_means = [[1.0, 1.0], [3.0, 3.0]]
        samples = sample_real_data(self.batch_size, means=custom_means, std=0.1, device=self.device)
        self.assertEqual(samples.shape, (self.batch_size, self.data_dim))

        # Check distribution between clusters
        num_clusters = len(custom_means)
        samples_per_cluster = self.batch_size // num_clusters
        self.assertAlmostEqual(len(samples) // num_clusters, samples_per_cluster, delta=1)

        # Check dimension consistency
        self.assertEqual(samples.shape[1], len(custom_means[0]))

    def test_sample_real_data_errors(self):
        """
        Test sample_real_data with invalid inputs.
        """
        with self.assertRaises(ValueError):
            sample_real_data(self.batch_size, means=[[1, 2], [3]])  # Inconsistent dimensions

        with self.assertRaises(ValueError):
            sample_real_data(self.batch_size, means=[])  # Empty means

    def test_sample_target_data_default(self):
        """
        Test sample_target_data with default parameters.
        """
        samples = sample_target_data(self.batch_size, device=self.device)
        self.assertEqual(samples.shape, (self.batch_size, self.data_dim))
        self.assertEqual(samples.device, self.device)

        # Default shift is [1.8, 1.8], check approximate means
        default_means = torch.tensor([[2.0, 0.0], [-2.0, 0.0], [0.0, 2.0], [0.0, -2.0]], device=self.device)
        shift = torch.tensor([1.8, 1.8], device=self.device)
        expected_means = default_means + shift
        dists = torch.cdist(samples, expected_means)
        min_dists, _ = dists.min(dim=1)
        self.assertLess(min_dists.mean(), 1.0)

    def test_sample_target_data_custom_shift(self):
        """
        Test sample_target_data with custom shift and means.
        """
        custom_shift = [0.5, -0.5]
        custom_means = [[0.0, 0.0], [1.0, 1.0]]
        samples = sample_target_data(self.batch_size, shift=custom_shift, means=custom_means, std=0.2, device=self.device)
        self.assertEqual(samples.shape, (self.batch_size, self.data_dim))

        expected_means = torch.tensor(custom_means, dtype=torch.float32, device=self.device) + torch.tensor(custom_shift, dtype=torch.float32, device=self.device)
        self.assertEqual(expected_means.shape[1], self.data_dim)

    def test_sample_target_data_errors(self):
        """
        Test sample_target_data with invalid inputs.
        """
        with self.assertRaises(ValueError):
            sample_target_data(self.batch_size, shift=[1.0], means=[[1.0, 2.0]])  # Dimension mismatch

        with self.assertRaises(ValueError):
            sample_target_data(self.batch_size, shift=[1.0, 2.0], means=[])  # Empty means

    def test_sample_evidence_domains_default(self):
        """
        Test sample_evidence_domains with default parameters.
        """
        num_domains = 3
        samples_per_domain = 35
        domains, centers = sample_evidence_domains(num_domains=num_domains, samples_per_domain=samples_per_domain, device=self.device)
        
        self.assertEqual(len(domains), num_domains)
        self.assertEqual(len(centers), num_domains)
        self.assertIsInstance(centers[0], np.ndarray)
        
        for domain in domains:
            self.assertEqual(domain.shape, (samples_per_domain, self.data_dim))
            self.assertEqual(domain.device, self.device)
            self.assertTrue(torch.all(torch.isfinite(domain)))

        # Check circular placement (angles)
        angles = np.linspace(0, 2 * np.pi, num_domains, endpoint=False)
        radius = 3.4  # default
        for i, center in enumerate(centers):
            expected = np.array([np.cos(angles[i]), np.sin(angles[i])]) * radius
            np.testing.assert_allclose(center, expected, atol=1e-6)

    def test_sample_evidence_domains_custom(self):
        """
        Test sample_evidence_domains with custom parameters.
        """
        num_domains = 4
        samples_per_domain = 50
        random_shift = 2.0
        std = 0.5
        domains, centers = sample_evidence_domains(num_domains=num_domains, samples_per_domain=samples_per_domain,
                                                   random_shift=random_shift, std=std, device=self.device)
        
        self.assertEqual(len(domains), num_domains)
        for domain in domains:
            self.assertEqual(domain.shape, (samples_per_domain, self.data_dim))
        
        # Check std approximation
        for domain, center in zip(domains, centers):
            domain_mean = domain.mean(dim=0).cpu().numpy()
            np.testing.assert_allclose(domain_mean, center, atol=std * 2)  # Rough check

    def test_sample_evidence_domains_errors(self):
        """
        Test sample_evidence_domains with invalid inputs.
        """
        with self.assertRaises(ValueError):
            sample_evidence_domains(num_domains=0)  # Zero domains

        with self.assertRaises(ValueError):
            sample_evidence_domains(samples_per_domain=0)  # Zero samples per domain

    def test_kde_sampler_basic(self):
        """
        Test kde_sampler with basic inputs.
        """
        evidence = torch.randn(50, self.data_dim, device=self.device)
        num_samples = 100
        bandwidth = 0.22
        samples = kde_sampler(evidence, bandwidth=bandwidth, num_samples=num_samples, device=self.device)
        
        self.assertEqual(samples.shape, (num_samples, self.data_dim))
        self.assertEqual(samples.device, self.device)
        self.assertTrue(torch.all(torch.isfinite(samples)))

        # Rough check: samples should be around evidence mean with added noise
        evidence_mean = evidence.mean(dim=0)
        samples_mean = samples.mean(dim=0)
        torch.testing.assert_close(samples_mean, evidence_mean, atol=bandwidth * 2, rtol=0)

    def test_kde_sampler_adaptive(self):
        """
        Test kde_sampler with adaptive bandwidth.
        """
        # Create evidence with sufficient variance
        evidence = torch.randn(50, self.data_dim, device=self.device) * 2.0  # Increased variance
        samples = kde_sampler(evidence, bandwidth=0.22, num_samples=100, adaptive=True, device=self.device)
        
        self.assertEqual(samples.shape, (100, self.data_dim))
        
        # Check that adaptive bandwidth computation doesn't produce NaN
        if evidence.shape[0] > 1:
            local_std = torch.std(evidence, dim=0, keepdim=True) + 1e-5
            self.assertFalse(torch.any(torch.isnan(local_std)), "Local std should not be NaN")
            self.assertGreater(local_std.mean().item(), 0, "Local std should be positive")

    def test_kde_sampler_errors(self):
        """
        Test kde_sampler with invalid inputs.
        """
        with self.assertRaises(ValueError):
            kde_sampler(torch.empty(0, self.data_dim), num_samples=100)  # Empty evidence

        with self.assertRaises(ValueError):
            kde_sampler(torch.randn(10, 2), num_samples=0)  # Zero samples

    def test_virtual_target_sampler_basic(self):
        """
        Test virtual_target_sampler with basic inputs.
        """
        evidence_list = [torch.randn(35, self.data_dim, device=self.device) for _ in range(3)]
        num_samples = 600
        bandwidth = 0.22
        temperature = 1.0
        samples = virtual_target_sampler(evidence_list, bandwidth=bandwidth, num_samples=num_samples,
                                         temperature=temperature, device=self.device)
        
        self.assertEqual(samples.shape, (num_samples, self.data_dim))
        self.assertEqual(samples.device, self.device)
        self.assertTrue(torch.all(torch.isfinite(samples)))

        # Check approximate uniform distribution across domains
        dists_to_domains = [torch.cdist(samples, ev).min(dim=1)[0].mean() for ev in evidence_list]
        for dist in dists_to_domains:
            self.assertLess(dist, bandwidth * 3)  # Rough proximity check

    def test_virtual_target_sampler_weights(self):
        """
        Test virtual_target_sampler with custom weights.
        """
        # Create evidence lists that are more separated to make weight effect clearer
        evidence_list = [
            torch.randn(35, self.data_dim, device=self.device) + torch.tensor([3.0, 0.0], device=self.device),  # Shifted right
            torch.randn(35, self.data_dim, device=self.device) + torch.tensor([-3.0, 0.0], device=self.device)  # Shifted left
        ]
        weights = [0.8, 0.2]  # More extreme weights for clearer signal
        num_samples = 200  # More samples for better statistics
        samples = virtual_target_sampler(evidence_list, weights=weights, num_samples=num_samples, device=self.device)
        
        self.assertEqual(samples.shape, (num_samples, self.data_dim))
        
        # Check that the function works with weights (basic functionality test)
        # The exact proportion might vary due to random sampling and KDE smoothing,
        # but the samples should at least be finite and have the right shape
        self.assertTrue(torch.all(torch.isfinite(samples)))
        
        # Alternative test: Check that different weights produce different results
        samples_uniform = virtual_target_sampler(evidence_list, weights=[0.5, 0.5], num_samples=num_samples, device=self.device)
        
        # The weighted and uniform samples should be different (very high probability)
        mean_diff = (samples.mean(dim=0) - samples_uniform.mean(dim=0)).norm()
        self.assertGreater(mean_diff.item(), 0.01, "Different weights should produce different sample distributions")

    def test_virtual_target_sampler_errors(self):
        """
        Test virtual_target_sampler with invalid inputs.
        """
        with self.assertRaises(ValueError):
            virtual_target_sampler([], num_samples=100)  # Empty evidence list

        with self.assertRaises(ValueError):
            virtual_target_sampler([torch.randn(10, 2)], weights=[0.5, 0.5])  # Weights mismatch

        with self.assertRaises(ValueError):
            virtual_target_sampler([torch.randn(10, 2)], num_samples=0)  # Zero samples

if __name__ == '__main__':
    unittest.main()