import unittest
import torch
import numpy as np
from typing import List

from weight_perturbation.perturbation import WeightPerturberTargetGiven, WeightPerturberTargetNotGiven
from weight_perturbation.models import Generator, Critic
from weight_perturbation.samplers import sample_target_data, sample_evidence_domains
from weight_perturbation.utils import parameters_to_vector, vector_to_parameters
from weight_perturbation.losses import compute_wasserstein_distance
from weight_perturbation.pretrain import pretrain_wgan_gp
from weight_perturbation.samplers import sample_real_data

class TestWeightPerturberTargetGiven(unittest.TestCase):
    """
    Unit tests for WeightPerturberTargetGiven class in perturbation.py.
    
    These tests cover initialization, perturbation process, internal methods,
    output validity, and error handling.
    """

    def setUp(self):
        """
        Set up common parameters, models, and data for tests.
        """
        self.noise_dim = 2
        self.data_dim = 2
        self.hidden_dim = 64  # Smaller for faster tests
        self.device = torch.device('cpu')  # Use CPU for consistency
        self.batch_size = 100
        self.target_samples = sample_target_data(self.batch_size, device=self.device)
        
        # Pretrain a small generator for testing
        self.generator = Generator(self.noise_dim, self.data_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.data_dim, self.hidden_dim).to(self.device)  # Fixed: use Critic class
        real_sampler = lambda bs: sample_real_data(bs, device=self.device)
        self.pretrained_gen, _ = pretrain_wgan_gp(
            self.generator, self.critic, real_sampler,
            epochs=5, batch_size=32, noise_dim=self.noise_dim, device=self.device, verbose=False
        )

    def test_initialization(self):
        """
        Test WeightPerturberTargetGiven initialization.
        """
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples)
        self.assertIsInstance(perturber.generator, Generator)
        self.assertEqual(perturber.target_samples.shape, (self.batch_size, self.data_dim))
        self.assertEqual(perturber.device, self.device)
        self.assertIn('noise_dim', perturber.config)
        self.assertIn('eval_batch_size', perturber.config)

    def test_initialization_custom_config(self):
        """
        Test initialization with custom config.
        """
        custom_config = {'noise_dim': 4, 'eval_batch_size': 200}
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples, config=custom_config)
        self.assertEqual(perturber.noise_dim, 4)
        self.assertEqual(perturber.eval_batch_size, 200)

    def test_perturb_basic(self):
        """
        Test perturb method with basic parameters.
        """
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples)
        perturbed_gen = perturber.perturb(steps=3, eta_init=0.01, clip_norm=0.1, momentum=0.9, patience=3, verbose=False)
        self.assertIsInstance(perturbed_gen, Generator)
        
        # Check if perturbation happened (parameters changed)
        orig_params = parameters_to_vector(self.pretrained_gen.parameters())
        pert_params = parameters_to_vector(perturbed_gen.parameters())
        # Allow some small changes
        self.assertFalse(torch.allclose(orig_params, pert_params, atol=1e-6))

    def test_compute_delta_theta(self):
        """
        Test _compute_delta_theta internal method.
        """
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples)
        grads = torch.randn(100, device=self.device)  # Fake grads
        eta = 0.01
        clip_norm = 0.1
        momentum = 0.9
        prev_delta = torch.zeros_like(grads)
        
        delta_theta = perturber._compute_delta_theta(grads, eta, clip_norm, momentum, prev_delta)
        self.assertEqual(delta_theta.shape, grads.shape)
        self.assertTrue(torch.all(torch.isfinite(delta_theta)))
        
        # Check clipping works
        large_grads = grads * 100
        delta_large = perturber._compute_delta_theta(large_grads, eta, clip_norm, momentum, prev_delta)
        self.assertTrue(torch.all(torch.isfinite(delta_large)))

    def test_validate_and_adapt(self):
        """
        Test _validate_and_adapt internal method.
        """
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples)
        pert_gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim).to(self.device)
        pert_gen.load_state_dict(self.pretrained_gen.state_dict())
        eta = 0.01
        w2_hist = [1.0, 0.9]  # Fake history
        patience = 3
        verbose = False
        step = 2
        
        w2_pert, improvement = perturber._validate_and_adapt(pert_gen, eta, w2_hist, patience, verbose, step)
        self.assertIsInstance(w2_pert, float)
        self.assertIsInstance(improvement, float)
        self.assertTrue(torch.isfinite(torch.tensor(w2_pert)))

    def test_perturb_errors(self):
        """
        Test perturb method with invalid inputs.
        """
        with self.assertRaises(ValueError):
            WeightPerturberTargetGiven(self.pretrained_gen, torch.empty(0, self.data_dim))

    def test_early_stopping(self):
        """
        Test early stopping functionality.
        """
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples)
        loss_hist = [1.0, 1.1, 1.2, 1.3]  # Increasing losses
        self.assertTrue(perturber._check_early_stopping(loss_hist, patience=3))
        
        loss_hist = [1.0, 0.9, 0.8, 0.7]  # Decreasing losses
        self.assertFalse(perturber._check_early_stopping(loss_hist, patience=3))

    def test_best_state_management(self):
        """
        Test best state update and restoration.
        """
        perturber = WeightPerturberTargetGiven(self.pretrained_gen, self.target_samples)
        pert_gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim).to(self.device)
        
        # Test update_best_state
        best_loss, best_vec = perturber._update_best_state(0.5, pert_gen, 1.0, None)
        self.assertEqual(best_loss, 0.5)
        self.assertIsNotNone(best_vec)
        
        # Test restore_best_state
        original_params = parameters_to_vector(pert_gen.parameters()).clone()
        # Modify parameters
        with torch.no_grad():
            for param in pert_gen.parameters():
                param.add_(0.1)
        
        # Restore
        perturber._restore_best_state(pert_gen, best_vec)
        restored_params = parameters_to_vector(pert_gen.parameters())
        torch.testing.assert_close(original_params, restored_params)

class TestWeightPerturberTargetNotGiven(unittest.TestCase):
    """
    Unit tests for WeightPerturberTargetNotGiven class in perturbation.py.
    
    These tests cover initialization, perturbation process, internal methods,
    output validity, and error handling.
    """

    def setUp(self):
        """
        Set up common parameters, models, and data for tests.
        """
        self.noise_dim = 2
        self.data_dim = 2
        self.hidden_dim = 64
        self.device = torch.device('cpu')
        self.num_domains = 2  # Smaller for tests
        self.evidence_list, self.centers = sample_evidence_domains(
            num_domains=self.num_domains, samples_per_domain=20, device=self.device
        )
        
        # Pretrain a small generator
        self.generator = Generator(self.noise_dim, self.data_dim, self.hidden_dim).to(self.device)
        self.critic = Critic(self.data_dim, self.hidden_dim).to(self.device)  # Fixed: use Critic class
        real_sampler = lambda bs: sample_real_data(bs, device=self.device)
        self.pretrained_gen, _ = pretrain_wgan_gp(
            self.generator, self.critic, real_sampler,
            epochs=5, batch_size=32, noise_dim=self.noise_dim, device=self.device, verbose=False
        )

    def test_initialization(self):
        """
        Test WeightPerturberTargetNotGiven initialization.
        """
        perturber = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers)
        self.assertIsInstance(perturber.generator, Generator)
        self.assertEqual(len(perturber.evidence_list), self.num_domains)
        self.assertEqual(perturber.device, self.device)
        self.assertIn('noise_dim', perturber.config)
        self.assertIn('eval_batch_size', perturber.config)

    def test_initialization_custom_config(self):
        """
        Test initialization with custom config.
        """
        custom_config = {'noise_dim': 4, 'eval_batch_size': 200}
        perturber = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers, config=custom_config)
        self.assertEqual(perturber.noise_dim, 4)
        self.assertEqual(perturber.eval_batch_size, 200)

    def test_perturb_basic(self):
        """
        Test perturb method with basic parameters.
        """
        perturber = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers)
        perturbed_gen = perturber.perturb(epochs=3, eta_init=0.01, clip_norm=0.1, momentum=0.9, patience=3, lambda_entropy=0.01, verbose=False)
        self.assertIsInstance(perturbed_gen, Generator)
        
        # Check if perturbation happened
        orig_params = parameters_to_vector(self.pretrained_gen.parameters())
        pert_params = parameters_to_vector(perturbed_gen.parameters())
        # Allow some small changes
        self.assertFalse(torch.allclose(orig_params, pert_params, atol=1e-6))

    def test_estimate_virtual_target(self):
        """
        Test _estimate_virtual_target internal method.
        """
        perturber = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers)
        epoch = 0
        bandwidth_base = 0.22
        virtuals = perturber._estimate_virtual_target(self.evidence_list, epoch, bandwidth_base)
        self.assertEqual(virtuals.shape[1], self.data_dim)
        self.assertGreaterEqual(virtuals.shape[0], 100)  # At least some samples
        self.assertTrue(torch.all(torch.isfinite(virtuals)))

    def test_compute_delta_theta(self):
        """
        Test _compute_delta_theta internal method.
        """
        perturber = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers)
        grads = torch.randn(100, device=self.device)
        eta = 0.01
        clip_norm = 0.1
        momentum = 0.9
        prev_delta = torch.zeros_like(grads)
        
        delta_theta = perturber._compute_delta_theta(grads, eta, clip_norm, momentum, prev_delta)
        self.assertEqual(delta_theta.shape, grads.shape)
        self.assertTrue(torch.all(torch.isfinite(delta_theta)))

    def test_validate_and_adapt(self):
        """
        Test _validate_and_adapt internal method.
        """
        perturber = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers)
        pert_gen = Generator(self.noise_dim, self.data_dim, self.hidden_dim).to(self.device)
        pert_gen.load_state_dict(self.pretrained_gen.state_dict())
        virtual_samples = torch.randn(100, self.data_dim, device=self.device)  # Fake
        eta = 0.01
        ot_hist = [1.0, 0.9]
        patience = 3
        verbose = False
        epoch = 2
        
        ot_pert, improvement = perturber._validate_and_adapt(pert_gen, virtual_samples, eta, ot_hist, patience, verbose, epoch)
        self.assertIsInstance(ot_pert, float)
        self.assertIsInstance(improvement, float)
        self.assertTrue(torch.isfinite(torch.tensor(ot_pert)))

    def test_perturb_errors(self):
        """
        Test perturb method with invalid inputs.
        """
        with self.assertRaises(ValueError):
            WeightPerturberTargetNotGiven(self.pretrained_gen, [], [])  # Empty evidence

    def test_inheritance_structure(self):
        """
        Test that both classes properly inherit from WeightPerturber.
        """
        from weight_perturbation.perturbation import WeightPerturber
        
        perturber1 = WeightPerturberTargetGiven(self.pretrained_gen, self.evidence_list[0])
        perturber2 = WeightPerturberTargetNotGiven(self.pretrained_gen, self.evidence_list, self.centers)
        
        self.assertIsInstance(perturber1, WeightPerturber)
        self.assertIsInstance(perturber2, WeightPerturber)
        
        # Test shared methods exist
        self.assertTrue(hasattr(perturber1, '_compute_delta_theta'))
        self.assertTrue(hasattr(perturber1, '_create_generator_copy'))
        self.assertTrue(hasattr(perturber1, '_check_early_stopping'))
        self.assertTrue(hasattr(perturber2, '_compute_delta_theta'))
        self.assertTrue(hasattr(perturber2, '_create_generator_copy'))
        self.assertTrue(hasattr(perturber2, '_check_early_stopping'))

    def test_backward_compatibility(self):
        """
        Test backward compatibility aliases.
        """
        from weight_perturbation.perturbation import WeightPerturberSection2, WeightPerturberSection3
        
        # Test aliases point to the new classes
        self.assertIs(WeightPerturberSection2, WeightPerturberTargetGiven)
        self.assertIs(WeightPerturberSection3, WeightPerturberTargetNotGiven)
        
        # Test they can be instantiated
        perturber2 = WeightPerturberSection2(self.pretrained_gen, self.evidence_list[0])
        perturber3 = WeightPerturberSection3(self.pretrained_gen, self.evidence_list, self.centers)
        
        self.assertIsInstance(perturber2, WeightPerturberTargetGiven)
        self.assertIsInstance(perturber3, WeightPerturberTargetNotGiven)

if __name__ == '__main__':
    unittest.main()