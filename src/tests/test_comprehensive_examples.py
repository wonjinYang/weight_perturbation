"""
Comprehensive test example for the Weight Perturbation library.

This script demonstrates all major features and serves as an integration test.
It has been enhanced with:
- Additional unit tests using unittest framework
- More detailed performance benchmarking
- Expanded error handling tests
- Integration with advanced congestion features
- Visualization tests with sample plots
- Full report generation including system info
- Command-line arguments for customization
- Logging with levels
- Mock data for isolated testing
- Comprehensive documentation in docstrings
- Type hints where appropriate
- Environment checks and recommendations
"""

import torch
from torch import nn
import argparse
import time
from pathlib import Path
import sys
import platform
import logging
import unittest
import io
from contextlib import redirect_stdout
from typing import Dict

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Basic imports that should always work
try:
    from weight_perturbation import (
        Generator, Critic,
        sample_real_data, sample_target_data, sample_evidence_domains,
        pretrain_wgan_gp, compute_wasserstein_distance,
        WeightPerturberTargetGiven, WeightPerturberTargetNotGiven,
        plot_distributions, set_seed, compute_device, get_version_info
    )
    BASIC_IMPORT_SUCCESS = True
except ImportError as e:
    logger.error(f"❌ Basic import failed: {e}")
    BASIC_IMPORT_SUCCESS = False

# Advanced imports that may not be available
try:
    from weight_perturbation import (
        CTWeightPerturberTargetGiven, CTWeightPerturberTargetNotGiven,
        CongestionTracker, compute_spatial_density, compute_traffic_flow,
        SobolevConstrainedCritic, check_theoretical_support
    )
    ADVANCED_IMPORT_SUCCESS = True
except ImportError as e:
    logger.warning(f"⚠️ Advanced imports not available: {e}")
    ADVANCED_IMPORT_SUCCESS = False

def test_basic_functionality():
    """Test basic functionality that should always work."""
    logger.info("\n🔧 Testing Basic Functionality...")
    if not BASIC_IMPORT_SUCCESS:
        logger.error("❌ Cannot test basic functionality - imports failed")
        return False

    try:
        # Test device computation
        device = compute_device()
        logger.info(f"✓ Device: {device}")

        # Test seed setting
        set_seed(42)
        logger.info("✓ Seed set")

        # Test version info
        try:
            version_info = get_version_info()
            logger.info(f"✓ Version info: {version_info}")
        except:
            logger.warning("⚠️ Version info not available")

        # Test model creation
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
        critic = Critic(data_dim=2, hidden_dim=64)
        logger.info("✓ Models created")

        # Test data sampling
        real_data = sample_real_data(100, device=device)
        target_data = sample_target_data(100, shift=[1.0, 1.0], device=device)
        evidence_list, centers = sample_evidence_domains(num_domains=2, samples_per_domain=50, device=device)
        logger.info("✓ Data sampling")

        # Test distance computation
        w2_dist = compute_wasserstein_distance(real_data, target_data)
        logger.info(f"✓ W2 distance: {w2_dist.item():.4f}")

        return True
    except Exception as e:
        logger.error(f"❌ Basic functionality test failed: {e}")
        return False

def test_basic_perturbation():
    """Test basic perturbation without advanced features."""
    logger.info("\n🔄 Testing Basic Perturbation...")
    if not BASIC_IMPORT_SUCCESS:
        logger.error("❌ Cannot test basic perturbation - imports failed")
        return False

    try:
        device = compute_device()
        set_seed(42)

        # Create and pretrain models
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        critic = Critic(data_dim=2, hidden_dim=64).to(device)
        real_sampler = lambda bs: sample_real_data(bs, device=device)
        logger.info(" Pretraining models...")
        pretrained_gen, _ = pretrain_wgan_gp(
            gen, critic, real_sampler,
            epochs=10, batch_size=32, device=device, verbose=False
        )
        logger.info(" ✓ Pretraining completed")

        # Test target-given perturbation
        target_data = sample_target_data(200, shift=[1.5, 1.5], device=device)
        perturber = WeightPerturberTargetGiven(pretrained_gen, target_data)
        logger.info(" Running target-given perturbation...")
        perturbed_gen = perturber.perturb(steps=5, verbose=False)
        logger.info(" ✓ Target-given perturbation completed")

        # Test evidence-based perturbation
        evidence_list, centers = sample_evidence_domains(num_domains=2, samples_per_domain=30, device=device)
        perturber_ev = WeightPerturberTargetNotGiven(pretrained_gen, evidence_list, centers)
        logger.info(" Running evidence-based perturbation...")
        perturbed_gen_ev = perturber_ev.perturb(epochs=5, verbose=False)
        logger.info(" ✓ Evidence-based perturbation completed")

        # Evaluate results
        noise = torch.randn(200, 2, device=device)
        with torch.no_grad():
            orig_samples = pretrained_gen(noise)
            pert_samples = perturbed_gen(noise)
        w2_orig = compute_wasserstein_distance(orig_samples, target_data)
        w2_pert = compute_wasserstein_distance(pert_samples, target_data)
        improvement = (w2_orig.item() - w2_pert.item()) / w2_orig.item() * 100
        logger.info(f" ✓ W2 improvement: {improvement:.2f}%")

        return True
    except Exception as e:
        logger.error(f"❌ Basic perturbation test failed: {e}")
        return False

def test_advanced_features():
    """Test advanced congestion tracking features."""
    logger.info("\n🚀 Testing Advanced Features...")
    if not ADVANCED_IMPORT_SUCCESS:
        logger.warning("⚠️ Advanced features not available - skipping")
        return True
    try:
        theoretical_ok = check_theoretical_support()
        logger.info(f" Theoretical support: {theoretical_ok}")
        device = compute_device()
        set_seed(42)
        tracker = CongestionTracker(lambda_param=0.1)
        logger.info(" ✓ Congestion tracker created")
        samples = torch.randn(100, 2, device=device)
        density_info = compute_spatial_density(samples, bandwidth=0.1)
        logger.info(f" ✓ Spatial density computed: {density_info['density_at_samples'].shape}")
        sobolev_critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=64).to(device)
        test_input = torch.randn(50, 2, device=device)
        output = sobolev_critic(test_input)
        logger.info(f" ✓ Sobolev critic output: {output.shape}")
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        noise = torch.randn(50, 2, device=device)
        sigma = density_info['density_at_samples'][:50]
        flow_info = compute_traffic_flow(sobolev_critic, gen, noise, sigma)
        logger.info(f" ✓ Traffic flow computed: {flow_info['traffic_flow'].shape}")
        real_sampler = lambda bs: sample_real_data(bs, device=device)
        logger.info(" Pretraining for congestion test...")
        pretrained_gen, pretrained_critic = pretrain_wgan_gp(
            gen, sobolev_critic, real_sampler,
            epochs=5, batch_size=32, device=device, verbose=False
        )
        target_data = sample_target_data(200, shift=[1.0, 1.0], device=device)
        ct_perturber = CTWeightPerturberTargetGiven(
            pretrained_gen, target_data,
            critic=pretrained_critic,  # 기존 코드 유지 (이 클래스에서 'critic' 지원 가정)
            enable_congestion_tracking=True
        )
        logger.info(" Running congestion-aware perturbation...")
        ct_perturbed_gen = ct_perturber.perturb(steps=3, verbose=False)
        logger.info(" ✓ Congestion-aware perturbation completed")
        evidence_list, centers = sample_evidence_domains(num_domains=3, samples_per_domain=50, device=device)
        ct_perturber_ev = CTWeightPerturberTargetNotGiven(
            pretrained_gen, evidence_list, centers,
            critics=[pretrained_critic],  # 'critic' 대신 'critics'로 변경 (다중 지원)
            enable_congestion_tracking=True
        )
        logger.info(" Running evidence-based congestion perturbation...")
        ct_perturbed_gen_ev = ct_perturber_ev.perturb(epochs=3, verbose=False)
        logger.info(" ✓ Evidence-based congestion perturbation completed")
        return True
    except Exception as e:
        logger.error(f"❌ Advanced features test failed: {e}")
        return False

def test_visualization():
    """Test visualization capabilities."""
    logger.info("\n📊 Testing Visualization...")
    if not BASIC_IMPORT_SUCCESS:
        logger.error("❌ Cannot test visualization - imports failed")
        return False

    try:
        device = compute_device()
        set_seed(42)

        # Generate test data
        original = torch.randn(100, 2, device=device)
        perturbed = original + 0.5 * torch.randn_like(original)
        target = torch.randn(100, 2, device=device) + torch.tensor([2.0, 2.0], device=device)
        evidence = [torch.randn(50, 2, device=device) + torch.tensor([i*2.0, 0.0], device=device) for i in range(2)]

        # Test plotting
        output_dir = Path("test_results/plots/comprehensive/")
        output_dir.mkdir(parents=True, exist_ok=True)
        plot_distributions(
            original, perturbed, target, evidence,
            title="Test Visualization",
            save_path=str(output_dir / "test_plot.png"),
            show=False
        )
        logger.info(" ✓ Visualization test completed")
        logger.info(f" Plot saved to: {output_dir / 'test_plot.png'}")

        # Test with advanced features if available
        if ADVANCED_IMPORT_SUCCESS:
            logger.info(" Testing advanced visualization integration...")
            # Mock some advanced data
            samples = torch.randn(100, 2, device=device)
            density = compute_spatial_density(samples)
            logger.info(" ✓ Advanced density visualization data prepared")

        return True
    except Exception as e:
        logger.error(f"❌ Visualization test failed: {e}")
        return False

def test_performance():
    """Test performance characteristics."""
    logger.info("\n⚡ Testing Performance...")
    if not BASIC_IMPORT_SUCCESS:
        logger.error("❌ Cannot test performance - imports failed")
        return False

    try:
        device = compute_device()
        set_seed(42)

        # Model creation timing
        start_time = time.time()
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=128).to(device)
        critic = Critic(data_dim=2, hidden_dim=128).to(device)
        model_time = time.time() - start_time
        logger.info(f" Model creation: {model_time:.4f}s")

        # Data sampling timing with larger batch
        start_time = time.time()
        real_data = sample_real_data(10000, device=device)
        target_data = sample_target_data(10000, device=device)
        sampling_time = time.time() - start_time
        logger.info(f" Data sampling (10k samples): {sampling_time:.4f}s")

        # Distance computation timing
        start_time = time.time()
        w2_dist = compute_wasserstein_distance(real_data[:1000], target_data[:1000])
        distance_time = time.time() - start_time
        logger.info(f" W2 distance (1k samples): {distance_time:.4f}s")

        # Memory usage check
        param_count_gen = sum(p.numel() for p in gen.parameters())
        param_count_critic = sum(p.numel() for p in critic.parameters())
        logger.info(f" Generator parameters: {param_count_gen:,}")
        logger.info(f" Critic parameters: {param_count_critic:,}")

        # Quick perturbation timing
        real_sampler = lambda bs: sample_real_data(bs, device=device)
        start_time = time.time()
        pretrained_gen, _ = pretrain_wgan_gp(
            gen, critic, real_sampler,
            epochs=5, batch_size=64, device=device, verbose=False
        )
        pretrain_time = time.time() - start_time
        logger.info(f" Pretraining (5 epochs): {pretrain_time:.4f}s")

        start_time = time.time()
        perturber = WeightPerturberTargetGiven(pretrained_gen, target_data[:500])
        perturbed_gen = perturber.perturb(steps=3, verbose=False)
        perturbation_time = time.time() - start_time
        logger.info(f" Perturbation (3 steps): {perturbation_time:.4f}s")

        # If advanced available, test performance of advanced features
        if ADVANCED_IMPORT_SUCCESS:
            start_time = time.time()
            samples = torch.randn(1000, 2, device=device)
            density = compute_spatial_density(samples)
            advanced_time = time.time() - start_time
            logger.info(f" Advanced density computation (1k samples): {advanced_time:.4f}s")

        logger.info(" ✓ Performance test completed")
        return True
    except Exception as e:
        logger.error(f"❌ Performance test failed: {e}")
        return False

def test_error_handling():
    """Test error handling and edge cases."""
    logger.info("\n🛡️ Testing Error Handling...")
    if not BASIC_IMPORT_SUCCESS:
        logger.error("❌ Cannot test error handling - imports failed")
        return False
    try:
        device = compute_device()
        try:
            Generator(noise_dim=0, data_dim=2, hidden_dim=64)
            logger.error("❌ Should have failed with invalid noise_dim")
            return False
        except ValueError:
            logger.info(" ✓ Correctly caught invalid noise_dim")
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        try:
            empty_targets = torch.empty(0, 2, device=device)
            WeightPerturberTargetGiven(gen, empty_targets)
            logger.error("❌ Should have failed with empty targets")
            return False
        except ValueError:
            logger.info(" ✓ Correctly caught empty target samples")
        try:
            real_data = torch.randn(100, 2)
            target_data = torch.randn(100, 3)
            compute_wasserstein_distance(real_data, target_data)
            logger.error("❌ Should have failed with dimension mismatch")
            return False
        except ValueError:
            logger.info(" ✓ Correctly caught dimension mismatch")
        try:
            WeightPerturberTargetNotGiven(gen, [], [])
            logger.error("❌ Should have failed with empty evidence")
            return False
        except ValueError:
            logger.info(" ✓ Correctly caught empty evidence list")
        try:
            compute_device('invalid')  # TypeError 발생 예상
            logger.error("❌ Should have raised an error for invalid device")
            return False
        except TypeError:  # TypeError를 잡도록 변경 (ValueError 대신)
            logger.info(" ✓ Correctly caught invalid device")
        try:
            pretrain_wgan_gp(gen, Critic(data_dim=2, hidden_dim=64), lambda bs: torch.randn(bs, 2), epochs=0)
            logger.info(" ✓ Handled zero epochs pretraining")
        except Exception as e:
            logger.error(f"❌ Zero epochs pretraining failed: {e}")
            return False
        try:
            sample_real_data(0)
            logger.error("❌ Should have failed with zero batch size")
            return False
        except ValueError:
            logger.info(" ✓ Correctly caught zero batch size")
        logger.info(" ✓ Error handling test completed")
        return True
    except Exception as e:
        logger.error(f"❌ Error handling test failed: {e}")
        return False

class UnitTests(unittest.TestCase):
    """Unit tests for library components."""

    def setUp(self):
        self.device = compute_device()

    def test_model_creation(self):
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
        self.assertIsInstance(gen, nn.Module)
        critic = Critic(data_dim=2, hidden_dim=64)
        self.assertIsInstance(critic, nn.Module)

    def test_data_sampling(self):
        real_data = sample_real_data(100, device=self.device)
        self.assertEqual(real_data.shape, (100, 2))
        target_data = sample_target_data(100, device=self.device)
        self.assertEqual(target_data.shape, (100, 2))
        evidence, centers = sample_evidence_domains(2, 50, device=self.device)
        self.assertEqual(len(evidence), 2)
        self.assertEqual(evidence[0].shape, (50, 2))

    def test_wasserstein_distance(self):
        data1 = torch.randn(50, 2, device=self.device)
        data2 = torch.randn(50, 2, device=self.device)
        dist = compute_wasserstein_distance(data1, data2)
        self.assertIsInstance(dist.item(), float)

    def test_advanced_components(self):
        if not ADVANCED_IMPORT_SUCCESS:
            self.skipTest("Advanced components not available")

        tracker = CongestionTracker()
        self.assertIsInstance(tracker, CongestionTracker)

        samples = torch.randn(50, 2, device=self.device)
        density = compute_spatial_density(samples)
        self.assertIn('density_at_samples', density)

        sobolev_critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=64)
        self.assertIsInstance(sobolev_critic, nn.Module)

def test_unit_tests():
    """Run unit tests."""
    logger.info("\n🧪 Running Unit Tests...")
    suite = unittest.TestLoader().loadTestsFromTestCase(UnitTests)
    output = io.StringIO()
    with redirect_stdout(output):
        result = unittest.TextTestRunner(stream=output, verbosity=2).run(suite)
    logger.info(output.getvalue())
    return result.wasSuccessful()

def create_comprehensive_report(test_results: Dict[str, bool]) -> None:
    """Create a comprehensive test report."""
    output_dir = Path("test_results/logs")
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "comprehensive_test_report.md"

    with open(report_path, 'w') as f:
        f.write("# Weight Perturbation Library - Comprehensive Test Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Python Version:** {sys.version}\n")
        f.write(f"**Platform:** {platform.platform()}\n")
        f.write(f"**Torch Version:** {torch.__version__}\n\n")

        # Test results summary
        f.write("## Test Results Summary\n\n")
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        f.write(f"- **Total Tests:** {total_tests}\n")
        f.write(f"- **Passed:** {passed_tests}\n")
        f.write(f"- **Failed:** {total_tests - passed_tests}\n")
        f.write(f"- **Success Rate:** {passed_tests/total_tests*100:.1f}%\n\n")

        # Individual test results
        f.write("## Individual Test Results\n\n")
        for test_name, result in test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            f.write(f"- **{test_name}:** {status}\n")

        # Environment Information
        f.write("\n## Environment Information\n\n")
        f.write(f"- **PyTorch Version:** {torch.__version__}\n")
        f.write(f"- **Device:** {compute_device()}\n")
        f.write(f"- **CUDA Available:** {torch.cuda.is_available()}\n")
        if BASIC_IMPORT_SUCCESS:
            try:
                version_info = get_version_info()
                f.write(f"- **Library Version:** {version_info.get('version', 'Unknown')}\n")
                f.write(f"- **Basic Functionality:** {version_info.get('basic_functionality', False)}\n")
                f.write(f"- **Theoretical Components:** {version_info.get('theoretical_components', False)}\n")
            except:
                f.write("- **Library Version:** Could not determine\n")

        # Notes and Recommendations
        f.write("\n## Notes and Recommendations\n\n")
        if not BASIC_IMPORT_SUCCESS:
            f.write("- ⚠️ Basic imports failed - library may not be properly installed\n")
            f.write("  - Recommendation: Run `pip install torch geomloss numpy matplotlib seaborn scipy pyyaml`\n")
            f.write("  - Ensure the `src` directory is in your Python path\n")
        if not ADVANCED_IMPORT_SUCCESS:
            f.write("- ⚠️ Advanced features not available - this is normal for basic installations\n")
            f.write("  - For full functionality, ensure all advanced modules are present\n")
        if all(test_results.values()):
            f.write("🎉 **All tests passed!** The library is working correctly.\n")
        else:
            f.write("❗ Some tests failed. Please check the individual results and recommendations above.\n")

    logger.info(f"\n📄 Comprehensive report saved to: {report_path}")

def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Comprehensive Weight Perturbation Library Test")
    parser.add_argument("--skip-slow", action="store_true", help="Skip slow tests")
    parser.add_argument("--basic-only", action="store_true", help="Run only basic tests")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    print("🧪 Weight Perturbation Library - Comprehensive Test")
    print("=" * 60)
    
    # Store test results
    test_results = {}
    
    # Always run basic functionality test
    test_results["Basic Functionality"] = test_basic_functionality()
    
    if not args.basic_only:
        test_results["Basic Perturbation"] = test_basic_perturbation()
        test_results["Advanced Features"] = test_advanced_features()
        test_results["Visualization"] = test_visualization()
        test_results["Error Handling"] = test_error_handling()
        
        if not args.skip_slow:
            test_results["Performance"] = test_performance()

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("📋 TEST SUMMARY")
    logger.info("=" * 60)
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name:<40} {status}")
    logger.info("-" * 60)
    logger.info(f"Total: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")

    # Create comprehensive report
    create_comprehensive_report(test_results)

    # Final status
    if all(test_results.values()):
        logger.info("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        logger.warning(f"\n❌ {total_tests - passed_tests} test(s) failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())
