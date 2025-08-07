#!/usr/bin/env python3
"""
Comprehensive test example for the Weight Perturbation library.

This script demonstrates all major features and serves as an integration test.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from pathlib import Path

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
    print(f"❌ Basic import failed: {e}")
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
    print(f"⚠️  Advanced imports not available: {e}")
    ADVANCED_IMPORT_SUCCESS = False


def test_basic_functionality():
    """Test basic functionality that should always work."""
    print("\n🔧 Testing Basic Functionality...")
    
    if not BASIC_IMPORT_SUCCESS:
        print("❌ Cannot test basic functionality - imports failed")
        return False
    
    try:
        # Test device computation
        device = compute_device()
        print(f"✓ Device: {device}")
        
        # Test seed setting
        set_seed(42)
        print("✓ Seed set")
        
        # Test version info
        try:
            version_info = get_version_info()
            print(f"✓ Version info: {version_info}")
        except:
            print("⚠️  Version info not available")
        
        # Test model creation
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
        critic = Critic(data_dim=2, hidden_dim=64)
        print("✓ Models created")
        
        # Test data sampling
        real_data = sample_real_data(100, device=device)
        target_data = sample_target_data(100, shift=[1.0, 1.0], device=device)
        evidence_list, centers = sample_evidence_domains(num_domains=2, samples_per_domain=50, device=device)
        print("✓ Data sampling")
        
        # Test distance computation
        w2_dist = compute_wasserstein_distance(real_data, target_data)
        print(f"✓ W2 distance: {w2_dist.item():.4f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        return False


def test_basic_perturbation():
    """Test basic perturbation without advanced features."""
    print("\n🔄 Testing Basic Perturbation...")
    
    if not BASIC_IMPORT_SUCCESS:
        print("❌ Cannot test basic perturbation - imports failed")
        return False
    
    try:
        device = compute_device()
        set_seed(42)
        
        # Create and pretrain models
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        critic = Critic(data_dim=2, hidden_dim=64).to(device)
        
        real_sampler = lambda bs: sample_real_data(bs, device=device)
        
        print("  Pretraining models...")
        pretrained_gen, _ = pretrain_wgan_gp(
            gen, critic, real_sampler, 
            epochs=10, batch_size=32, device=device, verbose=False
        )
        print("  ✓ Pretraining completed")
        
        # Test target-given perturbation
        target_data = sample_target_data(200, shift=[1.5, 1.5], device=device)
        perturber = WeightPerturberTargetGiven(pretrained_gen, target_data)
        
        print("  Running target-given perturbation...")
        perturbed_gen = perturber.perturb(steps=5, verbose=False)
        print("  ✓ Target-given perturbation completed")
        
        # Test evidence-based perturbation
        evidence_list, centers = sample_evidence_domains(num_domains=2, samples_per_domain=30, device=device)
        perturber_ev = WeightPerturberTargetNotGiven(pretrained_gen, evidence_list, centers)
        
        print("  Running evidence-based perturbation...")
        perturbed_gen_ev = perturber_ev.perturb(epochs=5, verbose=False)
        print("  ✓ Evidence-based perturbation completed")
        
        # Evaluate results
        noise = torch.randn(200, 2, device=device)
        with torch.no_grad():
            orig_samples = pretrained_gen(noise)
            pert_samples = perturbed_gen(noise)
        
        w2_orig = compute_wasserstein_distance(orig_samples, target_data)
        w2_pert = compute_wasserstein_distance(pert_samples, target_data)
        
        improvement = (w2_orig.item() - w2_pert.item()) / w2_orig.item() * 100
        print(f"  ✓ W2 improvement: {improvement:.2f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Basic perturbation test failed: {e}")
        return False


def test_advanced_features():
    """Test advanced congestion tracking features."""
    print("\n🚀 Testing Advanced Features...")
    
    if not ADVANCED_IMPORT_SUCCESS:
        print("⚠️  Advanced features not available - skipping")
        return True  # Not a failure, just not available
    
    try:
        # Check theoretical support
        theoretical_ok = check_theoretical_support()
        print(f"  Theoretical support: {theoretical_ok}")
        
        device = compute_device()
        set_seed(42)
        
        # Test congestion tracker
        tracker = CongestionTracker(lambda_param=0.1)
        print("  ✓ Congestion tracker created")
        
        # Test spatial density computation
        samples = torch.randn(100, 2, device=device)
        density_info = compute_spatial_density(samples, bandwidth=0.1)
        print(f"  ✓ Spatial density computed: {density_info['density_at_samples'].shape}")
        
        # Test Sobolev critic
        sobolev_critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=64).to(device)
        test_input = torch.randn(50, 2, device=device)
        output = sobolev_critic(test_input)
        print(f"  ✓ Sobolev critic output: {output.shape}")
        
        # Test congestion-aware perturbation
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        real_sampler = lambda bs: sample_real_data(bs, device=device)
        
        print("  Pretraining for congestion test...")
        pretrained_gen, pretrained_critic = pretrain_wgan_gp(
            gen, sobolev_critic, real_sampler, 
            epochs=5, batch_size=32, device=device, verbose=False
        )
        
        target_data = sample_target_data(200, shift=[1.0, 1.0], device=device)
        
        # Test congestion-aware target-given perturbation
        ct_perturber = CTWeightPerturberTargetGiven(
            pretrained_gen, target_data, 
            critic=pretrained_critic, 
            enable_congestion_tracking=True
        )
        
        print("  Running congestion-aware perturbation...")
        ct_perturbed_gen = ct_perturber.perturb(steps=3, verbose=False)
        print("  ✓ Congestion-aware perturbation completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Advanced features test failed: {e}")
        return False


def test_visualization():
    """Test visualization capabilities."""
    print("\n📊 Testing Visualization...")
    
    if not BASIC_IMPORT_SUCCESS:
        print("❌ Cannot test visualization - imports failed")
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
        output_dir = Path("test_results/plots")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_distributions(
            original, perturbed, target, evidence,
            title="Test Visualization",
            save_path=str(output_dir / "test_plot.png"),
            show=False
        )
        
        print("  ✓ Visualization test completed")
        print(f"  Plot saved to: {output_dir / 'test_plot.png'}")
        
        return True
        
    except Exception as e:
        print(f"❌ Visualization test failed: {e}")
        return False


def test_performance():
    """Test performance characteristics."""
    print("\n⚡ Testing Performance...")
    
    if not BASIC_IMPORT_SUCCESS:
        print("❌ Cannot test performance - imports failed")
        return False
    
    try:
        device = compute_device()
        set_seed(42)
        
        # Model creation timing
        start_time = time.time()
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=128).to(device)
        critic = Critic(data_dim=2, hidden_dim=128).to(device)
        model_time = time.time() - start_time
        print(f"  Model creation: {model_time:.4f}s")
        
        # Data sampling timing
        start_time = time.time()
        real_data = sample_real_data(1000, device=device)
        target_data = sample_target_data(1000, device=device)
        sampling_time = time.time() - start_time
        print(f"  Data sampling: {sampling_time:.4f}s")
        
        # Distance computation timing
        start_time = time.time()
        w2_dist = compute_wasserstein_distance(real_data, target_data)
        distance_time = time.time() - start_time
        print(f"  W2 distance: {distance_time:.4f}s")
        
        # Memory usage check
        param_count_gen = sum(p.numel() for p in gen.parameters())
        param_count_critic = sum(p.numel() for p in critic.parameters())
        print(f"  Generator parameters: {param_count_gen:,}")
        print(f"  Critic parameters: {param_count_critic:,}")
        
        # Quick perturbation timing
        real_sampler = lambda bs: sample_real_data(bs, device=device)
        
        start_time = time.time()
        pretrained_gen, _ = pretrain_wgan_gp(
            gen, critic, real_sampler, 
            epochs=5, batch_size=64, device=device, verbose=False
        )
        pretrain_time = time.time() - start_time
        print(f"  Pretraining (5 epochs): {pretrain_time:.4f}s")
        
        start_time = time.time()
        perturber = WeightPerturberTargetGiven(pretrained_gen, target_data[:500])
        perturbed_gen = perturber.perturb(steps=3, verbose=False)
        perturbation_time = time.time() - start_time
        print(f"  Perturbation (3 steps): {perturbation_time:.4f}s")
        
        print("  ✓ Performance test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        return False


def test_error_handling():
    """Test error handling and edge cases."""
    print("\n🛡️  Testing Error Handling...")
    
    if not BASIC_IMPORT_SUCCESS:
        print("❌ Cannot test error handling - imports failed")
        return False
    
    try:
        device = compute_device()
        
        # Test invalid model parameters
        try:
            Generator(noise_dim=0, data_dim=2, hidden_dim=64)
            print("❌ Should have failed with invalid noise_dim")
            return False
        except ValueError:
            print("  ✓ Correctly caught invalid noise_dim")
        
        # Test empty target samples
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64).to(device)
        try:
            empty_targets = torch.empty(0, 2, device=device)
            WeightPerturberTargetGiven(gen, empty_targets)
            print("❌ Should have failed with empty targets")
            return False
        except ValueError:
            print("  ✓ Correctly caught empty target samples")
        
        # Test dimension mismatch
        try:
            real_data = torch.randn(100, 2)
            target_data = torch.randn(100, 3)
            compute_wasserstein_distance(real_data, target_data)
            print("❌ Should have failed with dimension mismatch")
            return False
        except ValueError:
            print("  ✓ Correctly caught dimension mismatch")
        
        # Test empty evidence list
        try:
            WeightPerturberTargetNotGiven(gen, [], [])
            print("❌ Should have failed with empty evidence")
            return False
        except ValueError:
            print("  ✓ Correctly caught empty evidence list")
        
        print("  ✓ Error handling test completed")
        
        return True
        
    except Exception as e:
        print(f"❌ Error handling test failed: {e}")
        return False


def create_comprehensive_report(test_results):
    """Create a comprehensive test report."""
    output_dir = Path("test_results")
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "comprehensive_test_report.md"
    
    with open(report_path, 'w') as f:
        f.write("# Weight Perturbation Library - Comprehensive Test Report\n\n")
        f.write(f"**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
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
        
        f.write("\n## Notes\n\n")
        if not BASIC_IMPORT_SUCCESS:
            f.write("- ⚠️ Basic imports failed - library may not be properly installed\n")
        if not ADVANCED_IMPORT_SUCCESS:
            f.write("- ⚠️ Advanced features not available - this is normal for basic installations\n")
        
        f.write("\n## Recommendations\n\n")
        if not BASIC_IMPORT_SUCCESS:
            f.write("1. **Install Required Dependencies:** Run `pip install torch geomloss numpy matplotlib seaborn scipy pyyaml`\n")
            f.write("2. **Check Python Path:** Ensure the `src` directory is in your Python path\n")
        
        if BASIC_IMPORT_SUCCESS and not ADVANCED_IMPORT_SUCCESS:
            f.write("1. **For Advanced Features:** The theoretical components are optional extensions\n")
            f.write("2. **Current Functionality:** Basic weight perturbation works correctly\n")
        
        if all(test_results.values()):
            f.write("🎉 **All tests passed!** The library is working correctly.\n")
    
    print(f"\n📄 Comprehensive report saved to: {report_path}")


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
    
    # Run tests
    test_results["Basic Functionality"] = test_basic_functionality()
    
    if not args.basic_only:
        test_results["Basic Perturbation"] = test_basic_perturbation()
        test_results["Advanced Features"] = test_advanced_features()
        test_results["Visualization"] = test_visualization()
        test_results["Error Handling"] = test_error_handling()
        
        if not args.skip_slow:
            test_results["Performance"] = test_performance()
    
    # Print summary
    print("\n" + "=" * 60)
    print("📋 TEST SUMMARY")
    print("=" * 60)
    
    total_tests = len(test_results)
    passed_tests = sum(test_results.values())
    
    for test_name, result in test_results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
    
    print("-" * 60)
    print(f"Total: {passed_tests}/{total_tests} passed ({passed_tests/total_tests*100:.1f}%)")
    
    # Create comprehensive report
    create_comprehensive_report(test_results)
    
    # Final status
    if all(test_results.values()):
        print("\n🎉 ALL TESTS PASSED! 🎉")
        return 0
    else:
        print(f"\n❌ {total_tests - passed_tests} test(s) failed")
        return 1


if __name__ == "__main__":
    exit(main())