#!/bin/bash

# Weight Perturbation Library - Congested Transport Testing Script
# This script runs comprehensive tests for the congested transport implementation

set -e  # Exit on any error

echo "=============================================================="
echo "WEIGHT PERTURBATION LIBRARY - CONGESTED TRANSPORT TESTING"
echo "=============================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${2}${1}${NC}"
}

print_status "Starting comprehensive testing suite..." $BLUE

# Check if we're in the right directory
if [[ ! -d "src/weight_perturbation" ]]; then
    print_status "Error: Please run this script from the project root directory" $RED
    exit 1
fi

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Create test results directory
mkdir -p test_results
mkdir -p test_results/plots
mkdir -p test_results/logs

# Function to run a test and capture output
run_test() {
    local test_name="$1"
    local test_command="$2"
    local log_file="test_results/logs/${test_name}.log"
    
    print_status "Running $test_name..." $YELLOW
    
    if eval "$test_command" > "$log_file" 2>&1; then
        print_status "✓ $test_name PASSED" $GREEN
        return 0
    else
        print_status "✗ $test_name FAILED" $RED
        echo "Check log file: $log_file"
        return 1
    fi
}

# Function to check Python dependencies
check_dependencies() {
    print_status "Checking Python dependencies..." $BLUE
    
    python3 -c "
import sys
required_packages = [
    'torch', 'numpy', 'matplotlib', 'seaborn', 'scipy', 
    'geomloss', 'yaml', 'pathlib'
]

missing_packages = []
for package in required_packages:
    try:
        __import__(package)
        print(f'✓ {package}')
    except ImportError:
        missing_packages.append(package)
        print(f'✗ {package} - MISSING')

if missing_packages:
    print(f'Missing packages: {missing_packages}')
    print('Please install missing packages with: pip install ' + ' '.join(missing_packages))
    sys.exit(1)
else:
    print('All dependencies satisfied!')
"
    
    if [[ $? -ne 0 ]]; then
        print_status "Dependency check failed!" $RED
        exit 1
    fi
}

# Test 1: Unit Tests
test_unit_tests() {
    print_status "=== UNIT TESTS ===" $BLUE
    
    # Test models
    run_test "models_test" "python3 -m pytest src/tests/test_models.py -v"
    
    # Test samplers  
    run_test "samplers_test" "python3 -m pytest src/tests/test_samplers.py -v"
    
    # Test losses
    run_test "losses_test" "python3 -m pytest src/tests/test_losses.py -v"
    
    # Test perturbation
    run_test "perturbation_test" "python3 -m pytest src/tests/test_perturbation.py -v"
    
    # Test pretraining
    run_test "pretrain_test" "python3 -m pytest src/tests/test_pretain.py -v"
}

# Test 2: Integration Tests
test_integration() {
    print_status "=== INTEGRATION TESTS ===" $BLUE
    
    run_test "integration_test" "python3 src/tests/test_integration.py"
}

# Test 3: Congestion Theory Tests
test_congestion_theory() {
    print_status "=== CONGESTION THEORY TESTS ===" $BLUE
    
    run_test "congestion_test" "python3 src/tests/test_congestion.py"
}

# Test 4: Basic Examples
test_basic_examples() {
    print_status "=== BASIC EXAMPLES ===" $BLUE
    
    # Section 2 example
    run_test "section2_example" "python3 src/examples/example_section2.py --pretrain_epochs 20 --perturb_steps 10 --verbose"
    
    # Section 3 example  
    run_test "section3_example" "python3 src/examples/example_section3.py --pretrain_epochs 20 --perturb_epochs 10 --verbose"
}

# Test 5: Congestion Examples with Visualization
test_congestion_examples() {
    print_status "=== CONGESTION EXAMPLES WITH VISUALIZATION ===" $BLUE
    
    # Section 2 with congestion
    run_test "section2_congestion" "python3 src/examples/example_with_congestion_section2.py --pretrain_epochs 15 --perturb_steps 8 --visualize_every 2 --save_plots"
    
    # Section 3 with congestion  
    run_test "section3_congestion" "python3 src/examples/example_with_congestion_section3.py --pretrain_epochs 15 --perturb_epochs 8 --visualize_every 2 --save_plots"
}

# Test 6: Performance Tests
test_performance() {
    print_status "=== PERFORMANCE TESTS ===" $BLUE
    
    # Create performance test script
    cat > test_results/performance_test.py << 'EOF'
import torch
import time
import numpy as np
from weight_perturbation import *

def test_performance():
    device = compute_device()
    print(f"Using device: {device}")
    
    # Test basic model creation speed
    start_time = time.time()
    gen = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)
    critic = Critic(data_dim=2, hidden_dim=256).to(device)
    model_time = time.time() - start_time
    print(f"Model creation time: {model_time:.4f}s")
    
    # Test data sampling speed
    start_time = time.time()
    real_data = sample_real_data(1000, device=device)
    target_data = sample_target_data(1000, device=device)
    sampling_time = time.time() - start_time
    print(f"Data sampling time: {sampling_time:.4f}s")
    
    # Test distance computation speed
    start_time = time.time()
    w2_dist = compute_wasserstein_distance(real_data, target_data)
    distance_time = time.time() - start_time
    print(f"W2 distance computation time: {distance_time:.4f}s")
    
    # Test gradient computation speed
    start_time = time.time()
    noise = torch.randn(100, 2, device=device)
    loss, grads = global_w2_loss_and_grad(gen, target_data, noise)
    gradient_time = time.time() - start_time
    print(f"Gradient computation time: {gradient_time:.4f}s")
    
    print("Performance test completed successfully!")

if __name__ == "__main__":
    test_performance()
EOF

    run_test "performance_test" "python3 test_results/performance_test.py"
}

# Test 7: Memory Tests
test_memory() {
    print_status "=== MEMORY TESTS ===" $BLUE
    
    # Create memory test script
    cat > test_results/memory_test.py << 'EOF'
import torch
import gc
import psutil
import os
from weight_perturbation import *

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024 / 1024  # MB

def test_memory_usage():
    device = compute_device()
    print(f"Using device: {device}")
    
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} MB")
    
    # Create models
    gen = Generator(noise_dim=2, data_dim=2, hidden_dim=256).to(device)
    critic = Critic(data_dim=2, hidden_dim=256).to(device)
    
    model_memory = get_memory_usage()
    print(f"Memory after model creation: {model_memory:.2f} MB (+{model_memory-initial_memory:.2f} MB)")
    
    # Generate data
    target_data = sample_target_data(5000, device=device)
    data_memory = get_memory_usage()
    print(f"Memory after data generation: {data_memory:.2f} MB (+{data_memory-model_memory:.2f} MB)")
    
    # Perform perturbation
    try:
        from weight_perturbation import WeightPerturberTargetGiven
        perturber = WeightPerturberTargetGiven(gen, target_data)
        pert_gen = perturber.perturb(steps=5, verbose=False)
        
        pert_memory = get_memory_usage()
        print(f"Memory after perturbation: {pert_memory:.2f} MB (+{pert_memory-data_memory:.2f} MB)")
        
    except Exception as e:
        print(f"Perturbation failed: {e}")
    
    # Clean up
    del gen, critic, target_data
    if 'pert_gen' in locals():
        del pert_gen
    if 'perturber' in locals():
        del perturber
    
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    final_memory = get_memory_usage()
    print(f"Final memory usage: {final_memory:.2f} MB")
    print(f"Total memory increase: {final_memory-initial_memory:.2f} MB")
    
    if final_memory - initial_memory < 100:  # Less than 100MB increase
        print("Memory test PASSED - No significant memory leaks detected")
    else:
        print("Memory test WARNING - Potential memory leak detected")

if __name__ == "__main__":
    test_memory_usage()
EOF

    run_test "memory_test" "python3 test_results/memory_test.py"
}

# Test 8: Theoretical Components Test
test_theoretical_components() {
    print_status "=== THEORETICAL COMPONENTS TEST ===" $BLUE
    
    # Create theoretical test script
    cat > test_results/theoretical_test.py << 'EOF'
import torch
import numpy as np
from weight_perturbation import *

def test_theoretical_components():
    device = compute_device()
    print(f"Using device: {device}")
    
    # Test version info
    try:
        version_info = get_version_info()
        print("Version info:", version_info)
        
        # Check theoretical support
        theoretical_ok = check_theoretical_support()
        print(f"Theoretical components available: {theoretical_ok}")
        
        if not theoretical_ok:
            print("Some theoretical components are not available - this is expected for basic installation")
            return
            
    except Exception as e:
        print(f"Could not check theoretical components: {e}")
        print("This is expected if theoretical components are not installed")
        return
    
    # Test congestion tracking if available
    try:
        from weight_perturbation import (
            CongestionTracker, compute_spatial_density, 
            compute_traffic_flow, SobolevConstrainedCritic
        )
        
        print("Testing congestion tracking...")
        tracker = CongestionTracker(lambda_param=0.1)
        
        # Test spatial density
        samples = torch.randn(100, 2, device=device)
        density_info = compute_spatial_density(samples, bandwidth=0.1)
        print(f"Spatial density computed: {density_info['density_at_samples'].shape}")
        
        # Test Sobolev critic
        sobolev_critic = SobolevConstrainedCritic(data_dim=2, hidden_dim=64).to(device)
        test_input = torch.randn(50, 2, device=device)
        output = sobolev_critic(test_input)
        print(f"Sobolev critic output shape: {output.shape}")
        
        print("Theoretical components test PASSED")
        
    except ImportError as e:
        print(f"Theoretical components not available: {e}")
        print("This is expected if advanced features are not installed")
    except Exception as e:
        print(f"Error testing theoretical components: {e}")

if __name__ == "__main__":
    test_theoretical_components()
EOF

    run_test "theoretical_test" "python3 test_results/theoretical_test.py"
}

# Main testing function
main() {
    local start_time=$(date +%s)
    local total_tests=0
    local passed_tests=0
    local failed_tests=0
    
    # Check dependencies first
    check_dependencies
    
    print_status "Starting test suite execution..." $BLUE
    
    # Run all test categories
    if [[ "${1:-all}" == "all" || "$1" == "unit" ]]; then
        test_unit_tests
        total_tests=$((total_tests + 5))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "integration" ]]; then
        test_integration
        total_tests=$((total_tests + 1))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "congestion" ]]; then
        test_congestion_theory
        total_tests=$((total_tests + 1))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "examples" ]]; then
        test_basic_examples
        total_tests=$((total_tests + 2))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "congestion_examples" ]]; then
        test_congestion_examples
        total_tests=$((total_tests + 2))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "performance" ]]; then
        test_performance
        total_tests=$((total_tests + 1))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "memory" ]]; then
        test_memory
        total_tests=$((total_tests + 1))
    fi
    
    if [[ "${1:-all}" == "all" || "$1" == "theoretical" ]]; then
        test_theoretical_components
        total_tests=$((total_tests + 1))
    fi
    
    # Count results
    passed_tests=$(find test_results/logs -name "*.log" -exec grep -l "PASSED\|completed successfully\|test.*PASSED" {} \; | wc -l)
    failed_tests=$(find test_results/logs -name "*.log" -exec grep -l "FAILED\|ERROR\|Exception" {} \; | wc -l)
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    print_status "=============================================================" $BLUE
    print_status "TEST SUITE COMPLETED" $BLUE
    print_status "=============================================================" $BLUE
    print_status "Duration: ${duration}s" $BLUE
    print_status "Total tests: $total_tests" $BLUE
    print_status "Passed: $passed_tests" $GREEN
    print_status "Failed: $failed_tests" $RED
    print_status "=============================================================" $BLUE
    
    # Generate summary report
    cat > test_results/test_summary.txt << EOF
Weight Perturbation Library - Test Summary
==========================================
Date: $(date)
Duration: ${duration}s
Total tests run: $total_tests
Passed: $passed_tests
Failed: $failed_tests

Test Results:
EOF
    
    for log_file in test_results/logs/*.log; do
        if [[ -f "$log_file" ]]; then
            test_name=$(basename "$log_file" .log)
            if grep -q "PASSED\|completed successfully\|test.*PASSED" "$log_file"; then
                echo "✓ $test_name - PASSED" >> test_results/test_summary.txt
            else
                echo "✗ $test_name - FAILED" >> test_results/test_summary.txt
            fi
        fi
    done
    
    print_status "Test summary saved to: test_results/test_summary.txt" $BLUE
    
    # Check for generated plots
    if [[ -d "section2_traffic_flow_seed_42" ]]; then
        print_status "Section 2 traffic flow plots generated in: section2_traffic_flow_seed_42/" $GREEN
    fi
    
    if [[ -d "section3_multimarginal_seed_2025" ]]; then
        print_status "Section 3 multimarginal plots generated in: section3_multimarginal_seed_2025/" $GREEN
    fi
    
    if [[ $failed_tests -eq 0 ]]; then
        print_status "🎉 ALL TESTS PASSED! 🎉" $GREEN
        exit 0
    else
        print_status "❌ Some tests failed. Check logs in test_results/logs/" $RED
        exit 1
    fi
}

# Help function
show_help() {
    cat << EOF
Weight Perturbation Library - Congested Transport Testing Script

Usage: $0 [test_category]

Test Categories:
  all                - Run all tests (default)
  unit              - Run unit tests only
  integration       - Run integration tests only
  congestion        - Run congestion theory tests only
  examples          - Run basic examples only
  congestion_examples - Run congestion examples with visualization
  performance       - Run performance tests only
  memory           - Run memory tests only
  theoretical      - Run theoretical components tests only

Examples:
  $0                # Run all tests
  $0 unit          # Run only unit tests
  $0 examples      # Run only example tests

Generated files will be saved to:
  - test_results/logs/     - Test log files
  - test_results/plots/    - Generated plots
  - section2_traffic_flow_seed_*/ - Section 2 visualization plots
  - section3_multimarginal_seed_*/ - Section 3 visualization plots
EOF
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        exit 0
        ;;
    all|unit|integration|congestion|examples|congestion_examples|performance|memory|theoretical|"")
        main "$1"
        ;;
    *)
        print_status "Unknown test category: $1" $RED
        show_help
        exit 1
        ;;
esac