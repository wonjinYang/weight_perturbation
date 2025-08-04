#!/usr/bin/env python3
"""
Debug script to test Generator creation and parameter vector issues.
Run this to identify the exact problem with parameter vector creation.
"""

import torch
import sys
import os

# Add src to path to import weight_perturbation
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

try:
    from weight_perturbation.models import Generator, Critic
    from weight_perturbation.utils import parameters_to_vector, vector_to_parameters
    from weight_perturbation.perturbation import WeightPerturberTargetGiven
    from weight_perturbation.samplers import sample_real_data, sample_target_data
    from weight_perturbation.pretrain import pretrain_wgan_gp
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

def debug_parameters_to_vector():
    """Debug the parameters_to_vector function specifically."""
    print("\n=== Debugging parameters_to_vector ===")
    
    # Create a simple generator
    gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
    
    # Check parameters exist
    param_list = list(gen.parameters())
    print(f"Number of parameters: {len(param_list)}")
    
    for i, param in enumerate(param_list):
        print(f"  Param {i}: shape={param.shape}, numel={param.numel()}, requires_grad={param.requires_grad}")
    
    # Test parameters_to_vector with explicit debugging
    try:
        print("Attempting parameters_to_vector...")
        
        # Method 1: Direct call
        vec = parameters_to_vector(gen.parameters())
        print(f"✓ Success! Vector size: {vec.numel()}")
        return True
        
    except Exception as e:
        print(f"✗ Failed with error: {e}")
        import traceback
        traceback.print_exc()
        
        # Method 2: Manual implementation for debugging
        print("Trying manual implementation...")
        try:
            param_list = list(gen.parameters())
            vec_parts = []
            total_numel = 0
            
            for i, p in enumerate(param_list):
                print(f"  Processing param {i}: shape={p.shape}, numel={p.numel()}")
                if p.numel() > 0:
                    flat = p.contiguous().view(-1)
                    vec_parts.append(flat)
                    total_numel += p.numel()
                    print(f"    Added to vector, cumulative size: {total_numel}")
            
            if vec_parts:
                final_vec = torch.cat(vec_parts)
                print(f"✓ Manual method success! Vector size: {final_vec.numel()}")
                return True
            else:
                print("✗ No valid parameters found")
                return False
                
        except Exception as e2:
            print(f"✗ Manual method also failed: {e2}")
            import traceback
            traceback.print_exc()
            return False

def test_generator_creation():
    """Test basic generator creation and parameter counting."""
    print("\n=== Testing Generator Creation ===")
    
    try:
        gen = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
        param_count = sum(p.numel() for p in gen.parameters())
        print(f"✓ Generator created with {param_count} parameters")
        
        # List all parameters
        print("Generator parameters:")
        for name, param in gen.named_parameters():
            print(f"  {name}: shape={param.shape}, numel={param.numel()}, device={param.device}")
        
        # Test parameter vector creation with the new debug info
        if debug_parameters_to_vector():
            print("✓ Parameter vector creation successful")
        else:
            print("✗ Parameter vector creation failed")
            return None
        
        # Test forward pass
        noise = torch.randn(10, 2)
        output = gen(noise)
        print(f"✓ Forward pass successful: input {noise.shape} -> output {output.shape}")
        
        return gen
        
    except Exception as e:
        print(f"✗ Generator creation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_generator_copy(original_gen):
    """Test generator copying mechanism."""
    print("\n=== Testing Generator Copy ===")
    
    if original_gen is None:
        print("✗ Skipping copy test - no original generator")
        return None
    
    try:
        # Create a mock perturber to test copy mechanism
        target_samples = torch.randn(100, 2)
        perturber = WeightPerturberTargetGiven(original_gen, target_samples)
        
        # Test copy creation
        data_dim = 2
        copy_gen = perturber._create_generator_copy(data_dim)
        
        copy_param_count = sum(p.numel() for p in copy_gen.parameters())
        print(f"✓ Generator copy created with {copy_param_count} parameters")
        
        # Test parameter vector on copy with debug
        print("Testing parameter vector on copy...")
        if debug_parameters_to_vector():
            print("✓ Copy parameter vector creation successful")
        else:
            print("✗ Copy parameter vector creation failed")
            return None
        
        return copy_gen
        
    except Exception as e:
        print(f"✗ Generator copy failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_perturbation_initialization():
    """Test the full perturbation initialization."""
    print("\n=== Testing Perturbation Initialization ===")
    
    try:
        # Create models
        generator = Generator(noise_dim=2, data_dim=2, hidden_dim=64)
        critic = Critic(data_dim=2, hidden_dim=64)
        
        # Simple pretraining (1 epoch)
        real_sampler = lambda bs: sample_real_data(bs, device='cpu')
        pretrained_gen, _ = pretrain_wgan_gp(
            generator, critic, real_sampler,
            epochs=1, batch_size=32, noise_dim=2, device='cpu', verbose=False
        )
        print("✓ Pretraining completed")
        
        # Create target samples
        target_samples = sample_target_data(100, device='cpu')
        print("✓ Target samples created")
        
        # Initialize perturber
        perturber = WeightPerturberTargetGiven(pretrained_gen, target_samples)
        print("✓ Perturber initialized")
        
        # Try one perturbation step
        print("Attempting perturbation step...")
        perturbed_gen = perturber.perturb(steps=1, verbose=True)
        print("✓ Perturbation step completed")
        
        return True
        
    except Exception as e:
        print(f"✗ Perturbation initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all debug tests."""
    print("Starting Weight Perturbation Debug Tests...")
    
    # Test 0: Debug parameters_to_vector function specifically
    success_param_debug = debug_parameters_to_vector()
    
    # Test 1: Basic generator creation
    gen = test_generator_creation()
    
    # Test 2: Generator copying
    copy_gen = test_generator_copy(gen)
    
    # Test 3: Full perturbation initialization
    success = test_perturbation_initialization()
    
    print("\n=== Summary ===")
    if success and success_param_debug:
        print("✓ All tests passed! The library should work correctly.")
    else:
        print("✗ Some tests failed. Check the error messages above.")
        print(f"  Parameter debug: {'✓' if success_param_debug else '✗'}")
        print(f"  Perturbation test: {'✓' if success else '✗'}")

if __name__ == "__main__":
    main()