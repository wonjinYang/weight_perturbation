import torch
import gc
import psutil
import os
from weight_perturbation import (
    Generator,
    Critic,
    sample_target_data,
    compute_device
)

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
