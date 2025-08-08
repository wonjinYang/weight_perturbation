import torch
import time
from weight_perturbation import (
    Generator,
    Critic,
    sample_target_data,
    sample_real_data,
    compute_device,
    compute_wasserstein_distance,
    global_w2_loss_and_grad
)

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
