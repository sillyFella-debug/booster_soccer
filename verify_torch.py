import torch
import numpy as np

def verify():
    model_path = "./converted_model.pt"
    print(f"Loading model from {model_path}...")
    model = torch.jit.load(model_path)
    model.eval()

    input_dim = 89
    dummy_input = torch.randn(1, input_dim)
    
    with torch.no_grad():
        mean, std = model(dummy_input)
    
    print("Inference successful!")
    print(f"Mean shape: {mean.shape}")
    print(f"Std shape: {std.shape}")
    print(f"Mean (first 5 values): {mean[0, :5]}")

if __name__ == "__main__":
    verify()
