"""
Inspect converted_model.pt architecture and extract layer dimensions
"""
import torch
from pathlib import Path

model_path = Path(__file__).parent.parent / "converted_model.pt"

print("="*60)
print("INSPECTING CONVERTED_MODEL.PT ARCHITECTURE")
print("="*60)

scripted = torch.jit.load(str(model_path), map_location='cpu')
state_dict = scripted.state_dict()

print("\nAll keys in model:")
for key in sorted(state_dict.keys()):
    shape = state_dict[key].shape
    print(f"  {key:<50} {shape}")

print("\nArchitecture Analysis:")
print("-"*60)

# Extract layer dimensions
hidden_layer_0 = state_dict['actor_net.hidden_layers.0.weight'].shape
hidden_layer_1 = state_dict['actor_net.hidden_layers.1.weight'].shape
hidden_layer_2 = state_dict['actor_net.hidden_layers.2.weight'].shape
mean_out = state_dict['mean_net.weight'].shape

input_dim = hidden_layer_0[1]
hidden_1_dim = hidden_layer_0[0]
hidden_2_dim = hidden_layer_1[0]
hidden_3_dim = hidden_layer_2[0]
output_dim = mean_out[0]

print(f"Input dimension:     {input_dim}")
print(f"Hidden layer 1:      {hidden_1_dim}")
print(f"Hidden layer 2:      {hidden_2_dim}")
print(f"Hidden layer 3:      {hidden_3_dim}")
print(f"Output dimension:    {output_dim}")
print(f"\nNetwork architecture: [{hidden_1_dim}, {hidden_2_dim}, {hidden_3_dim}]")
print(f"Special features: Layer Normalization, Gaussian output (mean + log_std)")
