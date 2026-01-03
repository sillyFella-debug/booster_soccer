import torch
import torch.nn as nn
from stable_baselines3 import SAC

MODEL_PATH = "sac_models/sac_model_100000_steps.zip" # UPDATE THIS
OUTPUT_PATH = "sac_submission.pt"

class BakedSAC(nn.Module):
    def __init__(self, original_actor):
        super().__init__()
        self.actor = original_actor
        # Hardcoded Action Bounds
        self.register_buffer('low', torch.tensor([-45.]*12, dtype=torch.float32))
        self.register_buffer('high', torch.tensor([45.]*12, dtype=torch.float32))

    def forward(self, x):
        # SB3 actor outputs mean in [-1, 1] directly via .forward() logic usually
        # But specifically, we access the latent deterministic action
        raw_action = self.actor(x)
        
        # Scale
        action_percent = (raw_action + 1.0) / 2.0
        action_percent = torch.clamp(action_percent, 0.0, 1.0)
        scaled_action = self.low + (self.high - self.low) * action_percent
        return scaled_action

def convert():
    print("Loading SB3 Model...")
    model = SAC.load(MODEL_PATH)
    
    # Extract the Actor Network
    # SB3 Actor is complex, we need the mu (mean) network basically
    # Actually, converting the whole .predict logic is safer
    
    print("Compiling...")
    wrapped_model = BakedSAC(model.policy)
    wrapped_model.eval()
    
    dummy_input = torch.rand(1, 89, dtype=torch.float32)
    
    # Trace it
    traced = torch.jit.trace(wrapped_model, dummy_input)
    traced.save(OUTPUT_PATH)
    print(f"Saved {OUTPUT_PATH}")

if __name__ == "__main__":
    convert()