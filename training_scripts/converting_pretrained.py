import torch
from hybrid_ddpg import DDPG_Gaussian

# 1. SETUP
# Points to your best training checkpoint (The Dict)
run = "run8"
CHECKPOINT_PATH = f"/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/exp_local/booster/checkpoints/{run}/baseline_ddpg_final.pt"
# The name of the file we will generate (The Brain)
OUTPUT_PATH = f"/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/exp_local/booster/checkpoints/{run}/submission_model.pt"

def convert():
    print(f"Loading checkpoint: {CHECKPOINT_PATH}")
    
    # 2. Initialize the architecture EXACTLY as you trained it
    # input_dim=89 is critical because of your training fix
    model = DDPG_Gaussian(input_dim=89, action_dim=12, actor_hidden=[256, 256, 256])
    
    # 3. Load the Weights
    checkpoint = torch.load(CHECKPOINT_PATH, map_location='cpu')
    
    # Handle the dictionary structure
    if 'actor' in checkpoint:
        model.actor.load_state_dict(checkpoint['actor'])
    else:
        model.actor.load_state_dict(checkpoint)
        
    print("Weights loaded successfully.")
    
    # 4. Compile to TorchScript (The "Baking" process)
    model.actor.eval()
    print("Compiling model to TorchScript...")
    
    # We create a dummy input to trace the graph
    dummy_input = torch.rand(1, 89, dtype=torch.float32)
    traced_actor = torch.jit.trace(model.actor, dummy_input)
    
    # 5. Save
    traced_actor.save(OUTPUT_PATH)
    print(f"SUCCESS! Saved submission file to: {OUTPUT_PATH}")
    print("Use this file for the Web UI or submit_sai.py")

if __name__ == "__main__":
    convert()