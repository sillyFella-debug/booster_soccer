import numpy as np
import torch
from sai_rl import SAIClient

# 1. CONFIG
COMP_ID = "lower-t1-kick-to-target" 
MODEL_PATH = "/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/exp_local/booster/checkpoints/run8/submission_model.pt" # The file from Part 1

# 2. PREPROCESSOR (Paste the Universal one from Part 2 Here)
class Preprocessor:
    # ... PASTE THE CODE FROM PART 2 ABOVE HERE ...
    # (Including get_task_onehot, quat_rotate_inverse, modify_state)
    pass 

# 3. MODEL WRAPPER (With Built-in Scaling)
class BoosterModel:
    def __init__(self, model_path):
        # Load the TorchScript model
        self.model = torch.jit.load(model_path)
        
        # Define Action Bounds (Hardcoded for this robot)
        self.low = np.array([-45., -45., -30., -65., -24., -15., -45., -45., -30., -65., -24., -15.], dtype=np.float32)
        self.high = np.array([45., 45., 30., 65., 24., 15., 45., 45., 30., 65., 24., 15.], dtype=np.float32)

    def __call__(self, obs):
        # 1. Run Neural Network
        with torch.no_grad():
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
            # Get output (-1 to 1)
            raw_action = self.model(tensor_obs).numpy()
            
        # 2. Scale Action (Here is the fix for action_fn!)
        # Map [-1, 1] -> [low, high]
        action_percent = (raw_action + 1.0) / 2.0
        action_percent = np.clip(action_percent, 0, 1)
        scaled_action = self.low + (self.high - self.low) * action_percent
        
        return scaled_action

if __name__ == "__main__":
    print(f"Submitting to {COMP_ID}...")
    sai = SAIClient(comp_id=COMP_ID)
    
    # Load Model
    model = BoosterModel(model_path=MODEL_PATH)
    
    # Benchmark Locally First (Optional)
    print("Benchmarking locally...")
    # sai.benchmark(model, preprocessor_class=Preprocessor)
    sai.evaluate(model, preprocessor_class=Preprocessor)
    
    # Submit (Notice we REMOVED action_fn=...)
    print("Uploading...")
    sai.submit(
        name="RL_Agent_gen1",
        model=model,
        preprocessor_class=Preprocessor
    )
    print("Done!")