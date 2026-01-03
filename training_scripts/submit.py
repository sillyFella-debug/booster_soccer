import os
import numpy as np
import torch
from sai_rl import SAIClient

# =========================================================
# 1. CONFIGURATION
# =========================================================
# Choose the competition you are submitting to:
COMP_ID = "lower-t1-kick-to-target" 
# Path to the CONVERTED model you just made:
MODEL_PATH = "/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/exp_local/booster/checkpoints/run8/submission_model.pt" 

# =========================================================
# 2. YOUR CUSTOM PREPROCESSOR (The 89-Dim Fix)
# =========================================================
class Preprocessor():
    def get_task_onehot(self, info):
        return info.get('task_index', np.array([]))

    def quat_rotate_inverse(self, q, v):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        # 1. Standard Stack (Matches your training code)
        if len(obs.shape) == 1: obs = np.expand_dims(obs, axis=0)
        
        # Expand info dimensions if needed
        for key in ["robot_quat", "robot_gyro"]:
            if key in info and len(info[key].shape) == 1:
                info[key] = np.expand_dims(info[key], axis=0)

        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        # Stack basic physics (This usually results in 45 or 87 dims)
        obs = np.concatenate([obs, project_gravity, base_ang_vel], dtype=np.float32, axis=1)

        # 2. FORCE DIMENSION TO 89 (Your Critical Fix)
        target_dim = 89
        current_dim = obs.shape[-1]
        
        if current_dim < target_dim:
            diff = target_dim - current_dim
            padding = np.zeros((obs.shape[0], diff), dtype=np.float32)
            obs = np.hstack((obs, padding))
        elif current_dim > target_dim:
            obs = obs[:, :target_dim]

        return obs

# =========================================================
# 3. YOUR ACTION FUNCTION (Scaling -1/1 to Motors)
# =========================================================
def action_function(policy, env_low=None, env_high=None):
    # Note: SAI might pass raw policy output. We need to scale it.
    # Hardcoded bounds for this robot (standard across repo)
    if env_low is None: env_low = np.array([-45]*12)
    if env_high is None: env_high = np.array([45]*12)
    
    expected_bounds = [-1, 1]
    action_percent = (policy - expected_bounds[0]) / (expected_bounds[1] - expected_bounds[0])
    bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
    return env_low + (env_high - env_low) * bounded_percent

# Wrapper to make it look like what SAI expects
class BoosterModel:
    def __init__(self, model_path):
        self.model = torch.jit.load(model_path)
    
    def __call__(self, obs):
        with torch.no_grad():
            tensor_obs = torch.as_tensor(obs, dtype=torch.float32)
            return self.model(tensor_obs).numpy()

# =========================================================
# 4. SUBMISSION LOGIC
# =========================================================
if __name__ == "__main__":
    print(f"Submitting to: {COMP_ID}")
    sai = SAIClient(comp_id=COMP_ID)
    
    # Need to instantiate env once to get action bounds for action_function closure
    env = sai.make_env()
    low, high = env.action_space.low, env.action_space.high
    
    # create a partial function for the submission that includes the bounds
    import functools
    submit_action_fn = functools.partial(action_function, env_low=low, env_high=high)

    model = BoosterModel(model_path=MODEL_PATH)

    print("Benchmarking locally first...")
    try:
        sai.benchmark(model, preprocessor_class=Preprocessor, action_fn=submit_action_fn)
    except Exception as e:
        print(f"Local Benchmark Warning: {e}")
        print("Submitting anyway...")

    sai.submit(
        name="RL_Finetuned_Gen1",
        model=model,
        preprocessor_class=Preprocessor,
        action_fn=submit_action_fn
    )