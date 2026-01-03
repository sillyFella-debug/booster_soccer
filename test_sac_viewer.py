import gymnasium as gym
import numpy as np
import torch
import time
from stable_baselines3 import SAC
from sai_rl import SAIClient

# =========================================================
# CONFIGURATION
# =========================================================
# Update this to your latest checkpoint
MODEL_PATH = "sac_models/sac_model_240000_steps.zip" # <--- CHECK THIS PATH

# The 3 Tasks to test
TASKS = [
    "lower-t1-kick-to-target",
    "lower-t1-penalty-kick-goalie",
    "lower-t1-penalty-obstacle" # Note: Check if you found the correct ID for this
]

# =========================================================
# UNIVERSAL PREPROCESSOR (The Fixed Version)
# =========================================================
class UniversalPreprocessor:
    def get_task_onehot(self, info):
        val = info.get('task_index', np.array([]))
        if len(val.shape) == 1: val = np.expand_dims(val, axis=0)
        return val

    def quat_rotate_inverse(self, q, v):
        # q is (N, 4), v is (N, 3) or (3,)
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        
        # Ensure v matches batch size
        if len(v.shape) == 1: 
            v = np.tile(v, (q.shape[0], 1))
        
        # 1. Scaling term
        a = v * (2.0 * q_w**2 - 1.0)
        # 2. Cross product term
        b = np.cross(q_vec, v) * (q_w * 2.0)
        # 3. Dot product term (Safe Version)
        dot_prod = np.sum(q_vec * v, axis=1, keepdims=True)
        c = q_vec * (dot_prod * 2.0)
        
        return a - b + c

    def modify_state(self, obs, info):
        # 1. Expand Observation
        if len(obs.shape) == 1: 
            obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)
        batch_size = obs.shape[0]

        # 2. Helper to force Info values to 2D
        def get_2d(key, default):
            val = info.get(key, default)
            if len(val.shape) == 1:
                val = np.expand_dims(val, axis=0)
            return val.astype(np.float32)

        # 3. Physics
        default_quat = np.zeros((batch_size, 4), dtype=np.float32)
        default_quat[:, -1] = 1.0
        
        quat = get_2d("robot_quat", default_quat)
        gravity = self.quat_rotate_inverse(quat, np.array([0., 0., -1.], dtype=np.float32))
        gyro = get_2d("robot_gyro", np.zeros((batch_size, 3)))

        # 4. Stack
        obs = np.concatenate([obs, gravity, gyro], axis=1).astype(np.float32)

        # 5. Force 89 Dims
        target_dim = 89
        if obs.shape[-1] < target_dim:
            padding = np.zeros((batch_size, target_dim - obs.shape[-1]), dtype=np.float32)
            obs = np.hstack((obs, padding))
        elif obs.shape[-1] > target_dim:
            obs = obs[:, :target_dim]
            
        return obs

# =========================================================
# TEST LOOP
# =========================================================
def run_test():
    print(f"Loading Model: {MODEL_PATH}")
    try:
        model = SAC.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"❌ Error: Could not find model at {MODEL_PATH}")
        return

    prep = UniversalPreprocessor()

    for task_name in TASKS:
        print(f"\n" + "="*50)
        print(f"Testing Environment: {task_name}")
        print("="*50)
        
        try:
            # Initialize Client & Env
            sai = SAIClient(comp_id=task_name)
            # render_mode="human" pops up the window
            env = sai.make_env(render_mode="human")
        except Exception as e:
            print(f"⚠️ Could not load {task_name}: {e}")
            continue

        obs, info = env.reset()
        done = False
        step = 0
        total_reward = 0

        print("Running simulation... Press Ctrl+C to skip to next task.")
        
        try:
            while not done:
                # 1. Preprocess State
                # SB3 expects a single observation (not a batch) for predict if not vectorized
                # But our preprocessor creates a batch of 1.
                processed_obs = prep.modify_state(obs, info).squeeze()
                
                # 2. Get Action from SAC
                # deterministic=True removes the random noise (best performance)
                action, _states = model.predict(processed_obs, deterministic=True)
                
                # 3. Step Environment
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                total_reward += reward
                step += 1
                
                # Slow down slightly so you can see it
                time.sleep(0.01)
                
                # Optional: Restart if it ends quickly to watch again
                if done:
                    print(f"Episode finished. Steps: {step}, Total Reward: {total_reward:.2f}")
                    # Uncomment to loop forever on this task:
                    # done = False
                    # obs, info = env.reset()
                    # step = 0
                    # total_reward = 0
                    
        except KeyboardInterrupt:
            print("\nSkipping to next task...")
            env.close()
            continue
        
        env.close()

if __name__ == "__main__":
    run_test()