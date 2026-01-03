import gymnasium as gym
import numpy as np
import os
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from sai_rl import SAIClient
from sac_wrapper import StandingWrapper

# --- CONFIG ---
TASK_ID = "lower-t1-kick-to-target" # Train on Target first to learn walking
SAVE_DIR = "./sac_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# --- UNIVERSAL PREPROCESSOR (Keep this to prevent crashes) ---
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
        
        # 3. Dot product term (THE FIX)
        # We calculate the dot product for each row individually
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
        # Default quaternion (0,0,0,1)
        default_quat = np.zeros((batch_size, 4), dtype=np.float32)
        default_quat[:, -1] = 1.0 # Set w=1
        
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
    
class PreprocessEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prep = UniversalPreprocessor()
        # Update observation space to match 89 dims
        low = np.full((89,), -np.inf, dtype=np.float32)
        high = np.full((89,), np.inf, dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
    
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs = self.prep.modify_state(obs, info).squeeze()
        return obs, info
    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs = self.prep.modify_state(obs, info).squeeze()
        return obs, reward, terminated, truncated, info

def train():
    print(f"Setting up SAC for {TASK_ID}...")
    
    # 1. Setup Env
    sai = SAIClient(comp_id=TASK_ID)
    env = sai.make_env()
    env = StandingWrapper(env)   # Add Rewards
    env = PreprocessEnv(env)     # Add Preprocessor
    
    # 2. Setup SAC
    # ent_coef="auto" is the MAGIC that prevents curling. 
    # It automatically tunes how much randomness is needed.
    model = SAC(
        "MlpPolicy", 
        env, 
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        ent_coef="auto",  # <--- CRITICAL FOR FIXING CURLING
        train_freq=1,
        gradient_steps=1,
        learning_starts=1000
    )
    
    # 3. Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=SAVE_DIR,
        name_prefix="sac_model"
    )
    
    # 4. Train
    print("Starting Training (From Scratch)...")
    model.learn(total_timesteps=1000000, callback=checkpoint_callback)
    model.save("sac_final_model")

if __name__ == "__main__":
    train()