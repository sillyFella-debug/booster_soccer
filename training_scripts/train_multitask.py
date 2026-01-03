import gymnasium as gym
import torch
import numpy as np
import random
import os
from hybrid_ddpg import DDPG_Gaussian
from sai_rl import SAIClient

# --- CONFIG ---
CHECKPOINT_PATH = "./exp_local/booster/checkpoints/run8/checkpoint_60000.pt" # Your last best model
SAVE_DIR = "./exp_local/booster/multitask_checkpoints"
os.makedirs(SAVE_DIR, exist_ok=True)

# Correct IDs (Verify the obstacle one with the snippet above if it fails)
TASKS = [
    "lower-t1-kick-to-target"
    # "lower-t1-penalty-kick-goalie"
    # "lower-t1-penalty-with-obstacles" 
]

# --- 1. UNIVERSAL PREPROCESSOR (The one you used for submission) ---
class UniversalPreprocessor:
    def get_task_onehot(self, info):
        return info.get('task_index', np.array([]))

    def quat_rotate_inverse(self, q, v):
        q_w = q[:, [-1]]
        q_vec = q[:, :3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1, 1) * 2.0)
        return a - b + c

    def modify_state(self, obs, info):
        # 1. Expand
        if len(obs.shape) == 1: obs = np.expand_dims(obs, axis=0)
        obs = obs.astype(np.float32)
        batch_size = obs.shape[0]

        # 2. Physics
        quat = info.get("robot_quat", np.array([[0,0,0,1]]*batch_size))
        gravity = self.quat_rotate_inverse(quat, np.array([0.,0.,-1.], dtype=np.float32))
        gyro = info.get("robot_gyro", np.zeros((batch_size, 3)))

        # 3. Stack (Basic + Physics)
        # We stick to the basics that are PRESENT in all tasks
        # (qpos, qvel, gravity, gyro) usually sum to ~45 dims
        # The raw obs is usually qpos+qvel
        obs = np.concatenate([obs, gravity, gyro], axis=1).astype(np.float32)

        # 4. Force 89 Dims
        target_dim = 89
        if obs.shape[-1] < target_dim:
            padding = np.zeros((batch_size, target_dim - obs.shape[-1]), dtype=np.float32)
            obs = np.hstack((obs, padding))
        elif obs.shape[-1] > target_dim:
            obs = obs[:, :target_dim]
        return obs

# --- 2. MULTI-TASK REWARD WRAPPER ---
class MultiTaskWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_dist = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dist = self._get_dist(info)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # A. ALIVE BONUS (Keep standing)
        shaped_reward = 0.05
        
        # B. DISTANCE BONUS (Go to ball)
        curr_dist = self._get_dist(info)
        if self.last_dist is not None and curr_dist is not None:
            # Reward moving closer
            shaped_reward += (self.last_dist - curr_dist) * 50.0
        
        self.last_dist = curr_dist
        
        # C. FALL PENALTY
        if done and info.get("robot_fallen", False):
            shaped_reward -= 5.0
            
        return obs, shaped_reward, done, truncated, info

    def _get_dist(self, info):
        if 'ball_xpos_rel_robot' in info:
            return np.linalg.norm(info['ball_xpos_rel_robot'])
        return None

# --- 3. TRAINING LOOP ---
def train():
    # Load Agent
    agent = DDPG_Gaussian(input_dim=89, action_dim=12, actor_hidden=[256, 256, 256])
    
    # Load previous weights
    print(f"Loading weights from {CHECKPOINT_PATH}")
    ckpt = torch.load(CHECKPOINT_PATH, map_location='cpu')
    if 'actor' in ckpt:
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic']) # Load Critic to keep learning!
    else:
        agent.actor.load_state_dict(ckpt)

    preprocessor = UniversalPreprocessor()
    
    # Setup Envs
    envs = {}
    print("Initializing Environments...")
    for task_id in TASKS:
        try:
            # We use gym.make direct if possible, or SAI client
            # Assuming SAI client handles the string ID mapping:
            c = SAIClient(comp_id=task_id)
            env = c.make_env()
            env = MultiTaskWrapper(env)
            envs[task_id] = env
            print(f"Loaded {task_id}")
        except Exception as e:
            print(f"Skipping {task_id}: {e}")

    # Loop
    total_steps = 0
    buffer = [] # Simple buffer
    BATCH_SIZE = 128
    
    for episode in range(2000):
        # 1. Pick Random Task
        task_id = random.choice(list(envs.keys()))
        env = envs[task_id]
        
        state, info = env.reset()
        state = preprocessor.modify_state(state, info).squeeze()
        
        ep_reward = 0
        done = False
        
        while not done:
            # Action with Noise
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            with torch.no_grad():
                action, _ = agent.actor.get_mean_and_std(state_tensor)
                action = action.numpy().squeeze()
            
            # Add exploration noise
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
            # Step
            next_state_raw, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            next_state = preprocessor.modify_state(next_state_raw, info).squeeze()
            
            # Store
            buffer.append((state, action, reward, next_state, done))
            if len(buffer) > 20000: buffer.pop(0)
            
            # Update
            if len(buffer) > BATCH_SIZE:
                # (Insert your DDPG update logic here or call agent.train(...))
                # For brevity, assuming agent.train handles batch extraction
                # You can copy the batch sampling logic from previous scripts
                batch = random.sample(buffer, BATCH_SIZE)
                s, a, r, ns, d = zip(*batch)
                agent.train(s, a, np.array(r).reshape(-1,1), ns, np.array(d).reshape(-1,1))

            state = next_state
            ep_reward += reward
            total_steps += 1
        
        print(f"Ep {episode} [{task_id}] Reward: {ep_reward:.2f}")
        
        if episode % 50 == 0:
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, f"{SAVE_DIR}/multitask_model_{episode}.pt")

if __name__ == "__main__":
    train()