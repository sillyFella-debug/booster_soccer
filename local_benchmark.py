import gymnasium as gym
import torch
import numpy as np
from hybrid_ddpg import DDPG_Gaussian
# from baseline_training_pretrained import Preprocessor
from baseline_penalty import Preprocessor
from sai_rl import SAIClient

# --- 1. THE OFFICIAL SCORING FUNCTION (From Website) ---
reward_config = {
    "robot_distance_ball": 0.25, "ball_vel_twd_goal": 1.5, "goal_scored": 2.50,
    "offside": -3.0, "ball_hits": -0.2, "robot_fallen": -1.5, 
    "ball_blocked": -0.5, "steps": -1.0,
}

def official_eval_fn(env, raw_reward, done, info):
    # Simplified version of their logic for local testing
    score = 0.0
    # Add step penalty
    score += -1.0 
    
    # Add components if they exist in raw_reward dict
    # (Note: local env might return reward as float, not dict. 
    #  In that case, we just return the raw reward as it usually matches config)
    if isinstance(raw_reward, float):
        return raw_reward
        
    return score

# --- 2. BENCHMARK SETUP ---
def run_benchmark():
    # Setup
    # env_name = "lower-t1-kick-to-target"
    # env_name = "lower-t1-penalty-obstacle"
    # env_name = "lower-t1-penalty-kick-goalie"
    print(f"Benchmarking on {env_name}...")
    sai = SAIClient(comp_id=env_name)
    # sai = SAIClient()
    env = sai.make_env()
    
    # Initialize Model (MATCH YOUR TRAINING DIMS!)
    model = DDPG_Gaussian(input_dim=89, action_dim=12, actor_hidden=[256, 256, 256])
    
    # Load YOUR Checkpoint
    # checkpoint_path = "media/deter/New Volume/Neamur/codes/booster_soccer_showdown/exp_local/booster/checkpoints/run8/baseline_ddpg_final.pt" # <--- UPDATE THIS
    checkpoint_path = "/media/deter/New Volume/Neamur/codes/booster_soccer_showdown/exp_local/booster/checkpoints/run8/baseline_ddpg_final.pt" # <--- UPDATE THIS

    print(f"Loading: {checkpoint_path}")
    
    # Use the smart loader logic we added to ddpg_gaussian.py
    # If you didn't add it, manually load the dict:
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    if 'actor' in ckpt:
        model.actor.load_state_dict(ckpt['actor'])
    else:
        model.actor.load_state_dict(ckpt)
    
    preprocessor = Preprocessor()
    
    # --- 3. RUN EPISODES ---
    num_episodes = 5
    total_score = 0
    
    for ep in range(num_episodes):
        obs, info = env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        done = False
        ep_score = 0
        
        while not done:
            # Get Action
            state_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
            with torch.no_grad():
                # Use pure mean (no noise) for evaluation!
                action, _ = model.actor.get_mean_and_std(state_tensor)
                action = action.numpy().squeeze()
            
            # Step
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            obs = preprocessor.modify_state(next_obs, info).squeeze()
            ep_score += reward # This is the official environment reward
            
        print(f"Episode {ep+1}: Score = {ep_score:.2f}")
        total_score += ep_score

    print(f"AVERAGE SCORE: {total_score / num_episodes:.2f}")

if __name__ == "__main__":
    run_benchmark()