"""
Test script to evaluate the baseline DDPG model.
Runs the trained model on the environment and reports performance.
"""
import torch
import torch.nn.functional as F
import numpy as np
from pathlib import Path

from sai_rl import SAIClient
from ddpg import DDPG_FF
from baseline_training import Preprocessor


def test_model(checkpoint_path, num_episodes=5):
    """Test trained DDPG model on the environment."""
    
    print("="*60)
    print("BOOSTER SOCCER BASELINE TESTING")
    print("="*60)
    
    # Initialize environment
    print("\n[1/3] Initializing environment...")
    sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")
    env = sai.make_env()
    print(f"✓ Environment initialized")
    
    # Create and load model
    print("\n[2/3] Loading trained DDPG model...")
    model = DDPG_FF(
        n_features=87,
        action_space=env.action_space,
        neurons=[128, 64, 128],
        activation_function=F.relu,
        learning_rate=0.0001,
    )
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.actor.load_state_dict(checkpoint['actor'])
    model.critic.load_state_dict(checkpoint['critic'])
    model.target_actor.load_state_dict(checkpoint['target_actor'])
    model.target_critic.load_state_dict(checkpoint['target_critic'])
    print(f"✓ Model loaded from: {checkpoint_path}")
    
    # Test model on environment
    print(f"\n[3/3] Running {num_episodes} test episodes...")
    print("-"*60)
    
    preprocessor = Preprocessor()
    episode_rewards = []
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        episode_reward = 0.0
        done = False
        steps = 0
        
        while not done:
            # Get action from actor network (deterministic, no noise)
            with torch.no_grad():
                state_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
                policy_output = model.actor(state_tensor).detach().numpy()
            
            # Map to action space
            expected_bounds = [-1, 1]
            action_percent = (policy_output - expected_bounds[0]) / (
                expected_bounds[1] - expected_bounds[0]
            )
            bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
            action = (
                env.action_space.low
                + (env.action_space.high - env.action_space.low) * bounded_percent
            )[0]
            
            # Step environment
            obs, reward, terminated, truncated, info = env.step(action)
            obs = preprocessor.modify_state(obs, info).squeeze()
            done = terminated or truncated
            episode_reward += reward
            steps += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode+1:2d} | Reward: {episode_reward:8.2f} | Steps: {steps:4d}")
    
    # Summary statistics
    print("-"*60)
    avg_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    max_reward = np.max(episode_rewards)
    min_reward = np.min(episode_rewards)
    
    print("\n" + "="*60)
    print("TEST RESULTS")
    print("="*60)
    print(f"Average Reward: {avg_reward:8.2f}")
    print(f"Std Dev:        {std_reward:8.2f}")
    print(f"Max Reward:     {max_reward:8.2f}")
    print(f"Min Reward:     {min_reward:8.2f}")
    print(f"Episodes:       {num_episodes}")
    print("="*60)
    
    env.close()
    return avg_reward


if __name__ == "__main__":
    # Find checkpoint
    script_dir = Path(__file__).parent
    # checkpoint_path = script_dir.parent / "exp_local" / "booster" / "chekpoints"/"run8"/"baseline_ddpg_final.pt"
    checkpoint_path = script_dir.parent / "exp_local" / "booster" / "chekpoints"/"run8"/"submission_model.pt"
    
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found at: {checkpoint_path}")
        print("Run baseline_training.py first to train the model.")
        exit(1)
    
    test_model(str(checkpoint_path), num_episodes=10)
