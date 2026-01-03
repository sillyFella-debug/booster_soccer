"""
DDPG baseline training with Gaussian actor matching converted_model.pt architecture.
Loads pre-trained BC model weights as initialization.
Saves checkpoints every 5K steps to exp_local/booster/checkpoints/
"""
import torch
import numpy as np
import random
from pathlib import Path
from collections import deque

from sai_rl import SAIClient
# from ddpg_gaussian import DDPG_Gaussian
from hybrid_ddpg import DDPG_Gaussian
from tqdm import tqdm
from reward_wrapper import WakeUpWrapper

def train_with_checkpoints(env, model, action_function, preprocessor, timesteps=500000, 
                           checkpoint_dir=None, checkpoint_interval=5000):
    """Training loop with periodic checkpointing."""
    
    if checkpoint_dir is None:
        checkpoint_dir = Path("./checkpoints")
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    replay_buffer = deque(maxlen=100000)
    batch_size = 64
    update_frequency = 4
    
    total_steps = 0
    episode_count = 0
    pbar = tqdm(total=timesteps, desc="Training", unit="steps")
    
    while total_steps < timesteps:
        done = False
        obs, info = env.reset()
        obs = preprocessor.modify_state(obs, info).squeeze()
        episode_reward = 0.0
        episode_steps = 0
        
        while not done and total_steps < timesteps:
            # Select action
            state_tensor = torch.from_numpy(np.expand_dims(obs, axis=0)).float()
            with torch.no_grad():
                policy_output = model(state_tensor).detach().numpy()
            
            # Map to action space
            action = action_function(policy_output)[0].squeeze()
            
            # Add exploration noise
            noise = np.random.normal(0, 0.1, size=action.shape)
            action = np.clip(action + noise, -1, 1)
            
            # Step environment
            new_obs, reward, terminated, truncated, info = env.step(action)
            new_obs = preprocessor.modify_state(new_obs, info).squeeze()
            # === INSERT DEBUG PRINTS HERE ===
            # if total_steps % 5000 == 0: # Print every 500 steps
            #     print(f"\n--- DEBUG STEP {total_steps} ---")
            #     print(f"Action (First 3): {action[:3]}") # Check if robot is trying to move
            #     print(f"Reward: {reward}")
            #     print(f"Obs Shape: {new_obs.shape}") # Should be (89,)
            #     print(f"Done: {terminated or truncated}")
            #     print("------------------------------")
            # ================================
            done = terminated or truncated
            
            episode_reward += reward
            episode_steps += 1
            
            # Add to replay buffer
            replay_buffer.append((obs, action, reward, new_obs, done))
            obs = new_obs
            total_steps += 1
            pbar.update(1)
            
            # Train
            if len(replay_buffer) >= batch_size and total_steps % update_frequency == 0:
                batch = random.sample(replay_buffer, batch_size)
                states, actions, rewards, next_states, dones = zip(*batch)
                
                critic_loss, actor_loss = model.train(
                    np.array(states),
                    np.array(actions),
                    np.array(rewards).reshape(-1, 1),
                    np.array(next_states),
                    np.array(dones).reshape(-1, 1),
                    1
                )
                
                pbar.set_description(
                    f"Ep {episode_count} | Reward: {episode_reward:7.2f} | "
                    f"Critic: {critic_loss:.4f} | Actor: {actor_loss:.4f}"
                )
            
            # Save checkpoint
            if total_steps % checkpoint_interval == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_{total_steps}.pt"
                torch.save({
                    'actor': model.actor.state_dict(),
                    'critic': model.critic.state_dict(),
                    'target_actor': model.target_actor.state_dict(),
                    'target_critic': model.target_critic.state_dict(),
                    'timesteps': total_steps,
                    'episode': episode_count,
                }, ckpt_path)
                print(f"\n✓ Checkpoint saved: {ckpt_path}")
        
        episode_count += 1
    
    pbar.close()


class Preprocessor():
    """State preprocessing for multi-task observation handling."""

    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        """Rotate vector v by inverse of quaternion q."""
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
        """Augment observation with sensor fusion and task info."""
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, axis=0)

        task_onehot = self.get_task_onehot(info)
        if len(task_onehot.shape) == 1:
            task_onehot = np.expand_dims(task_onehot, axis=0)
        
        # Expand all info arrays to batch dimension
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis=0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis=0)
            info["robot_accelerometer"] = np.expand_dims(info["robot_accelerometer"], axis=0)
            info["robot_velocimeter"] = np.expand_dims(info["robot_velocimeter"], axis=0)
            info["goal_team_0_rel_robot"] = np.expand_dims(info["goal_team_0_rel_robot"], axis=0)
            info["goal_team_1_rel_robot"] = np.expand_dims(info["goal_team_1_rel_robot"], axis=0)
            info["goal_team_0_rel_ball"] = np.expand_dims(info["goal_team_0_rel_ball"], axis=0)
            info["goal_team_1_rel_ball"] = np.expand_dims(info["goal_team_1_rel_ball"], axis=0)
            info["ball_xpos_rel_robot"] = np.expand_dims(info["ball_xpos_rel_robot"], axis=0) 
            info["ball_velp_rel_robot"] = np.expand_dims(info["ball_velp_rel_robot"], axis=0) 
            info["ball_velr_rel_robot"] = np.expand_dims(info["ball_velr_rel_robot"], axis=0) 
            info["player_team"] = np.expand_dims(info["player_team"], axis=0)
            info["goalkeeper_team_0_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_xpos_rel_robot"], axis=0)
            info["goalkeeper_team_0_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_velp_rel_robot"], axis=0)
            info["goalkeeper_team_1_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_xpos_rel_robot"], axis=0)
            info["goalkeeper_team_1_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_velp_rel_robot"], axis=0)
            info["target_xpos_rel_robot"] = np.expand_dims(info["target_xpos_rel_robot"], axis=0)
            info["target_velp_rel_robot"] = np.expand_dims(info["target_velp_rel_robot"], axis=0)
            info["defender_xpos"] = np.expand_dims(info["defender_xpos"], axis=0)
        
        robot_qpos = obs[:,:12]
        robot_qvel = obs[:,12:24]
        quat = info["robot_quat"]
        base_ang_vel = info["robot_gyro"]
        project_gravity = self.quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        
        obs = np.hstack((robot_qpos, 
                         robot_qvel,
                         project_gravity,
                         base_ang_vel,
                         info["robot_accelerometer"],
                         info["robot_velocimeter"],
                         info["goal_team_0_rel_robot"], 
                         info["goal_team_1_rel_robot"], 
                         info["goal_team_0_rel_ball"], 
                         info["goal_team_1_rel_ball"], 
                         info["ball_xpos_rel_robot"], 
                         info["ball_velp_rel_robot"], 
                         info["ball_velr_rel_robot"], 
                         info["player_team"], 
                         info["goalkeeper_team_0_xpos_rel_robot"], 
                         info["goalkeeper_team_0_velp_rel_robot"], 
                         info["goalkeeper_team_1_xpos_rel_robot"], 
                         info["goalkeeper_team_1_velp_rel_robot"], 
                         info["target_xpos_rel_robot"], 
                         info["target_velp_rel_robot"], 
                         info["defender_xpos"],
                         task_onehot))
  # --- NEW PADDING LOGIC ---
        target_dim = 89 #89  # The magic number your model wants
        current_dim = obs.shape[-1]
        
        if current_dim < target_dim:
            # Calculate how much is missing (likely 2, since 87 + 2 = 89)
            diff = target_dim - current_dim
            padding = np.zeros(diff)
            
            if len(obs.shape) == 1:
                obs = np.concatenate([obs, padding])
            else:
                # Handle batch padding if needed
                padding_batch = np.zeros((obs.shape[0], diff))
                obs = np.hstack((obs, padding_batch))
        
        # Safety Clip: If we somehow have too many, cut them off
        elif current_dim > target_dim:
             obs = obs[..., :target_dim]
    # --- END NEW PADDING LOGIC ---
        return obs

def main():
    print("="*60)
    print("BOOSTER SOCCER BASELINE RL TRAINING")
    print("With pre-trained BC weights (Gaussian Actor)")
    print("="*60)
    
    # Initialize environment
    # print("\n[1/5] Initializing SAI client for LowerT1GoaliePenaltyKick-v0...")
    print("\n[1/5] Initializing SAI client for LowerT1KickTarget-v0...")
    # sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")
    sai = SAIClient(comp_id="lower-t1-kick-to-target")
    env = sai.make_env()
        # 2. APPLY THE WRAPPER
    print("Applying Custom Reward Shaping...")
    env = WakeUpWrapper(env)
    print(f"✓ Environment initialized")
    print(f"  Action space: {env.action_space}")
    
    # Create DDPG agent matching converted_model.pt architecture
    print("\n[2/5] Creating DDPG agent with [256, 256, 256] Gaussian actor...")
    model = DDPG_Gaussian(
        input_dim= 89,  #89 is original model input dim 87 Observation dimension (no task onehot from preprocessor)
        action_dim=12,
        actor_hidden=[256, 256, 256],  # Matches converted_model.pt
        critic_hidden=[256, 256],
        learning_rate=0.0003,
    )
    print("✓ DDPG model created")
    
    # Load pre-trained BC weights
    print("\n[3/5] Loading pre-trained BC model weights...")
    pretrained_path = Path(__file__).parent.parent / "converted_model.pt"
    # pretrained_path = Path(__file__).parent.parent / "exp_local" / "booster" / "checkpoints" / "run2" / "checkpoint_500000.pt"
    if model.load_pretrained_actor(str(pretrained_path)):
        print("✓ Pre-trained actor weights loaded successfully")
    else:
        print("⚠️  Failed to load pretrained weights")
    
    # Action function
    print("\n[4/5] Setting up training...")
    def action_function(policy):
        expected_bounds = [-1, 1]
        action_percent = (policy - expected_bounds[0]) / (
            expected_bounds[1] - expected_bounds[0]
        )
        bounded_percent = np.minimum(np.maximum(action_percent, 0), 1)
        return (
            env.action_space.low
            + (env.action_space.high - env.action_space.low) * bounded_percent
        )
    
    # Run training with checkpointing
    print("\n[5/5] Starting RL training loop...")
    print("  Architecture: [256, 256, 256] with LayerNorm + Gaussian output")
    print("  Total timesteps: 500,000")
    print("  Checkpoint interval: 5,000 steps")
    print("  Pre-trained: Yes (BC model)")
    # print("  Environment: LowerT1GoaliePenaltyKick-v0")
    print("-"*60)
    
    # Custom training loop with checkpointing
    train_with_checkpoints(
        env=env,
        model=model,
        action_function=action_function,
        preprocessor=Preprocessor(),
        timesteps=60000,
        checkpoint_dir=Path(__file__).parent.parent / "exp_local" / "booster" / "checkpoints" / "run9",
        checkpoint_interval=6000,
    )
    
    # Save final checkpoint
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    exp_dir = Path(__file__).parent.parent / "exp_local" / "booster" / "checkpoints" / "run9"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    final_checkpoint = exp_dir / "baseline_ddpg_final.pt"
    torch.save({
        'actor': model.actor.state_dict(),
        'critic': model.critic.state_dict(),
        'target_actor': model.target_actor.state_dict(),
        'target_critic': model.target_critic.state_dict(),
    }, final_checkpoint)
    print(f"✓ Final model checkpoint saved to: {final_checkpoint}")
    
    print(f"✓ All checkpoint files saved to: {exp_dir}/checkpoints/")
    print(f"  (Checkpoints saved every 5,000 steps)")
    
    env.close()
    print("✓ Environment closed")


if __name__ == "__main__":
    main()
