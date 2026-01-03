import torch
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

from sai_rl import SAIClient
from ddpg import DDPG_FF
from training import training_loop


def load_pretrained_weights(model, pretrained_path):
    """Load pre-trained weights into DDPG actor network."""
    if not os.path.exists(pretrained_path):
        print(f"⚠️  Pre-trained model not found at {pretrained_path}")
        return False
    
    try:
        print(f"Loading pre-trained model from: {pretrained_path}")
        
        # Try loading as TorchScript first
        try:
            scripted_module = torch.jit.load(pretrained_path, map_location='cpu')
            print("Detected TorchScript format")
            
            # Extract weights from TorchScript module
            state_dict = scripted_module.state_dict()
            print(f"Available keys in TorchScript: {list(state_dict.keys())}")
            
            # Try to load into actor network
            model.actor.load_state_dict(state_dict)
            print("✓ Successfully loaded TorchScript weights into actor network")
            return True
        except Exception as e:
            print(f"TorchScript loading failed: {e}")
            print("Attempting standard PyTorch checkpoint loading...")
        
        # Fallback to standard torch.load with weights_only=False
        checkpoint = torch.load(pretrained_path, map_location='cpu', weights_only=False)
        
        # Handle PyTorch checkpoint dict
        if isinstance(checkpoint, dict):
            if 'actor' in checkpoint:
                model.actor.load_state_dict(checkpoint['actor'])
                print("✓ Successfully loaded 'actor' weights")
            elif 'state_dict' in checkpoint:
                model.actor.load_state_dict(checkpoint['state_dict'])
                print("✓ Successfully loaded 'state_dict' weights")
            else:
                # Try loading directly as state dict
                model.actor.load_state_dict(checkpoint)
                print("✓ Successfully loaded checkpoint as state dict")
        else:
            print(f"Unexpected checkpoint type: {type(checkpoint)}")
            return False
        
        print("✓ Successfully loaded pre-trained weights into actor network")
        return True
    except Exception as e:
        print(f"Error loading pre-trained weights: {e}")
        import traceback
        traceback.print_exc()
        return False


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
        
        if len(info["robot_quat"].shape) == 1:
            info["robot_quat"] = np.expand_dims(info["robot_quat"], axis = 0)
            info["robot_gyro"] = np.expand_dims(info["robot_gyro"], axis = 0)
            info["robot_accelerometer"] = np.expand_dims(info["robot_accelerometer"], axis = 0)
            info["robot_velocimeter"] = np.expand_dims(info["robot_velocimeter"], axis = 0)
            info["goal_team_0_rel_robot"] = np.expand_dims(info["goal_team_0_rel_robot"], axis = 0)
            info["goal_team_1_rel_robot"] = np.expand_dims(info["goal_team_1_rel_robot"], axis = 0)
            info["goal_team_0_rel_ball"] = np.expand_dims(info["goal_team_0_rel_ball"], axis = 0)
            info["goal_team_1_rel_ball"] = np.expand_dims(info["goal_team_1_rel_ball"], axis = 0)
            info["ball_xpos_rel_robot"] = np.expand_dims(info["ball_xpos_rel_robot"], axis = 0) 
            info["ball_velp_rel_robot"] = np.expand_dims(info["ball_velp_rel_robot"], axis = 0) 
            info["ball_velr_rel_robot"] = np.expand_dims(info["ball_velr_rel_robot"], axis = 0) 
            info["player_team"] = np.expand_dims(info["player_team"], axis = 0)
            info["goalkeeper_team_0_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_xpos_rel_robot"], axis = 0)
            info["goalkeeper_team_0_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_0_velp_rel_robot"], axis = 0)
            info["goalkeeper_team_1_xpos_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_xpos_rel_robot"], axis = 0)
            info["goalkeeper_team_1_velp_rel_robot"] = np.expand_dims(info["goalkeeper_team_1_velp_rel_robot"], axis = 0)
            info["target_xpos_rel_robot"] = np.expand_dims(info["target_xpos_rel_robot"], axis = 0)
            info["target_velp_rel_robot"] = np.expand_dims(info["target_velp_rel_robot"], axis = 0)
            info["defender_xpos"] = np.expand_dims(info["defender_xpos"], axis = 0)
        
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

        return obs


def main():
    print("="*60)
    print("BOOSTER SOCCER BASELINE RL TRAINING")
    print("="*60)
    
    # Initialize SAI client for goalie penalty kick task
    print("\n[1/5] Initializing SAI client for LowerT1GoaliePenaltyKick-v0...")
    sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")
    env = sai.make_env()
    print(f"✓ Environment initialized: {env}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space}")
    
    # Create model with moderate network architecture
    print("\n[2/5] Creating DDPG agent with [128, 64, 128] network...")
    model = DDPG_FF(
        n_features=87,
        action_space=env.action_space,
        neurons=[128, 64, 128],
        activation_function=F.relu,
        learning_rate=0.0001,
    )
    print("✓ DDPG model created")
    print(f"  Actor: {model.actor}")
    print(f"  Critic: {model.critic}")
    
    # Load pre-trained weights from converted_model.pt
    print("\n[3/5] Loading pre-trained BC model weights...")
    # Use absolute path relative to script location
    script_dir = Path(__file__).parent
    pretrained_path = script_dir.parent / "converted_model.pt"
    if load_pretrained_weights(model, str(pretrained_path)):
        print("✓ Pre-trained weights loaded successfully")
    else:
        print("⚠️  Proceeding with random initialization")
    
    # Define action function to map policy output to env action space
    print("\n[4/5] Defining action function...")
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
    
    # Run training loop for 500K timesteps
    print("\n[5/5] Starting RL training loop...")
    print("  Network: [128, 64, 128]")
    print("  Total timesteps: 500,000")
    print("  Environment: LowerT1GoaliePenaltyKick-v0")
    print("  Learning rate: 0.0001")
    print("-"*60)
    
    training_loop(
        env=env,
        model=model,
        action_function=action_function,
        preprocess_class=Preprocessor,
        timesteps=500000
    )
    
    # Save trained model
    print("\n" + "="*60)
    print("Training complete!")
    print("="*60)
    
    checkpoint_path = "baseline_ddpg_checkpoint.pt"
    torch.save({
        'actor': model.actor.state_dict(),
        'critic': model.critic.state_dict(),
        'target_actor': model.target_actor.state_dict(),
        'target_critic': model.target_critic.state_dict(),
    }, checkpoint_path)
    print(f"✓ Model checkpoint saved to: {checkpoint_path}")
    
    env.close()
    print("✓ Environment closed")


if __name__ == "__main__":
    main()