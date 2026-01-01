#!/usr/bin/env python3
"""
Simple training script using the existing training infrastructure.
Basic hyperparameter tuning without fancy notebook stuff.
"""

import sys
import torch
import torch.nn.functional as F
import numpy as np

# Add training_scripts to path
sys.path.insert(0, './training_scripts')

from sai_rl import SAIClient
from training_scripts.ddpg import DDPG_FF
from training_scripts.training import training_loop

print("=" * 70)
print("🚀 STARTING BASIC TRAINING")
print("=" * 70)

# Initialize the SAI client
print("\n[Setup] Initializing SAI client...")
sai = SAIClient(comp_id="lower-t1-penalty-kick-goalie")
env = sai.make_env()
print(f"[Setup] ✓ Environment created")
print(f"[Setup] Observation space: {env.observation_space.shape}")
print(f"[Setup] Action space: {env.action_space.shape}")

# Preprocessor class
class Preprocessor():
    def get_task_onehot(self, info):
        if 'task_index' in info:
            return info['task_index']
        else:
            return np.array([])

    def quat_rotate_inverse(self, q: np.ndarray, v: np.ndarray):
        q_w = q[:,[-1]]
        q_vec = q[:,:3]
        a = v * (2.0 * q_w**2 - 1.0)
        b = np.cross(q_vec, v) * (q_w * 2.0)
        c = q_vec * (np.dot(q_vec, v).reshape(-1,1) * 2.0)    
        return a - b + c 

    def modify_state(self, obs, info):
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

# HYPERPARAMETERS - Adjust these for tuning
HYPERPARAMS = {
    'n_features': 87,
    'neurons': [24, 12, 6],           # Network architecture
    'learning_rate': 0.0001,           # Try: 0.00005, 0.0001, 0.0002
    'timesteps': 50000,                # Start small to test
}

print("\n[Setup] Hyperparameters:")
for key, val in HYPERPARAMS.items():
    print(f"  {key}: {val}")

# Create the model
print("\n[Model] Creating DDPG model...")
model = DDPG_FF(
    n_features=HYPERPARAMS['n_features'],
    action_space=env.action_space,
    neurons=HYPERPARAMS['neurons'],
    activation_function=F.relu,
    learning_rate=HYPERPARAMS['learning_rate'],
)
print(f"[Model] ✓ Model created with {sum(p.numel() for p in model.parameters())} parameters")

# Define action function
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

# Train the model
print("\n[Training] Starting training loop...")
print(f"[Training] Total timesteps: {HYPERPARAMS['timesteps']}")
try:
    training_loop(
        env, 
        model, 
        action_function=action_function, 
        preprocess_class=Preprocessor,
        timesteps=HYPERPARAMS['timesteps']
    )
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETED!")
    print("=" * 70)
    
    # Save model
    torch.save(model.state_dict(), 'ddpg_trained_model.pt')
    print("[Model] ✓ Model saved to ddpg_trained_model.pt")
    
except KeyboardInterrupt:
    print("\n⚠️ Training interrupted by user")
    torch.save(model.state_dict(), 'ddpg_interrupted_model.pt')
    print("[Model] ✓ Checkpoint saved to ddpg_interrupted_model.pt")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
    try:
        torch.save(model.state_dict(), 'ddpg_error_checkpoint.pt')
        print("[Model] ✓ Error checkpoint saved")
    except:
        pass
