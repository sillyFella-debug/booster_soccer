import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym

# Make repo root importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(repo_root)

import sai_mujoco
from booster_control.t1_utils import LowerT1JoyStick
from imitation_learning.scripts.preprocessor import Preprocessor

def get_task_one_hot(env_name):
    if "GoaliePenaltyKick" in env_name:
        task_one_hot = np.array([1.0, 0.0, 0.0])
    elif "ObstaclePenaltyKick" in env_name:
        task_one_hot = np.array([0.0, 1.0, 0.0])
    elif "KickToTarget" in env_name:
        task_one_hot = np.array([0.0, 0.0, 1.0])
    else:
        task_one_hot = np.array([0.0, 0.0, 0.0])
    return task_one_hot

def main(args):
    # Load model
    print(f"Loading PyTorch model from {args.model_path}...")
    model = torch.jit.load(args.model_path)
    model.eval()

    # Build env
    print(f"Initializing environment: {args.env_name}...")
    env = gym.make(args.env_name, render_mode="human")
    lower_t1_robot = LowerT1JoyStick(env.unwrapped)
    preprocessor = Preprocessor()
    task_one_hot = get_task_one_hot(args.env_name)

    observation, info = env.reset()
    
    print("Starting simulation loop. Close the MuJoCo window to stop.")
    try:
        while True:
            # Preprocess
            preprocessed_observation = preprocessor.modify_state(observation.copy(), info.copy(), task_one_hot)
            
            # Inference
            obs_tensor = torch.tensor(preprocessed_observation, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                # The model returns (mean, std)
                mean, _ = model(obs_tensor)
            
            action = mean.squeeze(0).numpy()
            
            # Control
            ctrl = lower_t1_robot.get_torque(observation, action)
            observation, reward, terminated, truncated, info = env.step(ctrl)

            if terminated or truncated:
                observation, info = env.reset()
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        env.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='LowerT1GoaliePenaltyKick-v0', help='Environment name.')
    parser.add_argument('--model_path', type=str, default='./converted_model.pt', help='Path to converted TorchScript model.')
    args = parser.parse_args()
    main(args)
