"""
TD3 Reinforcement Learning Training for Booster Soccer Showdown

This script trains a TD3 agent on one or more SAI soccer environments.
The agent learns through trial-and-error using environment rewards.

Key features:
- Multi-task environment support
- Pre-trained model initialization
- Checkpoint saving every 5K steps
- W&B logging for monitoring

Usage:
    python train_td3_rl.py --env LowerT1GoaliePenaltyKick-v0 --timesteps 5000000 --use_wandb
"""

import os
import sys
import argparse
import numpy as np
import torch
import gymnasium as gym
from datetime import datetime

# FIX: Set environment variable BEFORE importing sai_mujoco
# This allows sai_rl to be flexible with body naming conventions
os.environ['SAI_MUJOCO_ALLOW_MISSING_BODIES'] = '1'

# Add repo root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import SAI environments
import sai_mujoco  # noqa: F401

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import CheckpointCallback

# Local imports
from multi_task_env import MultiTaskWrapper

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("[WARNING] wandb not available. Install with: pip install wandb")


def main(args):
    """Main training function."""
    
    # Setup timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Setup directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # Setup W&B logging
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.init(
            project="booster-soccer",
            name=f"td3_rl_{timestamp}",
            config=vars(args),
            sync_tensorboard=True,
        )
        print(f"[W&B] Logging to: {wandb.run.url}")
    
    # Create environment
    print(f"\n[Environment] Creating multi-task wrapper...")
    env_names = [args.env]  # Can be extended to multiple
    if args.second_env:
        env_names.append(args.second_env)
    
    env = MultiTaskWrapper(env_names)
    
    # Load or create model
    print(f"\n[Model] Initializing TD3 model with DEEP & LARGE network architecture...")
    
    # Define deep and large network architecture for faster grokking
    policy_kwargs = {
        'net_arch': {
            'pi': [512, 256, 128, 64],   # Actor: Deep & Large (4 layers)
            'qf': [512, 256, 128, 64],   # Critic: Deep & Large (4 layers)
        },
        'activation_fn': torch.nn.ReLU,
    }
    
    print(f"[Model] Network Configuration:")
    print(f"  Actor network:  {policy_kwargs['net_arch']['pi']}")
    print(f"  Critic network: {policy_kwargs['net_arch']['qf']}")
    print(f"  Activation: ReLU")
    
    model = TD3(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        learning_starts=args.learning_starts,
        tau=args.tau,
        gamma=args.gamma,
        policy_delay=args.policy_delay,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
        policy_kwargs=policy_kwargs,
        verbose=1,
        tensorboard_log=f"{args.save_dir}/tensorboard",
        device=args.device,
    )
    
    # Load pre-trained weights if provided
    if args.pretrained_path and os.path.exists(args.pretrained_path):
        print(f"[Model] Loading pre-trained weights from: {args.pretrained_path}")
        try:
            # Try loading as PyTorch model
            checkpoint = torch.load(args.pretrained_path, map_location=args.device)
            
            if isinstance(checkpoint, dict):
                # Handle different checkpoint formats
                if 'policy_state_dict' in checkpoint:
                    model.policy.load_state_dict(checkpoint['policy_state_dict'])
                    print("[Model] ✓ Loaded pre-trained policy weights")
                elif 'actor' in checkpoint:
                    # Custom format
                    model.actor.load_state_dict(checkpoint['actor'])
                    if 'critic' in checkpoint:
                        model.critic.load_state_dict(checkpoint['critic'])
                    print("[Model] ✓ Loaded pre-trained actor/critic weights")
                else:
                    print(f"[WARNING] Checkpoint format not recognized. Starting fresh.")
            else:
                print(f"[WARNING] Pre-trained file is not a dict. Starting fresh.")
        except Exception as e:
            print(f"[WARNING] Could not load pre-trained weights: {e}")
            print("[Model] Starting fresh from random initialization")
    else:
        print(f"[Model] No pre-trained path provided, training from scratch")
    
    # Setup checkpoint callback (save every 5K steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_interval,
        save_path=args.checkpoint_dir,
        name_prefix="td3_checkpoint",
        save_replay_buffer=True,
    )
    
    print(f"\n[Training] Starting TD3 training...")
    print(f"  Total timesteps: {args.timesteps:,}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.learning_rate}")
    print(f"  Checkpoint interval: {args.checkpoint_interval:,} steps")
    print(f"  Save directory: {args.save_dir}")
    print()
    
    # Train model
    model.learn(
        total_timesteps=args.timesteps,
        callback=checkpoint_callback,
        log_interval=args.log_interval,
        progress_bar=True,
    )
    
    # Save final model
    final_path = os.path.join(args.save_dir, "td3_final_model")
    model.save(final_path)
    print(f"\n[Model] ✓ Saved final model to: {final_path}.zip")
    
    # Save model weights as PyTorch
    torch.save({
        'policy_state_dict': model.policy.state_dict(),
        'actor_state_dict': model.actor.state_dict(),
        'critic_state_dict': model.critic.state_dict(),
    }, os.path.join(args.save_dir, "td3_final_model.pt"))
    print(f"[Model] ✓ Saved PyTorch weights to: {args.save_dir}/td3_final_model.pt")
    
    # Cleanup
    env.close()
    
    if WANDB_AVAILABLE and args.use_wandb:
        wandb.finish()
    
    print(f"\n[Training] ✓ Training complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train TD3 agent on Booster Soccer environments"
    )
    
    # Environment args
    parser.add_argument(
        "--env",
        type=str,
        default="LowerT1GoaliePenaltyKick-v0",
        help="Primary environment name"
    )
    parser.add_argument(
        "--second_env",
        type=str,
        default=None,
        help="Secondary environment for multi-task (optional)"
    )
    
    # Model args
    parser.add_argument(
        "--pretrained_path",
        type=str,
        default="converted_model.pt",
        help="Path to pre-trained PyTorch model"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "mps"],
        help="Device to use for training"
    )
    
    # Training args
    parser.add_argument(
        "--timesteps",
        type=int,
        default=5000000,
        help="Total training timesteps (5M = ~5 hours on GPU)"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for actor and critic"
    )
    parser.add_argument(
        "--buffer_size",
        type=int,
        default=1000000,
        help="Size of replay buffer"
    )
    parser.add_argument(
        "--learning_starts",
        type=int,
        default=10000,
        help="Timesteps before learning starts"
    )
    
    # TD3 specific args
    parser.add_argument(
        "--tau",
        type=float,
        default=0.005,
        help="Soft update coefficient for target networks"
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor"
    )
    parser.add_argument(
        "--policy_delay",
        type=int,
        default=2,
        help="Policy is updated every policy_delay steps"
    )
    
    # Checkpoint args
    parser.add_argument(
        "--checkpoint_interval",
        type=int,
        default=5000,
        help="Save checkpoint every N steps (5K steps = every ~5 minutes)"
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=100,
        help="Logging interval (in episodes)"
    )
    
    # Output args
    parser.add_argument(
        "--save_dir",
        type=str,
        default="./exp_td3_rl",
        help="Directory to save models and logs"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="./exp_td3_rl/checkpoints",
        help="Directory to save checkpoints"
    )
    
    # W&B args
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Log to Weights & Biases"
    )
    
    args = parser.parse_args()
    main(args)
