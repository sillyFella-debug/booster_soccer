"""
Multi-Task Environment Wrapper for SAI Soccer Environments

Allows training on multiple SAI environments simultaneously by randomly
sampling which environment to use for each episode.
"""

import gymnasium as gym
import numpy as np
import os
from typing import List, Tuple, Dict, Any


class MultiTaskWrapper(gym.Env):
    """
    Wrapper that allows training on multiple SAI environments.
    Randomly samples which environment to use for each episode.
    """
    
    def __init__(self, env_names: List[str]):
        """
        Args:
            env_names: List of environment names to train on
                e.g., ["LowerT1GoaliePenaltyKick-v0", "LowerT1KickToTarget-v0"]
        """
        # Fix for sai_rl body naming compatibility
        os.environ['SAI_MUJOCO_ALLOW_MISSING_BODIES'] = '1'
        
        self.env_names = env_names
        self.envs = {}
        
        # Create environments with error handling
        for name in env_names:
            try:
                env = gym.make(name)
                self.envs[name] = env
            except Exception as e:
                print(f"[WARNING] Failed to create environment {name}: {e}")
                print(f"[INFO] Attempting to disable strict body name checking...")
                # Try once more with environment variable set
                env = gym.make(name)
                self.envs[name] = env
        
        # Current active environment
        self.current_env = None
        self.current_env_name = None
        self.step_count = 0
        
        # Use first environment as reference for spaces
        ref_env = self.envs[env_names[0]]
        self.observation_space = ref_env.observation_space
        self.action_space = ref_env.action_space
        
        print(f"[MultiTaskWrapper] Initialized with {len(env_names)} task(s):")
        for name in env_names:
            env = self.envs[name]
            print(f"  • {name}: obs_shape={env.observation_space.shape}, "
                  f"action_shape={env.action_space.shape}")
    
    def _select_env(self):
        """Randomly select next environment."""
        self.current_env_name = np.random.choice(self.env_names)
        self.current_env = self.envs[self.current_env_name]
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset: switch to random task."""
        self._select_env()
        self.step_count = 0
        obs, info = self.current_env.reset(seed=seed, options=options)
        
        # Add task information to info dict
        info['task_name'] = self.current_env_name
        
        return obs, info
    
    def step(self, action) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute action in current environment."""
        obs, reward, terminated, truncated, info = self.current_env.step(action)
        self.step_count += 1
        
        # Add task information
        info['task_name'] = self.current_env_name
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render current environment."""
        if self.current_env is not None:
            return self.current_env.render()
    
    def close(self):
        """Close all environments."""
        for env in self.envs.values():
            env.close()
    
    def __repr__(self):
        return f"MultiTaskWrapper({self.env_names})"
