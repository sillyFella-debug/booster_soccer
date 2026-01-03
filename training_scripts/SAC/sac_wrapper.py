import gymnasium as gym
import numpy as np

class StandingWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_dist = None
        
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 1. STANDING BONUS (The Fix for Curling)
        # We access the z-coordinate of the robot's base (height)
        # Usually found in info['robot_base_pos'][2] or similar, 
        # but let's approximate it using the raw observation if needed.
        # Most reliable way in this specific repo:
        if 'robot_base_pos' in info:
            height = info['robot_base_pos'][2]
            # Reward for keeping head above 0.5 meters
            reward += height * 2.0 
        
        # 2. JOINT LIMIT PENALTY (Stop Locking Joints)
        # If action is near -1 or 1, penalize heavily
        # This forces the robot to use the middle of its range (standing)
        action_magnitude = np.mean(np.abs(action))
        reward -= action_magnitude * 0.1

        # 3. ALIVE BONUS
        reward += 0.1

        # 4. MOVE TO BALL (Standard)
        if 'ball_xpos_rel_robot' in info:
            dist = np.linalg.norm(info['ball_xpos_rel_robot'])
            if self.last_dist is not None:
                reward += (self.last_dist - dist) * 50.0
            self.last_dist = dist
            
        # 5. FALL PENALTY
        if done and info.get("robot_fallen", False):
            reward -= 5.0

        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        if 'ball_xpos_rel_robot' in info:
            self.last_dist = np.linalg.norm(info['ball_xpos_rel_robot'])
        return obs, info