import gymnasium as gym
import numpy as np

class WakeUpWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.last_dist = None
        
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_dist = self._get_dist(info)
        return obs, info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # --- CUSTOM REWARD SHAPING ---
        
        # 1. Alive Bonus: Pay the robot just for not falling over.
        # This fixes the "Episode too short" fear.
        reward += 0.05 
        
        # 2. Distance Shaping: Pay it for moving towards the ball.
        # This fixes the "Standing Still" problem.
        current_dist = self._get_dist(info)
        
        if self.last_dist is not None and current_dist is not None:
            improvement = self.last_dist - current_dist
            # Multiply by 100 to make the signal strong enough for DDPG
            reward += improvement * 100.0 
            
        self.last_dist = current_dist
        
        # 3. Action Penalty: Penalize jerky, high-energy movements
        # (Optional: keeps movement smooth)
        reward -= np.sum(np.square(action)) * 0.01

        return obs, reward, done, truncated, info

    def _get_dist(self, info):
        # Extract ball distance from the info dictionary
        # Note: Keys depend on the specific environment version
        if 'ball_xpos_rel_robot' in info:
            # Calculate magnitude of the relative vector
            return np.linalg.norm(info['ball_xpos_rel_robot'])
        elif 'ball_distance' in info:
            return info['ball_distance']
        return None