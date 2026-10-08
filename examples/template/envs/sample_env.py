import gymnasium as gym
import numpy as np

class SampleEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        super().__init__()
        # TODO: define self.observation_space
        # TODO: define self.action_space
        # TODO: define physics constants (gravity, max_thrust, dt)
        pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        # TODO: initialize state
        # Return: (obs, info)
        pass

    def step(self, action):
        # TODO: apply action, update physics, compute reward
        # Return: (obs, reward, terminated, truncated, info)
        pass

    def _get_obs(self):
        # Helper: convert internal state → numpy array
        pass

    def _get_info(self):
        # Helper: diagnostic info (not used for training)
        pass