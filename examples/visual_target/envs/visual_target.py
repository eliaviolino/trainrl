import gymnasium as gym
import numpy as np

class VisualTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        super().__init__()

        config = config or {}
        self.max_steps = int(config.get("max_steps", 100))
        self.image_size = int(config.get("image_size", 64))
        self._obs_buffer = np.zeros((self.image_size, self.image_size, 3), dtype=np.float32)

        # Observation Space: 64x64 RGB image
        self.observation_space = gym.spaces.Box(low=0, high=1.0, shape=(self.image_size, self.image_size, 3), dtype=np.float32)
        # Action space: [left, right, up, down, up-left, up-right, down-left, down-right]
        self.action_space = gym.spaces.Discrete(8)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Random initial coordinates for the lander
        self.lander_x = int(self.np_random.integers(-32, 32))
        self.lander_y = int(self.np_random.integers(-32, 32))

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1

        # 1. Unpack and apply action
        if action == 0:   # left
            self.lander_x -= 1
        elif action == 1: # right
            self.lander_x += 1
        elif action == 2: # up
            self.lander_y += 1
        elif action == 3: # down
            self.lander_y -= 1
        elif action == 4: # up-left
            self.lander_x -= 1
            self.lander_y += 1
        elif action == 5: # up-right
            self.lander_x += 1
            self.lander_y += 1
        elif action == 6: # down-left
            self.lander_x -= 1
            self.lander_y -= 1
        elif action == 7: # down-right
            self.lander_x += 1
            self.lander_y -= 1

        # 2. Compute reward (negative distance to target)
        target_x, target_y = 0, 0  # fixed target at the center
        distance = np.sqrt((self.lander_x - target_x) ** 2 + (self.lander_y - target_y) ** 2)
        # reward = -0.001 * distance
        reward = -0.001

        # 3. Check termination (if we are close enough to the target)
        target_reached = distance < 1.0
        out_bound = self.lander_x < -32 or self.lander_x > 32 or self.lander_y < -32 or self.lander_y > 32
        terminated = target_reached or out_bound
        truncated = self.steps >= self.max_steps

        if target_reached:
            reward += 100.0  # bonus for reaching the target
        if out_bound:
            reward -= 100.0  # penalty for going out of bounds

        return self._get_obs(), reward, terminated, truncated, self._get_info()


    def _get_obs(self):
        # The observation is a 64x64 RGB image with a red dot representing the lander and a green dot representing the target
        self._obs_buffer[:] = 0  # clear buffer
        center = self.image_size // 2

        # Draw target (green dot)
        target_x, target_y = 0, 0
        self._obs_buffer[center - target_y, center + target_x] = [0, 1.0, 0]  # green

        # Draw lander (red dot)
        lander_row = np.clip(center - self.lander_y, 0, self.image_size-1)
        lander_col = np.clip(center + self.lander_x, 0, self.image_size-1)
        self._obs_buffer[lander_row, lander_col] = [1.0, 0.0, 0.0]

        return self._obs_buffer.copy()  # return a copy to prevent external modification

    def _get_info(self):
        # Helper: diagnostic info (not used for training)
        return {"lander_x": self.lander_x, "lander_y": self.lander_y}
    
if __name__ == "__main__":
    env = VisualTargetEnv({"max_steps": 100})
    obs, _ = env.reset(seed=0)
    
    terminated = False
    truncated  = False
    steps      = 0

    while not (terminated or truncated):
        action = env.action_space.sample()  # random action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        print(f"Step: {steps}, Action: {action}, Reward: {reward:.2f}, Lander Pos: ({info['lander_x']}, {info['lander_y']})")
    
    print(f"Episode ended after {steps} steps. Terminated: {terminated}, Truncated: {truncated}")