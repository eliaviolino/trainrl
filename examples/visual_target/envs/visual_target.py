import gymnasium as gym
import numpy as np
from collections import deque

class VisualTargetEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        super().__init__()

        config = config or {}
        self.max_steps = int(config.get("max_steps", 100))
        self.image_size = int(config.get("image_size", 64))

        # Momentum: False -> each action moves the agent 1 pixel (the frame shows the full state)
        #           True  -> each action changes the velocity, and the agent must stop on the target.
        #                    Velocity is not visible in a single frame, so the policy needs history.
        self.momentum = bool(config.get("momentum", False))
        self.max_speed = int(config.get("max_speed", 2))

        # Pixel format: float32 in [0, 1] (default) or uint8 in [0, 255] (4x less memory per frame)
        self.uint8_obs = bool(config.get("uint8_obs", False))
        self._pixel_max = 255 if self.uint8_obs else 1.0
        obs_dtype = np.uint8 if self.uint8_obs else np.float32
        self._obs_buffer = np.zeros((self.image_size, self.image_size, 3), dtype=obs_dtype)

        # Observation history: None -> obs is the current frame, shape (H, W, 3)
        #                      N    -> obs is the last N frames, shape (N, H, W, 3), oldest first
        self.history_len = config.get("history_len", None)

        # Observation Space: 64x64 RGB image (stacked N times if history_len is set)
        frame_shape = (self.image_size, self.image_size, 3)
        obs_shape = frame_shape if self.history_len is None else (self.history_len, *frame_shape)
        self.observation_space = gym.spaces.Box(low=0, high=self._pixel_max, shape=obs_shape, dtype=obs_dtype)
        # Action space: [left, right, up, down, up-left, up-right, down-left, down-right]
        self.action_space = gym.spaces.Discrete(8)


    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Random initial coordinates for the lander
        self.lander_x = int(self.np_random.integers(-32, 32))
        self.lander_y = int(self.np_random.integers(-32, 32))
        self.vel_x = 0
        self.vel_y = 0

        # Fill the history with the first frame (the agent starts at rest).
        # Deployment must pad the same way (see deployment/deploy_vit_temporal.py).
        if self.history_len is not None:
            self._history = deque([self._get_frame()] * self.history_len, maxlen=self.history_len)

        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1

        # 1. Unpack and apply action: a unit move in one of 8 directions
        dx, dy = {
            0: (-1, 0),   # left
            1: (1, 0),    # right
            2: (0, 1),    # up
            3: (0, -1),   # down
            4: (-1, 1),   # up-left
            5: (1, 1),    # up-right
            6: (-1, -1),  # down-left
            7: (1, -1),   # down-right
        }[int(action)]

        if self.momentum:
            # The move is an acceleration: velocity accumulates, position follows velocity
            self.vel_x = int(np.clip(self.vel_x + dx, -self.max_speed, self.max_speed))
            self.vel_y = int(np.clip(self.vel_y + dy, -self.max_speed, self.max_speed))
            self.lander_x += self.vel_x
            self.lander_y += self.vel_y
        else:
            self.lander_x += dx
            self.lander_y += dy

        # 2. Compute reward (negative distance to target)
        target_x, target_y = 0, 0  # fixed target at the center
        distance = np.sqrt((self.lander_x - target_x) ** 2 + (self.lander_y - target_y) ** 2)
        # reward = -0.001 * distance
        reward = -0.001

        # 3. Check termination (if we are close enough to the target)
        target_reached = distance < 1.0
        if self.momentum:
            target_reached = target_reached and self.vel_x == 0 and self.vel_y == 0  # must stop on it
        out_bound = self.lander_x < -32 or self.lander_x > 32 or self.lander_y < -32 or self.lander_y > 32
        terminated = target_reached or out_bound
        truncated = self.steps >= self.max_steps

        if target_reached:
            reward += 100.0  # bonus for reaching the target
        if out_bound:
            reward -= 100.0  # penalty for going out of bounds

        if self.history_len is not None:
            self._history.append(self._get_frame())

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        if self.history_len is None:
            return self._get_frame()
        return np.stack(self._history)  # (history_len, H, W, 3), oldest first

    def _get_frame(self):
        # The observation is a 64x64 RGB image with a red dot representing the lander and a green dot representing the target
        self._obs_buffer[:] = 0  # clear buffer
        center = self.image_size // 2

        # Draw target (green dot)
        target_x, target_y = 0, 0
        self._obs_buffer[center - target_y, center + target_x] = [0, self._pixel_max, 0]  # green

        # Draw lander (red dot)
        lander_row = np.clip(center - self.lander_y, 0, self.image_size-1)
        lander_col = np.clip(center + self.lander_x, 0, self.image_size-1)
        self._obs_buffer[lander_row, lander_col] = [self._pixel_max, 0, 0]  # red

        return self._obs_buffer.copy()  # return a copy to prevent external modification

    def _get_info(self):
        # Helper: diagnostic info (not used for training)
        return {"lander_x": self.lander_x, "lander_y": self.lander_y, "vel_x": self.vel_x, "vel_y": self.vel_y}
    
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