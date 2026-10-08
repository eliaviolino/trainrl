import gymnasium as gym
import numpy as np

class Lander2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, config=None):
        super().__init__()
        config = config or {}

        # Physics constants
        self.gravity = config.get("gravity", 0.5)
        self.dt = config.get("dt", 0.1)
        self.max_thrust = config.get("max_thrust", 1.0)
        self.max_steps = config.get("max_steps", 100)

        # Landing boundaries 
        self.land_pos = config.get("land_pos", 0.1)
        self.land_vel = config.get("land_vel", 0.1)
        self.boundary = config.get("boundary", 10.0)

        # Observation space: [x, y, vx, vy]
        obs_dimension = 4
        obs_high = np.array([self.boundary, self.boundary, 5.0, 5.0], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=-obs_high, high=obs_high, shape=(obs_dimension,), dtype=np.float32)

        # Action space: [thrust_x, thrust_y]
        action_dimension = 2
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dimension,), dtype=np.float32)
        # pass

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.steps = 0

        # Random start position and velocity
        self.x = self.np_random.uniform(-5, 5)
        self.y = self.np_random.uniform(0, 5)
        self.vx = 0.0
        self.vy = 0.0
        # Return: (obs, info)
        return self._get_obs(), self._get_info()

    def step(self, action):
        self.steps += 1

        # 1. Unpack and apply action
        ax = float(action[0]) * self.max_thrust
        ay = float(action[1]) * self.max_thrust

        # 2. Update velocities (hint: gravity only affects one axis)
        self.vx += ax * self.dt
        self.vy += (ay - self.gravity) * self.dt

        # 3. Update positions
        self.x += self.vx * self.dt
        self.y += self.vy * self.dt

        # 4. Compute reward
        reward = - 0.01 * (abs(self.x) + abs(self.y) + abs(self.vx) + abs(self.vy))  # negative distance and velocity

        # 5. Check termination conditions
        landed    = abs(self.x) < self.land_pos and abs(self.y) < self.land_pos and abs(self.vx) < self.land_vel and abs(self.vy) < self.land_vel  # close to origin, low velocity
        out_bound = abs(self.x) > self.boundary or abs(self.y) > self.boundary  # beyond boundary
        timeout   = self.steps >= self.max_steps  # steps exceeded max_steps

        if landed:
            reward += 100.0  # bonus for landing
        if out_bound:
            reward -= 100.0  # penalty for going out of bounds

        terminated = landed or out_bound
        truncated  = timeout

        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        return np.array([self.x, self.y, self.vx, self.vy], dtype=np.float32)

    def _get_info(self):
        return {"x": float(self.x), "y": float(self.y), "vx": float(self.vx), "vy": float(self.vy)}
    
if __name__ == "__main__":
    env = Lander2DEnv({"max_steps": 100})
    obs, _ = env.reset(seed=0)
    
    terminated = False
    truncated  = False
    steps      = 0

    while not (terminated or truncated):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        steps += 1
        print(f"step={steps:>3} | obs={obs} | reward={reward:.2f} | term={terminated} | trunc={truncated}")

    print(f"\nEpisode ended after {steps} steps")
    print(f"terminated={terminated}, truncated={truncated}")