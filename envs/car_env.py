import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from render.pygame_render import PygameRenderer


class SimpleCarEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, spawn, walls, track, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.spawn = spawn

        if self.render_mode == "human":
            self.renderer = PygameRenderer()
        else:
            self.renderer = None

        # Dessiner les murs
        self.walls = walls
        self.track = track

        # Actions discrètes
        self.action_space = spaces.Discrete(5)
        # Observation discrète (exemple)
        self.observation_space = spaces.MultiDiscrete(
            [5, 5, 5, 5, 8]  # distances + vitesse + angle
        )
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x, self.y, self.theta = self.spawn
        self.v = 1.0
        self.steps = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        self.steps += 1

        # Actions
        if action == 0:
            self.theta -= 0.1
        elif action == 2:
            self.theta += 0.1
        elif action == 3:
            self.v += 0.2
        elif action == 4:
            self.v -= 0.2

        self.v = np.clip(self.v, 0, 5)

        self.x += self.v * math.cos(self.theta)
        self.y += self.v * math.sin(self.theta)

        terminated = self._collision()
        reward = -100 if terminated else 1

        obs = self._get_obs()
        if self.render_mode == "human":
            self.renderer.render(self.x, self.y, self.theta, self.track, self.walls)


        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # Version simplifiée pour l’instant
        d_left = 2
        d_front = 2
        d_right = 2
        v_bin = min(int(self.v), 4)
        theta_bin = int((self.theta % (2*math.pi)) / (2*math.pi / 8))
        return np.array([d_left, d_front, d_right, v_bin, theta_bin])

    def _collision(self): 
        return not self.track.is_inside(self.x, self.y) 
    
    def close(self):
        if self.renderer:
            self.renderer.close()

    """def find_walls(self):
        self.walls = get_walls()"""
    

