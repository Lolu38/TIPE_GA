import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from render.pygame_render import PygameRenderer
from learnings.ray_casting.ray_lauchement import generate_rays
from learnings.ray_casting.intersections import ray_distance

class SimpleCarEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, spawn, walls, track, nbr_rays=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.spawn = spawn
        self.nbr_rays = nbr_rays

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
        
        self.rays = generate_rays((self.x, self.y), self.theta, self.nbr_rays)
        self.ray_distance = [ray_distance(ray["origin"], ray["direction"], self.walls) for ray in self.rays]

        terminated = self._collision()
        reward = -100 if terminated else 1

        obs = self._get_obs()
        if self.render_mode == "human":
            if self.nbr_rays is not None:
                self.renderer.render(self.x, self.y, self.theta, self.track, self.walls, rays=self.rays, ray_distance=self.ray_distance)
            else: 
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
    

