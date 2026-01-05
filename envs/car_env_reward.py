import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from render.pygame_render import PygameRenderer
from learnings.ray_casting.ray_lauchement import generate_rays
from learnings.ray_casting.intersections import ray_distance
from tracks.track_geometry import build_walls_with_aabb
from envs.rewards.progress_reward import ProgressReward

class SimpleCarEnv(gym.Env):
    metadata = {"render_modes": ["human", None]}

    def __init__(self, spawn, walls, track, centerline, track_width=None, nbr_rays=None, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.spawn = spawn
        self.nbr_rays = nbr_rays
        self.track_width = track_width
        self.n_dist_bin = 7
        self.centerline = centerline
        self.reward_fn = ProgressReward(self.centerline)

        if self.render_mode == "human":
            self.renderer = PygameRenderer()
        else:
            self.renderer = None

        # Dessiner les murs
        self.walls = build_walls_with_aabb(walls)
        self.track = track

        # Actions discrètes
        self.action_space = spaces.Discrete(5)
        # Observation discrète (reliées au nombre de rayons)
        obs_dims = [self.n_dist_bin] * self.nbr_rays + [5,8]
        self.observation_space = spaces.MultiDiscrete(obs_dims)  # distances pour chaque rayons + vitesse + angle

        # Init de ray et ray distance comme None, comme ça il n'embête pas pour le _get_obs dans le reset
        self.rays = None
        self.ray_distance = [1.0] * self.nbr_rays

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x, self.y, self.theta = self.spawn
        self.v = 1.0
        self.steps = 0

        obs = self._get_obs()
        self.reward_fn.reset({"position" : (self.x, self.y)})
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

        if self.track_width is not None:
            max_range = 3.0 * self.track_width
            self.ray_distance_norm = [min(d/max_range, 1.0) for d in self.ray_distance]

        terminated = self._collision()
        car_state = {"position": (self.x, self.y),
                     "speed": self.v,
                     "heading": self.theta}
        reward = self.reward_fn.step(car_state, terminated)


        obs = self._get_obs()
        if self.render_mode == "human":
            if self.nbr_rays is not None:
                self.renderer.render(self.x, self.y, self.theta, self.track, self.walls, rays=self.rays, ray_distance=self.ray_distance)
            else: 
                self.renderer.render(self.x, self.y, self.theta, self.track, self.walls)


        return obs, reward, terminated, False, {}

    def _get_obs(self):
        # Version simplifiée pour l’instant
        if self.nbr_rays is not None:
            dist_bins = self._discretize_distances()
        else:
            dist_bins = [self.n_dist_bin - 1] * self.nbr_rays

        v_bin = min(int(self.v), 4)
        theta_bin = int((self.theta % (2*math.pi)) / (2*math.pi / 8))
        theta_bin = min(theta_bin, 7)

        return np.array(dist_bins + [v_bin, theta_bin], dtype=int) 

    def _collision(self): 
        return not self.track.is_inside(self.x, self.y) 
    
    def close(self):
        if self.renderer:
            self.renderer.close()
    
    def _discretize_distances(self):
        if self.ray_distance is not None:
            return [
                min(int(d * self.n_dist_bin), self.n_dist_bin - 1)
                for d in self.ray_distance
            ]
        else:
            return [None]


