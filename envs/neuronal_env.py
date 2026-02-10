import torch
import numpy as np
from tracks.track_geometry import build_walls_with_aabb # Tu pourras adapter ça pour récupérer juste les points
# Importe ton render plus tard

class VectorizedCarEnv:
    def __init__(self, spawn_points, walls, track_width,  n_cars=1000, n_rays=7, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_cars = n_cars
        self.n_rays = n_rays
        self.track_width = track_width
        
        # --- 1. Chargement des Murs sur GPU ---
        # walls est une liste de tuples [((x1,y1), (x2,y2)), ...]
        # On convertit en deux gros tenseurs
        starts = []
        ends = []
        for w in walls:
            starts.append(w[0])
            ends.append(w[1])
            
        self.wall_starts = torch.tensor(starts, dtype=torch.float32, device=self.device)
        self.wall_ends = torch.tensor(ends, dtype=torch.float32, device=self.device)
        
        # --- 2. Initialisation des Agents ---
        # Position (x, y)
        self.pos = torch.zeros((n_cars, 2), device=self.device)
        # Vitesse (scalaire)
        self.speed = torch.zeros((n_cars, 1), device=self.device)
        # Angle (radians)
        self.angle = torch.zeros((n_cars, 1), device=self.device)
        # Vivant ou crashé ?
        self.alive = torch.ones(n_cars, dtype=torch.bool, device=self.device)
        
        self.spawn_points = spawn_points # Liste de points de spawn
        self.reset()

        # Raycasting : Angles relatifs fixes
        # Ex: -90, -45, 0, 45, 90 degrés
        self.ray_angles = torch.linspace(-np.pi/2, np.pi/2, n_rays, device=self.device)

    def reset(self):
        """Remet toutes les voitures au départ"""
        # Pour l'instant, tout le monde au même endroit (ou dispersé si tu as plusieurs spawns)
        spawn = self.spawn_points[0] # Simplification
        self.pos[:, 0] = spawn[0]
        self.pos[:, 1] = spawn[1]
        self.speed[:] = 0
        self.angle[:] = 0 # Ou angle de départ du circuit
        self.alive[:] = True
        
        return self.get_observations()

    def step(self, actions):
        """
        actions: Tensor (N_cars, 2) -> [Volant, Gaz]
                 Volant entre -1 et 1
                 Gaz entre 0 et 1 (ou -1 et 1)
        """
        dt = 0.05 # Temps par frame
        
        # --- 1. Physique Vectorisée ---
        steering = actions[:, 0:1] # (N, 1)
        throttle = actions[:, 1:2] # (N, 1)
        
        # Mise à jour vitesse (simple pour commencer)
        # v = v + accel - frottement
        self.speed += throttle * 2.0 * dt 
        #self.speed *= 0.95 # Frottement de l'air pas encore pris en compte
        
        # Mise à jour angle
        # theta = theta + v * steering * maniabilité
        self.angle += self.speed * steering * 0.5 * dt
        
        # Mise à jour position
        # x = x + v * cos(theta)
        # y = y + v * sin(theta)
        dx = self.speed * torch.cos(self.angle) * dt
        dy = self.speed * torch.sin(self.angle) * dt
        
        # On ne bouge que ceux qui sont vivants (optionnel, mais plus propre)
        self.pos += torch.cat([dx, dy], dim=1) * self.alive.unsqueeze(1)

        # --- 2. Raycasting Vectorisé ---
        distances = self._compute_lidar()
        
        # --- 3. Détection de collision ---
        # Si une distance est très petite (< taille voiture), CRASH
        collision_dist = 0.1 # pixels
        crashes = (distances < collision_dist).any(dim=1) # Si au moins un rayon touche
        
        # Mise à jour état vivant
        self.alive = self.alive & (~crashes)
        
        # --- 4. Récompenses ---
        # À définir selon ta logique de Fitness
        rewards = self.speed.squeeze() * self.alive.float() # Ex: Vitesse = points
        
        return self.get_observations(), rewards, (~self.alive)

    def _compute_lidar(self):
        from learnings.ray_casting.gpu_raycasting import compute_ray_intersections
        
        # Calculer les angles absolus des rayons pour chaque voiture
        # (N_cars, 1) + (N_rays,) -> (N_cars, N_rays) via broadcast
        global_ray_angles = self.angle + self.ray_angles 
        
        # Directions des rayons (N_cars * N_rays, 2)
        # On aplatit pour envoyer à la fonction de calcul
        flat_angles = global_ray_angles.view(-1)
        ray_dirs = torch.stack([torch.cos(flat_angles), torch.sin(flat_angles)], dim=1)
        
        # Origines des rayons
        # On répète la position de chaque voiture N_rays fois
        ray_origins = self.pos.repeat_interleave(self.n_rays, dim=0)
        
        # Appel au moteur de calcul
        dists = compute_ray_intersections(
            ray_origins, ray_dirs, 
            self.wall_starts, self.wall_ends
        )
        
        # On remet sous forme (N_cars, N_rays)
        return dists.view(self.n_cars, self.n_rays)

    def get_observations(self):
        # Renvoie les distances normalisées + vitesse
        # (N_cars, N_rays + 1)
        rays = self._compute_lidar() / (3 * self.track_width) # Normalisation
        return torch.cat([rays, self.speed / 20.0], dim=1) # Vitesse normalisée
    
    def get_render_data(self):
        """Renvoie les données CPU pour l'affichage PyGame"""
        return {
            "pos": self.pos.cpu().numpy(),
            "angle": self.angle.cpu().numpy(),
            "alive": self.alive.cpu().numpy()
        }