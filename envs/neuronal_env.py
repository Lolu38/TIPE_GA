import torch
import numpy as np


class VectorizedCarEnv:
    """
    Environnement vectorisé pour simuler N voitures en parallèle sur GPU.
    
    Optimisé pour l'algorithme génétique avec:
    - Raycasting GPU ultra-rapide
    - Physique simple mais réaliste
    - Support pour checkpoints et détection de tours
    """
    
    def __init__(
        self,
        spawn_point,
        walls,
        track,
        track_width,
        n_cars=1000,
        n_rays=7,
        device='cuda',
        max_speed=20.0,
        collision_threshold=5.0
    ):
        """
        Args:
            spawn_point: Tuple (x, y, angle) - point de départ
            walls: Liste de segments [(p1, p2), ...] où p1, p2 sont des tuples (x, y)
            track: Objet track (RectangularTrack ou AngularTrack) pour vérifier les collisions
            track_width: Largeur du circuit (pour normalisation)
            n_cars: Nombre de voitures dans la population
            n_rays: Nombre de rayons de détection
            device: 'cuda' ou 'cpu'
            max_speed: Vitesse maximale (pixels/step)
            collision_threshold: Distance minimale avant collision (pixels)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_cars = n_cars
        self.n_rays = n_rays
        self.track_width = track_width
        self.max_speed = max_speed
        self.collision_threshold = collision_threshold
        
        # Point de spawn (avec angle)
        if len(spawn_point) == 3:
            self.spawn_x, self.spawn_y, self.spawn_angle = spawn_point
        else:
            self.spawn_x, self.spawn_y = spawn_point
            self.spawn_angle = 0.0
        
        # Track pour détection de collision
        self.track = track
        
        # --- 1. Chargement des Murs sur GPU ---
        starts = []
        ends = []
        for w in walls:
            starts.append(w[0])
            ends.append(w[1])
        
        self.wall_starts = torch.tensor(starts, dtype=torch.float32, device=self.device)
        self.wall_ends = torch.tensor(ends, dtype=torch.float32, device=self.device)
        
        # --- 2. Initialisation des États des Voitures ---
        self.pos = torch.zeros((n_cars, 2), device=self.device)
        self.speed = torch.zeros((n_cars, 1), device=self.device)
        self.angle = torch.zeros((n_cars, 1), device=self.device)
        self.alive = torch.ones(n_cars, dtype=torch.bool, device=self.device)
        
        # --- 3. Raycasting : Angles relatifs fixes ---
        self.ray_angles = torch.linspace(-np.pi/2, np.pi/2, n_rays, device=self.device)
        
        # --- 4. Paramètres physiques ---
        self.dt = 0.05  # Timestep
        self.friction = 0.98  # Frottement (0.98 = perd 2% de vitesse par frame)
        self.acceleration = 40.0  # Accélération (pixels/s²)
        self.steering_sensitivity = 0.8  # Sensibilité du volant
        
        # Initialiser au spawn
        self.reset()
    
    def reset(self, randomize_spawn=False, random_range=10.0, random_angle=0.1):
        """
        Remet toutes les voitures au départ
        
        Args:
            randomize_spawn: Si True, ajoute du bruit au spawn (évite surajustement)
            random_range: Amplitude du bruit de position (pixels)
            random_angle: Amplitude du bruit d'angle (radians)
        
        Returns:
            observations: Tensor (n_cars, n_rays + 1)
        """
        # Position de base
        self.pos[:, 0] = self.spawn_x
        self.pos[:, 1] = self.spawn_y
        
        # Randomisation pour éviter le sur-apprentissage
        if randomize_spawn:
            noise_pos = torch.randn((self.n_cars, 2), device=self.device) * random_range
            self.pos += noise_pos
            
            noise_angle = torch.randn((self.n_cars, 1), device=self.device) * random_angle
            self.angle[:] = self.spawn_angle + noise_angle
        else:
            self.angle[:] = self.spawn_angle
        
        # Reset vitesse et état
        self.speed.zero_()
        self.alive[:] = True
        
        return self.get_observations()
    
    def step(self, actions):
        """
        Avance d'un pas de temps
        
        Args:
            actions: Tensor (n_cars, 2)
                - actions[:, 0]: steering ∈ [-1, 1] (gauche/droite)
                - actions[:, 1]: throttle ∈ [0, 1] (freiner/accélérer)
        
        Returns:
            observations: Tensor (n_cars, n_rays + 1)
            rewards: Tensor (n_cars,) - non utilisé dans l'AG (on utilise fitness)
            dones: Tensor (n_cars,) - True si crashé
        """
        # --- 1. Extraire les commandes ---
        steering = actions[:, 0:1]  # (n_cars, 1)
        throttle = actions[:, 1:2]  # (n_cars, 1)
        
        # --- 2. Physique Vectorisée ---
        # Accélération
        acceleration = throttle * self.acceleration * self.dt
        self.speed += acceleration
        
        # Frottement
        self.speed *= self.friction
        
        # Limites de vitesse
        self.speed = torch.clamp(self.speed, 0.0, self.max_speed)
        
        # Mise à jour de l'angle (dépend de la vitesse)
        # Plus on va vite, plus le volant a d'effet
        angular_velocity = steering * self.steering_sensitivity * (self.speed / self.max_speed) * self.dt
        self.angle += angular_velocity
        
        # Normaliser l'angle dans [-π, π]
        self.angle = torch.atan2(torch.sin(self.angle), torch.cos(self.angle))
        
        # Mise à jour de la position
        dx = self.speed * torch.cos(self.angle) * self.dt
        dy = self.speed * torch.sin(self.angle) * self.dt
        
        # Appliquer le mouvement uniquement aux voitures vivantes
        movement = torch.cat([dx, dy], dim=1) * self.alive.unsqueeze(1).float()
        self.pos += movement
        
        # --- 3. Raycasting ---
        distances = self._compute_lidar()
        
        # --- 4. Détection de Collision ---
        # Collision si rayon trop court
        ray_collision = (distances < self.collision_threshold).any(dim=1)
        
        # Collision si hors du circuit (plus coûteux, donc on le fait après)
        track_collision = self._check_track_collision()
        
        # Mise à jour de l'état vivant
        crashes = ray_collision | track_collision
        self.alive = self.alive & (~crashes)
        
        # --- 5. Observations et Retour ---
        observations = self.get_observations()
        
        # Rewards (non utilisés dans l'AG, mais on les garde pour compatibilité)
        rewards = self.speed.squeeze() * self.alive.float()
        
        # Dones
        dones = ~self.alive
        
        return observations, rewards, dones
    
    def _compute_lidar(self):
        """
        Calcule les distances des rayons pour toutes les voitures
        
        Returns:
            distances: Tensor (n_cars, n_rays) - distances normalisées
        """
        from learnings.ray_casting.gpu_raycasting import compute_ray_intersections
        
        # Angles absolus des rayons pour chaque voiture
        global_ray_angles = self.angle + self.ray_angles
        
        # Aplatir pour le calcul vectorisé
        flat_angles = global_ray_angles.view(-1)
        ray_dirs = torch.stack([torch.cos(flat_angles), torch.sin(flat_angles)], dim=1)
        
        # Origines des rayons (répéter chaque position n_rays fois)
        ray_origins = self.pos.repeat_interleave(self.n_rays, dim=0)
        
        # Calcul des intersections
        max_range = 3.0 * self.track_width  # Distance max de détection
        dists = compute_ray_intersections(
            ray_origins,
            ray_dirs,
            self.wall_starts,
            self.wall_ends,
            max_dist=max_range
        )
        
        # Remettre en forme (n_cars, n_rays)
        return dists.view(self.n_cars, self.n_rays)
    
    def _check_track_collision(self):
        """
        Vérifie si les voitures sont sorties du circuit
        
        Returns:
            collisions: Tensor (n_cars,) - True si hors circuit
        """
        collisions = torch.zeros(self.n_cars, dtype=torch.bool, device=self.device)
        
        # Passer sur CPU pour utiliser track.is_inside (plus lent mais nécessaire)
        pos_cpu = self.pos.cpu().numpy()
        
        for i in range(self.n_cars):
            if self.alive[i]:  # Seulement vérifier les voitures vivantes
                x, y = pos_cpu[i]
                if not self.track.is_inside(x, y):
                    collisions[i] = True
        
        return collisions
    
    def get_observations(self):
        """
        Génère les observations pour toutes les voitures
        
        Returns:
            observations: Tensor (n_cars, n_rays + 1)
                - n_rays premières valeurs: distances normalisées [0, 1]
                - dernière valeur: vitesse normalisée [0, 1]
        """
        # Distances normalisées
        max_range = 3.0 * self.track_width
        rays = self._compute_lidar() / max_range
        rays = torch.clamp(rays, 0.0, 1.0)  # S'assurer que c'est dans [0, 1]
        
        # Vitesse normalisée
        speed_norm = self.speed / self.max_speed
        
        # Concaténer
        return torch.cat([rays, speed_norm], dim=1)
    
    def get_render_data(self):
        """
        Renvoie les données CPU pour l'affichage PyGame
        
        Returns:
            dict avec 'pos', 'angle', 'alive', 'speed'
        """
        return {
            'pos': self.pos.cpu().numpy(),
            'angle': self.angle.cpu().numpy(),
            'alive': self.alive.cpu().numpy(),
            'speed': self.speed.cpu().numpy()
        }
    
    def get_alive_count(self):
        """Retourne le nombre de voitures encore en vie"""
        return self.alive.sum().item()
    
    def is_all_dead(self):
        """Vérifie si toutes les voitures sont mortes"""
        return not self.alive.any()


def build_env_from_track_config(track_name='nascar', n_cars=1000, n_rays=9, device='cuda'):
    """
    Fonction utilitaire pour créer un environnement à partir d'un circuit prédéfini
    
    Args:
        track_name: 'nascar', 'rectangle', ou 'high_speed_ring'
        n_cars: Nombre de voitures
        n_rays: Nombre de rayons
        device: 'cuda' ou 'cpu'
    
    Returns:
        env: VectorizedCarEnv
        checkpoints: Liste de points pour FitnessTracker
    """
    if track_name == 'nascar':
        from tracks.nascar_ring import get_walls as gw, get_spawn as gs
        outer, inner = gw()
        spawn = gs()
        
        from tracks.track_geometry import AngularTrack, compute_centerline
        track = AngularTrack(outer, inner)
        
        # Calculer la largeur moyenne
        widths = [
            np.linalg.norm(np.array(outer[i]) - np.array(inner[i]))
            for i in range(min(len(outer), len(inner)))
        ]
        track_width = np.mean(widths)
        
        # Construire les murs
        walls = [(outer[i], outer[i+1]) for i in range(len(outer)-1)]
        walls += [(inner[i], inner[i+1]) for i in range(len(inner)-1)]
        
        # Checkpoints (tous les 10 points de la centerline)
        centerline = compute_centerline(outer, inner)
        checkpoints = centerline[::10]
    
    elif track_name == 'rectangle':
        from tracks.simple_rectangle import get_walls as gw, get_spawn as gs
        walls = gw()
        spawn = gs()
        
        from tracks.track_geometry import RectangularTrack
        track = RectangularTrack(walls)
        track_width = 80  # Largeur approximative
        
        # Checkpoints simples (4 coins)
        checkpoints = [(400, 80), (720, 300), (400, 520), (80, 300)]
    
    elif track_name == 'high_speed_ring':
        from tracks.high_speed_ring_gt import get_center_line as gcl, get_spawn as gs
        from tracks.Catmull_Rom_geometry import catmull_rom_spline
        from tracks.track_geometry import generate_walls, AngularTrack, compute_centerline
        
        control_points = gcl()
        centerline_smooth = catmull_rom_spline(control_points)
        outer, inner, track_width = generate_walls(centerline_smooth)
        
        track = AngularTrack(outer, inner)
        spawn = gs()
        
        walls = [(outer[i], outer[i+1]) for i in range(len(outer)-1)]
        walls += [(inner[i], inner[i+1]) for i in range(len(inner)-1)]
        
        # Checkpoints (tous les 20 points)
        centerline = compute_centerline(outer, inner)
        checkpoints = centerline[::20]
    
    else:
        raise ValueError(f"Circuit inconnu: {track_name}")
    
    # Créer l'environnement
    env = VectorizedCarEnv(
        spawn_point=spawn,
        walls=walls,
        track=track,
        track_width=track_width,
        n_cars=n_cars,
        n_rays=n_rays,
        device=device
    )
    
    return env, checkpoints