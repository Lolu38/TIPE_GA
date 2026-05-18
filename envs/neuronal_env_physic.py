"""
neuronal_env_v2.py - Environnement vectorisé GPU avec physique réaliste
========================================================================

Changements par rapport à neuronal_env.py :

  Physique
    - heading  séparé de velocity_angle via slip_angle (dérive réaliste)
    - vitesse max effective = BASE_MAX_SPEED × grip
    - accélération et freinage modulés par grip et niveau fuel
    - tout délégué à CarPhysics (envs/physics.py)

  Systèmes
    - TireSystemGPU  -> grip, usure selon pluie et composé
    - FuelSystemGPU  -> consommation en v², accélération nulle à réservoir vide
    - RainSystemGPU  -> pluie globale, évolution dynamique ou fixe

  Pit stops
    - pit_tire (action 2) : arrêt n=PIT_TIRE_DURATION steps -> pneus neufs
    - pit_full (action 3) : arrêt N=PIT_FULL_DURATION steps -> pneus + refuel
    - déclenché uniquement si in_pit_zone (géré en amont par pop_manager_v2)
    - pendant le pit : voiture immobile, invulnérable aux collisions

  Observations : (N, n_rays + 5)
    distances lidar × n_rays | vitesse | usure | fuel | pluie | composé

  Collision GPU
    - point-in-polygon vectorisé, zéro transfert CPU (correctif vs v1)
    - AngularTrack  -> test dans polygone outer ET hors polygone inner
    - RectangularTrack -> AABB simple
    - fallback CPU si type inconnu (affiche un warning)

  Interface
    Identique à neuronal_env.py : reset / step / get_observations / get_render_data
    -> TrainingLoop et circuits_loader_v2 n'ont rien à changer côté appel.
    Nouveaux attributs publics : in_pit_zone (N,) bool, slip_angle (N, 1)
"""

import torch
import numpy as np

from physics.physic import CarPhysics, BASE_MAX_SPEED, DT
from physics.tires_system import TireSystemGPU, MEDIUM, N_COMPOUNDS
from physics.fuel_system import FuelSystemGPU
from physics.rain_system import RainSystemGPU

# -----------------------------------------------------------------------------
# Constantes propres à l'environnement
# -----------------------------------------------------------------------------

# Durées de pit stop en steps (dt=0.05 -> 1 step = 0.05 s)
PIT_TIRE_DURATION = 60    # ~3 secondes - changement pneus uniquement
PIT_FULL_DURATION = 120   # ~6 secondes - pneus + plein de carburant

# Rayon autour du spawn considéré comme zone de pit (pixels)
PIT_ZONE_RADIUS = 5.0

# Seuil sigmoid pour déclencher un pit stop (> seuil -> pit)
PIT_THRESHOLD = 0.5

# Seuil de proximité aux murs pour la collision par rayons (préfiltrage rapide)
RAY_COLLISION_THRESHOLD = 3.0


class VectorizedCarEnv:
    """
    Environnement vectorisé pour N voitures en parallèle sur GPU.

    Paramètres
    ----------
    spawn_point: (x, y, theta) - position et angle de départ
    walls: liste de ((x1,y1),(x2,y2)) - segments de murs
    track: RectangularTrack ou AngularTrack
    track_width: largeur moyenne du circuit (normalisation)
    n_cars: taille de la population
    n_rays: nombre de rayons lidar
    device: 'cuda' ou 'cpu'
    collision_threshold: distance min aux murs (préfiltre rayon)
    rain_mode: 'fixed' | 'dynamic' | 'preset'
    initial_rain: intensité initiale de la pluie [0, 1]
    initial_compound: indice du composé de départ (défaut MEDIUM=1)
    """

    def __init__(
        self,
        spawn_point,
        walls,
        track,
        track_width: float = 60.0,
        n_cars: int   = 1000,
        n_rays: int   = 9,
        device: str   = 'cuda',
        max_speed: float = None,   # ignoré - kept pour compatibilité interface
        collision_threshold: float = RAY_COLLISION_THRESHOLD,
        rain_mode: str = 'fixed',
        initial_rain: float = 0.0,
        initial_compound: int   = MEDIUM,
    ):
        self.device    = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_cars    = n_cars
        self.n_rays    = n_rays
        self.track_width        = track_width
        self.collision_threshold = collision_threshold

        # -- Spawn -------------------------------------------------------------
        if len(spawn_point) == 3:
            self.spawn_x, self.spawn_y, self.spawn_angle = spawn_point
        else:
            self.spawn_x, self.spawn_y = spawn_point
            self.spawn_angle = 0.0

        self.spawn_tensor = torch.tensor(
            [self.spawn_x, self.spawn_y],
            dtype=torch.float32, device=self.device
        )

        # -- Track -> tenseurs GPU pour collision -------------------------------
        self.track = track
        self._setup_track_tensors(track)

        # -- Murs -> tenseurs GPU pour raycasting -------------------------------
        starts, ends = [], []
        for w in walls:
            starts.append(w[0])
            ends.append(w[1])
        self.wall_starts = torch.tensor(starts, dtype=torch.float32, device=self.device)
        self.wall_ends   = torch.tensor(ends,   dtype=torch.float32, device=self.device)

        # -- Moteur physique ---------------------------------------------------
        self.physics = CarPhysics(device=str(self.device))

        # -- Systèmes ----------------------------------------------------------
        self.tires = TireSystemGPU(n_cars, initial_compound, str(self.device))
        self.fuel  = FuelSystemGPU(n_cars, device=str(self.device))
        self.rain  = RainSystemGPU(
            mode=rain_mode, initial_intensity=initial_rain, device=str(self.device)
        )

        # -- Angles des rayons (fixes, calculés une fois) -----------------------
        self.ray_angles = torch.linspace(
            -np.pi / 2, np.pi / 2, n_rays, device=self.device
        )   # (n_rays,)

        # -- Tenseurs d'état - initialisés dans reset() ------------------------
        self.pos        = torch.zeros((n_cars, 2), device=self.device)
        self.speed      = torch.zeros((n_cars, 1), device=self.device)
        self.heading    = torch.zeros((n_cars, 1), device=self.device)
        self.slip_angle = torch.zeros((n_cars, 1), device=self.device)
        self.alive      = torch.ones(n_cars, dtype=torch.bool, device=self.device)
        self.distances  = torch.zeros((n_cars, n_rays), device=self.device)

        # -- État pit stop ------------------------------------------------------
        self.pit_timer   = torch.zeros(n_cars, dtype=torch.int32, device=self.device)
        self.pit_is_full = torch.zeros(n_cars, dtype=torch.bool,  device=self.device)
        self.in_pit_zone = torch.zeros(n_cars, dtype=torch.bool,  device=self.device)

        self.reset()

    # Setup
    def _setup_track_tensors(self, track):
        """
        Convertit la géométrie du track en tenseurs GPU.
        Les sommets des polygones sont pré-shiftés (roll -1) une seule fois
        pour éviter de recalculer torch.roll à chaque step.
        """
        from tracks.track_geometry import AngularTrack, RectangularTrack

        if isinstance(track, AngularTrack):
            self.track_type = 'angular'

            outer = torch.tensor(track.outer, dtype=torch.float32, device=self.device)
            inner = torch.tensor(track.inner, dtype=torch.float32, device=self.device)

            # Pré-calcul des sommets suivants (edge i -> i+1)
            self.outer_poly      = outer
            self.outer_poly_next = torch.roll(outer, -1, dims=0)
            self.inner_poly      = inner
            self.inner_poly_next = torch.roll(inner, -1, dims=0)

        elif isinstance(track, RectangularTrack):
            self.track_type = 'rectangular'
            ox_min, ox_max, oy_min, oy_max = track.outer_bounds
            ix_min, ix_max, iy_min, iy_max = track.inner_bounds
            self.outer_bounds_t = torch.tensor(
                [ox_min, ox_max, oy_min, oy_max], dtype=torch.float32, device=self.device
            )
            self.inner_bounds_t = torch.tensor(
                [ix_min, ix_max, iy_min, iy_max], dtype=torch.float32, device=self.device
            )

        else:
            self.track_type = 'unknown'
            print(
                "[neuronal_env_v2] ⚠  Type de track inconnu - "
                "fallback CPU pour la détection de collision (lent)."
            )

    # Reset
    def reset(self, randomize_spawn=False, random_range=10.0, random_angle=0.1):
        """
        Remet toutes les voitures au départ.

        Returns : observations Tensor (N, n_rays + 5)
        """
        self.pos[:, 0] = self.spawn_x
        self.pos[:, 1] = self.spawn_y

        if randomize_spawn:
            self.pos     += torch.randn((self.n_cars, 2), device=self.device) * random_range
            self.heading[:] = self.spawn_angle + \
                torch.randn((self.n_cars, 1), device=self.device) * random_angle
        else:
            self.heading[:] = self.spawn_angle

        self.speed.zero_()
        self.slip_angle.zero_()
        self.alive[:] = True
        self.distances.zero_()

        self.pit_timer.zero_()
        self.pit_is_full.zero_()
        self.in_pit_zone.zero_()

        self.tires.reset()
        self.fuel.reset()
        self.rain.reset()

        return self.get_observations()

    # Step
    def step(self, actions: torch.Tensor):
        """
        Avance d'un pas de temps.

        Args
        ----
        actions : Tensor (N, 4)
            [:, 0] steering  dans [-1, 1]
            [:, 1] throttle  dans [-1, 1]  (négatif = frein)
            [:, 2] pit_tire  dans [0, 1]   (> PIT_THRESHOLD -> pit pneus)
            [:, 3] pit_full  dans [0, 1]   (> PIT_THRESHOLD -> pit complet)

        Returns
        -------
        observations : Tensor (N, n_rays + 5)
        rewards      : Tensor (N,)
        dones        : Tensor (N,)  bool
        """
        steering = actions[:, 0:1]   # (N, 1)
        throttle = actions[:, 1:2]   # (N, 1)
        sig_tire = actions[:, 2]     # (N,)
        sig_full = actions[:, 3]     # (N,)

        # -- 1. Gestion pit stop -----------------------------------------------

        in_pit = self.pit_timer > 0   # (N,) - voitures actuellement au stand

        # Décrémenter le timer des voitures au stand
        self.pit_timer = (self.pit_timer - 1).clamp(min=0)

        # Appliquer le pit stop quand le timer vient d'atteindre 0
        just_finished = in_pit & (self.pit_timer == 0)   # (N,)
        if just_finished.any():
            self.tires.apply_pit_stop(just_finished)
            full_done = just_finished & self.pit_is_full
            if full_done.any():
                self.fuel.refuel(full_done)
            self.pit_is_full[just_finished] = False

        # Déclencher un nouveau pit stop
        # Condition : vivante + pas déjà au stand + signal > seuil
        can_pit     = self.alive & ~in_pit             # (N,)
        new_tire    = can_pit & (sig_tire > PIT_THRESHOLD)
        new_full    = can_pit & (sig_full > PIT_THRESHOLD)
        new_tire    = new_tire & ~new_full             # pit_full prioritaire

        if new_tire.any():
            self.pit_timer[new_tire]   = PIT_TIRE_DURATION
            self.pit_is_full[new_tire] = False

        if new_full.any():
            self.pit_timer[new_full]   = PIT_FULL_DURATION
            self.pit_is_full[new_full] = True

        # -- 2. Physique -------------------------------------------------------

        # Voitures actives = vivantes ET pas au stand
        active = self.alive & ~in_pit   # (N,)

        rain = self.rain.intensity                             # float scalaire
        grip = self.tires.grip_multiplier(rain)               # (N, 1)
        effective_throttle = throttle * self.fuel.fuel_throttle_factor()  # (N, 1)

        self.pos, self.speed, self.heading, self.slip_angle = self.physics.step(
            self.pos, self.speed, self.heading, self.slip_angle,
            active, steering, effective_throttle, grip
        )

        # -- 3. Mise à jour des systèmes ---------------------------------------
        self.tires.update_wear(self.speed, rain)
        self.fuel.consume(self.speed)
        self.rain.step()

        # -- 4. Zone de pit ----------------------------------------------------
        dist_to_spawn    = torch.norm(
            self.pos - self.spawn_tensor.unsqueeze(0), dim=1
        )                                                      # (N,)
        self.in_pit_zone = dist_to_spawn < PIT_ZONE_RADIUS     # (N,) bool

        # -- 5. Raycasting -----------------------------------------------------
        self.distances = self._compute_lidar()

        # -- 6. Détection de collision -----------------------------------------
        ray_crash   = (self.distances < self.collision_threshold).any(dim=1)   # (N,)
        track_crash = self._check_track_collision_gpu()                        # (N,)

        # Les voitures au stand sont invulnérables
        crashes    = (ray_crash | track_crash) & ~in_pit
        self.alive = self.alive & ~crashes

        # -- 7. Observations et retour -----------------------------------------
        observations = self.get_observations()
        rewards      = self.speed.squeeze(1) * self.alive.float()
        dones        = ~self.alive

        return observations, rewards, dones

    # Raycasting
    def _compute_lidar(self) -> torch.Tensor:
        """
        Calcule les distances lidar pour toutes les voitures.

        heading  (N, 1) + ray_angles (n_rays,) -> broadcast (N, n_rays)
        Puis aplatissement -> (N×n_rays, 2) pour compute_ray_intersections.

        Returns : (N, n_rays) distances en pixels
        """
        from learnings.ray_casting.gpu_raycasting import compute_ray_intersections

        global_angles = self.heading + self.ray_angles # (N, n_rays) broadcast
        flat_angles = global_angles.view(-1) # (N×n_rays,)

        ray_dirs = torch.stack(
            [torch.cos(flat_angles), torch.sin(flat_angles)], dim=1
        ) # (N×n_rays, 2)
        ray_origins = self.pos.repeat_interleave(self.n_rays, dim=0)  # (N×n_rays, 2)

        dists = compute_ray_intersections(
            ray_origins, ray_dirs,
            self.wall_starts, self.wall_ends,
            max_dist=3.0 * self.track_width
        ) # (N×n_rays,)

        return dists.view(self.n_cars, self.n_rays) # (N, n_rays)

    # Collision GPU
    def _inside_polygon_gpu(
        self,
        points       : torch.Tensor, # (N, 2)
        poly         : torch.Tensor, # (M, 2)
        poly_next    : torch.Tensor, # (M, 2)  = roll(poly, -1)  pré-calculé
    ) -> torch.Tensor:
        """
        Test point-in-polygon par ray casting, 100% GPU.

        Pour chaque point, on lance un rayon horizontal vers +x et on compte
        le nombre d'arêtes du polygone qu'il traverse.
        Impair -> intérieur, pair -> extérieur.

        Complexité mémoire : O(N × M) - pour N=1000, M=200 -> ~3 MB. ✓

        Returns : (N,) bool - True si le point est à l'intérieur
        """
        # Coordonnées des points : (N, 1)
        px = points[:, 0].unsqueeze(1) # (N, 1)
        py = points[:, 1].unsqueeze(1) # (N, 1)

        # Coordonnées des arêtes : (1, M)
        x1 = poly[:, 0].unsqueeze(0) # (1, M)
        y1 = poly[:, 1].unsqueeze(0) # (1, M)
        x2 = poly_next[:, 0].unsqueeze(0) # (1, M)
        y2 = poly_next[:, 1].unsqueeze(0) # (1, M)

        # Condition 1 : py est strictement entre y1 et y2
        cond1 = (y1 > py) != (y2 > py) # (N, M) bool

        # Abscisse de l'intersection du rayon horizontal y=py avec l'arête
        dy = y2 - y1 # (1, M)
        safe_dy = torch.where(dy.abs() < 1e-9, torch.full_like(dy, 1e-9), dy)
        xinters = x1 + (py - y1) * (x2 - x1) / safe_dy # (N, M)

        # Condition 2 : l'intersection est à droite de px
        cond2 = px < xinters # (N, M) bool

        # Nombre de traversées - impair = intérieur
        crossings = (cond1 & cond2).int().sum(dim=1) # (N,)
        return (crossings % 2) == 1 # (N,) bool

    def _check_track_collision_gpu(self) -> torch.Tensor:
        """
        Retourne un masque (N,) bool : True = voiture hors circuit.
        Zéro transfert GPU->CPU.
        """
        if self.track_type == 'angular':
            in_outer = self._inside_polygon_gpu(
                self.pos, self.outer_poly, self.outer_poly_next
            )
            in_inner = self._inside_polygon_gpu(
                self.pos, self.inner_poly, self.inner_poly_next
            )
            # Sur le circuit = dans outer ET hors inner
            return ~(in_outer & ~in_inner)   # (N,) bool

        elif self.track_type == 'rectangular':
            x = self.pos[:, 0]
            y = self.pos[:, 1]
            ox_min, ox_max, oy_min, oy_max = self.outer_bounds_t
            ix_min, ix_max, iy_min, iy_max = self.inner_bounds_t
            in_outer = (x >= ox_min) & (x <= ox_max) & (y >= oy_min) & (y <= oy_max)
            in_inner = (x >= ix_min) & (x <= ix_max) & (y >= iy_min) & (y <= iy_max)
            return ~(in_outer & ~in_inner)

        else:
            # Fallback CPU - affiché au __init__, ne devrait pas arriver
            collisions = torch.zeros(self.n_cars, dtype=torch.bool, device=self.device)
            pos_cpu = self.pos.cpu().numpy()
            for i in range(self.n_cars):
                if self.alive[i] and not self.track.is_inside(*pos_cpu[i]):
                    collisions[i] = True
            return collisions

    # Observations
    def get_observations(self) -> torch.Tensor:
        """
        Construit le vecteur d'observation pour toutes les voitures.

        Returns : Tensor (N, n_rays + 5)
          [0 : n_rays]      distances lidar normalisées [0, 1]
          [n_rays]          vitesse normalisée          [0, 1]
          [n_rays + 1]      usure pneu                  [0, 1]
          [n_rays + 2]      niveau carburant             [0, 1]
          [n_rays + 3]      intensité pluie              [0, 1]
          [n_rays + 4]      composé normalisé            [0, 1]
        """
        max_range  = 3.0 * self.track_width

        rays     = (self.distances / max_range).clamp(0.0, 1.0)               # (N, n_rays)
        speed    = (self.speed / BASE_MAX_SPEED).clamp(0.0, 1.0)              # (N, 1)
        wear     = self.tires.wear.unsqueeze(1)                               # (N, 1)
        fuel     = self.fuel.fuel.unsqueeze(1)                                # (N, 1)
        rain     = torch.full(
            (self.n_cars, 1), self.rain.intensity, device=self.device
        )                                                                      # (N, 1)
        compound = (self.tires.compound.float() / N_COMPOUNDS).unsqueeze(1)   # (N, 1)

        return torch.cat([rays, speed, wear, fuel, rain, compound], dim=1)    # (N, n_rays+5)

    # Render
    def get_render_data(self) -> dict:
        """
        Renvoie les données CPU pour l'affichage PyGame.

        Clés ajoutées par rapport à v1 :
          slip_angle  : pour colorer les voitures en dérive
          in_pit_zone : pour afficher la zone de pit
          pit_timer   : pour afficher le compte à rebours au stand
        """
        return {
            'pos'        : self.pos.cpu().numpy(),
            'angle'      : self.heading.cpu().numpy(),
            'alive'      : self.alive.cpu().numpy(),
            'speed'      : self.speed.cpu().numpy(),
            'slip_angle' : self.slip_angle.cpu().numpy(),
            'in_pit_zone': self.in_pit_zone.cpu().numpy(),
            'pit_timer'  : self.pit_timer.cpu().numpy(),
        }

    # Utilitaires
    def get_alive_count(self) -> int:
        return self.alive.sum().item()

    def is_all_dead(self) -> bool:
        return not self.alive.any()