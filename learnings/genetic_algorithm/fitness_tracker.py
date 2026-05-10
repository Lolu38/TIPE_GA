import math
import torch


class FitnessTracker:
    """
    Suit la performance de chaque voiture pour calculer leur fitness.
    
    Métriques suivies:
    - Temps de survie (steps)
    - Vitesse moyenne
    - Checkpoints passés
    - Tours complets
    - Direction (bon sens ou mauvais sens)
    """
    
    def __init__(self, checkpoints, spawn_point, n_cars, track_width, walls, device='cuda'):
        """
        Args:
            checkpoints: Liste de points (x, y) définissant les checkpoints
            spawn_point: Tuple (x, y, angle) du point de départ
            n_cars: Nombre de voitures dans la population
            track_width: Largeur du circuit (rayon de détection des checkpoints)
            walls: Liste de murs pour calculer la ligne d'arrivée
            device: 'cuda' ou 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_cars = n_cars
        self.spawn_point = spawn_point
        self.treshold = track_width
        
        # Conversion checkpoints en tenseur GPU
        self.checkpoints = torch.tensor(checkpoints, dtype=torch.float32, device=self.device)
        self.n_checkpoints = len(checkpoints)
        
        # Métriques par voiture (tenseurs GPU pour calculs parallèles)
        self.survival_time = torch.zeros(n_cars, device=self.device)
        self.speed_sum = torch.zeros(n_cars, device=self.device)
        self.laps_completed = torch.zeros(n_cars, dtype=torch.int32, device=self.device)
        self.checkpoints_passed = torch.zeros(n_cars, dtype=torch.int32, device=self.device)
        
        # État des checkpoints pour chaque voiture (0 = non passé, 1 = passé)
        self.checkpoint_status = torch.zeros((n_cars, self.n_checkpoints), dtype=torch.bool, device=self.device)
        
        # Détection du sens de rotation
        self.first_checkpoint_direction = torch.zeros(n_cars, dtype=torch.int32, device=self.device)
        # 0 = pas encore déterminé, 1 = sens horaire, -1 = sens anti-horaire

        # Positions précédentes pour la détection de franchissement de ligne
        self.prev_positions = torch.zeros((n_cars, 2), device=self.device)

        # Calcul et stockage de la ligne d'arrivée (segment A → B perpendiculaire au spawn)
        self._compute_finish_line(spawn_point, walls)

    #  Calcul de la ligne d'arrivée
    def _compute_finish_line(self, spawn_point, walls):
        """
        Calcule les extrémités de la ligne d'arrivée en lançant deux rayons
        perpendiculaires à l'angle de spawn depuis le point de départ.
        """
        ox, oy, theta = spawn_point

        # Décalage vers l'arrière pour que les voitures spawnent devant la ligne
        offset_dist = self.treshold * 0.3
        ox_shifted = ox - math.cos(theta) * offset_dist
        oy_shifted = oy - math.sin(theta) * offset_dist

        perp_left = (math.cos(theta + math.pi / 2), math.sin(theta + math.pi / 2))
        perp_right = (math.cos(theta - math.pi / 2), math.sin(theta - math.pi / 2))

        A = self._cast_ray((ox_shifted, oy_shifted), perp_left,  walls)
        B = self._cast_ray((ox_shifted, oy_shifted), perp_right, walls)

        self.finish_line_A = torch.tensor(A, dtype=torch.float32, device=self.device)
        self.finish_line_B = torch.tensor(B, dtype=torch.float32, device=self.device)        

    def _cast_ray(self, origin, direction, walls):
        """
        Lance un rayon depuis origin dans direction et retourne le point
        d'impact avec le mur le plus proche.
        Compatible avec les formats (A, B) et (A, B, xmin, xmax, ymin, ymax).
        """
        from learnings.ray_casting.intersections import ray_segment_intersection
        ox, oy = origin
        dx, dy = direction
        best_t = float('inf')

        for wall in walls:
            A, B = wall[0], wall[1]
            t = ray_segment_intersection(origin, direction, A, B)
            if t is not None and t < best_t:
                best_t = t

        if best_t == float('inf'):
            best_t = self.treshold # Fallback si aucun mur trouvé

        return (ox + dx * best_t, oy + dy * best_t)

    #  Détection de franchissement de la ligne d'arrivée
    def _crosses_finish_line(self, positions):
        """
        Détecte les voitures qui ont franchi le segment finish_A → finish_B
        entre le step précédent et maintenant.

        Utilise le test d'intersection segment-segment vectorisé (2D).
        Faux positif au spawn impossible : au step 0, r = (0,0) → cross_rs ≈ 0
        → t très grand → crossed = False.
        """
        P = self.prev_positions # (N, 2)
        Q = positions # (N, 2)
        A = self.finish_line_A # (2,)
        B = self.finish_line_B # (2,)

        r   = Q - P # (N, 2) déplacement de la voiture
        s   = B - A # (2,)   vecteur de la ligne d'arrivée
        AmP = A.unsqueeze(0) - P # (N, 2) vecteur de P vers A

        # Produits vectoriels 2D
        cross_rs = r[:, 0] * s[1] - r[:, 1] * s[0] # (N,)
        cross_AmP_s = AmP[:, 0] * s[1] - AmP[:, 1] * s[0] # (N,)
        cross_AmP_r = AmP[:, 0] * r[:, 1]  - AmP[:, 1] * r[:, 0] # (N,)

        eps        = 1e-9
        safe_denom = torch.where(
            torch.abs(cross_rs) > eps,
            cross_rs,
            torch.full_like(cross_rs, eps)
        )

        t = cross_AmP_s / safe_denom # Position sur le déplacement voiture  dans [0,1] si ce step
        u = cross_AmP_r / safe_denom # Position sur la ligne d'arrivée dans [0,1] si dans le segment

        return (t >= 0) & (t <= 1) & (u >= 0) & (u <= 1)

    #  Reset
    def reset(self):
        """Réinitialise toutes les métriques pour une nouvelle génération."""
        self.survival_time.zero_()
        self.speed_sum.zero_()
        self.laps_completed.zero_()
        self.checkpoints_passed.zero_()
        self.checkpoint_status.zero_()
        self.first_checkpoint_direction.zero_()
        # Initialiser prev_positions au spawn (r = 0 au step 0 → pas de faux positif)
        self.prev_positions[:, 0] = self.spawn_point[0]
        self.prev_positions[:, 1] = self.spawn_point[1]

    #  Update
    def update(self, positions, speeds, alive_mask):
        """
        Mise à jour des métriques à chaque step.

        Args:
            positions:  Tensor (n_cars, 2) - positions (x, y)
            speeds:     Tensor (n_cars, 1) - vitesses
            alive_mask: Tensor (n_cars,)   - booléen, True si vivant
        """
        self.survival_time += alive_mask.float()
        self.speed_sum += speeds.squeeze() * alive_mask.float()

        self._check_checkpoints(positions, alive_mask)
        self._check_lap_completion(positions, alive_mask)

    #  Checkpoints intermédiaires
    def _check_checkpoints(self, positions, alive_mask):
        """
        Vérifie si des voitures ont franchi de nouveaux checkpoints.
        Utilise la distance euclidienne avec un seuil = track_width.
        """
        for cp_idx in range(self.n_checkpoints):
            cp_pos = self.checkpoints[cp_idx]  # (2,)

            diff = positions - cp_pos.unsqueeze(0)
            distances = torch.norm(diff, dim=1)  # (n_cars,)

            in_radius = distances < self.treshold
            not_passed_yet = ~self.checkpoint_status[:, cp_idx]
            newly_passed = in_radius & not_passed_yet & alive_mask

            self.checkpoint_status[:, cp_idx] |= newly_passed
            self.checkpoints_passed += newly_passed.int()

            first_timers = newly_passed & (self.first_checkpoint_direction == 0)
            if first_timers.any():
                self.first_checkpoint_direction[first_timers] = (
                    1 if cp_idx < self.n_checkpoints // 2 else -1
                )

    #  Détection de tour complet et raccourcis
    def _check_lap_completion(self, positions, alive_mask):
        """
        Détecte les tours complets et les raccourcis via le franchissement
        de la ligne d'arrivée (segment perpendiculaire au spawn).
        """
        all_checkpoints_passed = self.checkpoint_status.all(dim=1)
        crossed = self._crosses_finish_line(positions)

        # Raccourci : franchissement de la ligne sans avoir tout coché → tuer
        shortcut = ~all_checkpoints_passed & crossed & alive_mask
        alive_mask[shortcut] = False

        # Tour complet : franchissement de la ligne avec tous les checkpoints cochés
        lap_completed = all_checkpoints_passed & crossed & alive_mask
        self.laps_completed  += lap_completed.int()
        self.checkpoint_status[lap_completed]  = False
        self.checkpoints_passed[lap_completed] = 0

        # Mettre à jour les positions précédentes pour le prochain step
        self.prev_positions = positions.clone()

    #  Fitness
    def compute_fitness(self):
        avg_speed = self.speed_sum / torch.clamp(self.survival_time, min=1.0)

        direction_bonus = torch.ones(self.n_cars, device=self.device)
        direction_bonus[self.first_checkpoint_direction == 1] = 1.5
        direction_bonus[self.first_checkpoint_direction == -1] = 0.5

        checkpoint_bonus = 1.0 + (self.checkpoints_passed.float() / self.n_checkpoints) * 0.5

        has_laps = self.laps_completed > 0

        # Lap agents : uniquement laps/steps — plus c'est rapide, mieux c'est
        # 1e8 garantit que tout agent avec tour domine tout agent sans tour
        lap_fitness = (
            self.laps_completed.float()
            / torch.clamp(self.survival_time, min=1.0)
            * 1e8
            * direction_bonus
        )

        # Non-lap agents : survie + vitesse + checkpoints (inchangé)
        no_lap_fitness = (
            self.survival_time
            * avg_speed
            * checkpoint_bonus
            * direction_bonus
        )

        return torch.where(has_laps, lap_fitness, no_lap_fitness)

    #  Utilitaires
    def get_rankings(self):
        """Retourne les indices des voitures triées par fitness (meilleur → pire)."""
        return torch.argsort(self.compute_fitness(), descending=True)

    def get_statistics(self):
        """Retourne des statistiques pour affichage/logging."""
        fitness = self.compute_fitness()
        return {
            'best_fitness':  fitness.max().item(),
            'avg_fitness':   fitness.mean().item(),
            'max_laps':      self.laps_completed.max().item(),
            'avg_laps':      self.laps_completed.float().mean().item(),
            'avg_survival':  self.survival_time.mean().item(),
            'best_survival': self.survival_time.max().item(),
        }

    def get_render_checkpoints(self):
        """Retourne les positions des checkpoints pour l'affichage PyGame."""
        return self.checkpoints.cpu().numpy()

    def get_render_finish_line(self):
        """Retourne les extrémités de la ligne d'arrivée pour l'affichage PyGame."""
        return (
            tuple(self.finish_line_A.cpu().numpy()),
            tuple(self.finish_line_B.cpu().numpy()),
        )