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
    
    def __init__(self, checkpoints, spawn_point, n_cars, track_width, device='cuda'):
        """
        Args:
            checkpoints: Liste de points (x, y) définissant les checkpoints
            spawn_point: Tuple (x, y, angle) du point de départ
            n_cars: Nombre de voitures dans la population
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
        
        # Distance au spawn pour détecter le passage de ligne
        self.spawn_tensor = torch.tensor([spawn_point[0], spawn_point[1]], dtype=torch.float32, device=self.device)
        self.last_spawn_distance = torch.full((n_cars,), float('inf'), device=self.device)
        
    def reset(self):
        """Réinitialise toutes les métriques pour une nouvelle génération"""
        self.survival_time.zero_()
        self.speed_sum.zero_()
        self.laps_completed.zero_()
        self.checkpoints_passed.zero_()
        self.checkpoint_status.zero_()
        self.first_checkpoint_direction.zero_()
        self.last_spawn_distance.fill_(float('inf'))
    
    def update(self, positions, speeds, alive_mask):
        """
        Mise à jour des métriques à chaque step
        
        Args:
            positions: Tensor (n_cars, 2) - positions (x, y)
            speeds: Tensor (n_cars, 1) - vitesses
            alive_mask: Tensor (n_cars,) - booléen, True si vivant
        """
        # Incrémenter le temps de survie pour les voitures vivantes
        self.survival_time += alive_mask.float()
        
        # Accumuler la vitesse pour calcul de la moyenne
        self.speed_sum += speeds.squeeze() * alive_mask.float()
        
        # Vérifier les passages de checkpoints
        self._check_checkpoints(positions, alive_mask)
        
        # Vérifier les tours complets
        self._check_lap_completion(positions, alive_mask)
    
    def _check_checkpoints(self, positions, alive_mask):
        """
        Vérifie si des voitures ont franchi des nouveaux checkpoints
        
        Utilise la distance euclidienne avec un seuil de détection
        """        
        # Pour chaque checkpoint
        for cp_idx in range(self.n_checkpoints):
            cp_pos = self.checkpoints[cp_idx]  # (2,)
            
            # Distance de chaque voiture à ce checkpoint (vectorisé)
            # (n_cars, 2) - (2,) -> (n_cars, 2)
            diff = positions - cp_pos.unsqueeze(0)
            distances = torch.norm(diff, dim=1)  # (n_cars,)
            
            # Voitures qui sont dans le rayon ET n'ont pas encore passé ce checkpoint
            in_radius = distances < self.treshold
            not_passed_yet = ~self.checkpoint_status[:, cp_idx]
            newly_passed = in_radius & not_passed_yet & alive_mask
            
            # Marquer comme passé
            self.checkpoint_status[:, cp_idx] |= newly_passed
            
            # Incrémenter le compteur de checkpoints
            self.checkpoints_passed += newly_passed.int()
            
            # Déterminer le sens de rotation (première fois qu'on passe un checkpoint)
            first_timers = newly_passed & (self.first_checkpoint_direction == 0)
            if first_timers.any():
                # Si c'est le checkpoint N, déterminer le sens
                # Sens horaire = checkpoints dans l'ordre croissant
                # Sens anti-horaire = checkpoints dans l'ordre décroissant
                self.first_checkpoint_direction[first_timers] = 1 if cp_idx < self.n_checkpoints // 2 else -1
    
    def _check_lap_completion(self, positions, alive_mask):
        """
        Détecte quand une voiture termine un tour complet
        
        Critère: tous les checkpoints passés + retour au spawn
        """        
        # Distance au spawn
        diff = positions - self.spawn_tensor.unsqueeze(0)
        spawn_distances = torch.norm(diff, dim=1)
        
        # Voitures qui ont passé tous les checkpoints
        all_checkpoints_passed = self.checkpoint_status.all(dim=1)
        
        # Voitures qui sont revenues au spawn
        near_spawn = spawn_distances < self.treshold
        
        # Voitures qui étaient loin du spawn au step précédent (évite multi-comptage)
        was_far = self.last_spawn_distance > self.treshold
        
        # Tour complet = tous checkpoints + retour au spawn + on était loin avant
        lap_completed = all_checkpoints_passed & near_spawn & was_far & alive_mask
        
        # Incrémenter les tours
        self.laps_completed += lap_completed.int()
        
        # Réinitialiser les checkpoints pour ceux qui ont fini un tour
        self.checkpoint_status[lap_completed] = False
        self.checkpoints_passed[lap_completed] = 0
        
        # Mettre à jour la distance au spawn
        self.last_spawn_distance = spawn_distances
    
    def compute_fitness(self):
        """
        Calcule le score final de chaque voiture
        
        Formule: fitness = (temps_survie) × (vitesse_moyenne) × (1 + tours) × bonus_sens
        
        Returns:
            Tensor (n_cars,) - scores de fitness
        """
        # Vitesse moyenne (éviter division par zéro)
        avg_speed = self.speed_sum / torch.clamp(self.survival_time, min=1.0)
        
        # Bonus de tours (exponentiel pour récompenser fortement les tours complets)
        lap_bonus = 1.0 + self.laps_completed.float()
        
        # Bonus de sens (récompenser le bon sens)
        # Si first_checkpoint_direction == 1 (sens horaire attendu), bonus x1.5
        # Si == -1 (anti-horaire), malus x0.5
        # Si == 0 (pas encore déterminé), neutre x1.0
        direction_bonus = torch.ones(self.n_cars, device=self.device)
        direction_bonus[self.first_checkpoint_direction == 1] = 1.5
        direction_bonus[self.first_checkpoint_direction == -1] = 0.5
        
        # Bonus pour les checkpoints passés (même sans finir le tour)
        checkpoint_bonus = 1.0 + (self.checkpoints_passed.float() / self.n_checkpoints) * 0.5
        
        # Formule finale
        fitness = (
            self.survival_time 
            * avg_speed 
            * lap_bonus 
            * direction_bonus 
            * checkpoint_bonus
        )
        
        return fitness
    
    def get_rankings(self):
        """
        Retourne les indices des voitures triées par fitness (meilleur → pire)
        
        Returns:
            Tensor (n_cars,) - indices triés
        """
        fitness = self.compute_fitness()
        sorted_indices = torch.argsort(fitness, descending=True)
        return sorted_indices
    
    def get_statistics(self):
        """
        Retourne des statistiques pour affichage/logging
        
        Returns:
            dict avec clés: best_fitness, avg_fitness, max_laps, avg_survival
        """
        fitness = self.compute_fitness()
        
        return {
            'best_fitness': fitness.max().item(),
            'avg_fitness': fitness.mean().item(),
            'max_laps': self.laps_completed.max().item(),
            'avg_laps': self.laps_completed.float().mean().item(),
            'avg_survival': self.survival_time.mean().item(),
            'best_survival': self.survival_time.max().item(),
        }
    
    def get_render_checkpoints(self):
        """
        Retourne les données pour afficher les checkpoints dans PyGame
        
        Returns:
            numpy array (n_checkpoints, 2) - positions des checkpoints
        """
        return self.checkpoints.cpu().numpy()