from .geometry import curvilinear_abscissa as ca
from tracks.track_geometry import compute_track_length
from envs.rewards.reward_centerline import distance_to_centerline

class ProgressReward:
    def __init__(self, centerline):
        self.centerline = centerline
        self.track_length = compute_track_length(centerline)
        self.prev_s = None

    def reset(self, car_state):
        self.prev_s = ca(
            car_state["position"],
            self.centerline
        )
    
    def step(self, car_state, collision):
        if collision:
            return -1000.0 # Gros malus mais pas infini

        s = ca(car_state["position"], self.centerline)
        delta_s = s - self.prev_s
        
        # Gestion du franchissement de ligne
        if delta_s < -self.track_length / 2: delta_s += self.track_length
        elif delta_s > self.track_length / 2: delta_s -= self.track_length

        # 1. Récompense de progression pure (incite à avancer)
        reward = delta_s * 5.0 
        
        # 2. Bonus de vitesse (incite à aller vite)
        if car_state["speed"] < 1: 
            reward -= 1
        else: 
            reward += 0 # Par encore de reward pour la vitesse        
        
        # 3. Pénalité de trajectoire (légère pour ne pas bloquer l'apprentissage)
        dist = distance_to_centerline(car_state["position"], self.centerline)
        reward -= 0.0001 * dist

        dist_in_front = car_state["ray distance"][ int( len(car_state["ray distance"]) / 2 ) ]
        if dist_in_front < 0.2:
            reward -= (0.2 - dist_in_front) * 10

        reward -= 0.1
        self.prev_s = s
        return reward
