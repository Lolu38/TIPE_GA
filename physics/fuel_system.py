"""
fuel_system_gpu.py - Système de carburant vectorisé GPU
========================================================

Gère le niveau de carburant de N voitures simultanément sur GPU.

Physique :
    - La consommation est proportionnelle à v² (plus on va vite, plus on brûle)
    - Réservoir vide -> la voiture peut encore rouler mais l'accélération est nulle
      (le moteur tourne à vide, on avance uniquement par inertie + frottement)
    - Pas d'effet sur la vitesse max ni sur le grip - seule l'accélération est
      concernée, et c'est neuronal_env_v2 qui applique cette logique en multipliant
      throttle_gas par fuel_throttle_factor() avant de l'envoyer à CarPhysics.

Dépend de :
    envs/physics.py  -> DT, BASE_MAX_SPEED
"""

import torch
from physics.physic import DT, BASE_MAX_SPEED

# Constantes
# -----------------------------------------------------------------------------
# Fraction du réservoir consommée par seconde à vitesse normalisée = 1.0
# -> à pleine vitesse, un plein dure environ 1 / 0.0018 = 555 secondes = 9 min
# -> à 300 steps/tour × 0.05 s = 15 s/tour -> ~10 tours max avant d'être à sec
BASE_CONSUMPTION_RATE = 0.0067   # par seconde

# La consommation varie en v² (aérodynamique + frottements)
# Exposant : 1.0 = linéaire, 2.0 = quadratique
CONSUMPTION_EXPONENT = 1.5

# Facteur de consommation minimum même à l'arrêt (moteur au ralenti)
IDLE_CONSUMPTION = 0.05   # 5% de la consommation max, moteur allumé


class FuelSystemGPU:
    """
    Gère le carburant de N voitures simultanément sur GPU.

    État interne :
        fuel  (N,)  float  - niveau [0.0, 1.0]  (1.0 = plein)

    Interface pit stop :
        refuel(car_mask) -> remet le réservoir à 1.0 pour les voitures sélectionnées

    Interface physics :
        fuel_throttle_factor() -> Tensor (N, 1) ∈ [0, 1]
            Multiplie le throttle_gas avant envoi à CarPhysics.
            Réservoir plein -> 1.0 (aucune restriction)
            Réservoir vide  -> 0.0 (plus d'accélération possible)
    """

    def __init__(self, n_cars: int, initial_fuel: float = 1.0, device: str = 'cuda'):
        """
        n_cars       : nombre de voitures dans la population
        initial_fuel : niveau initial [0.0, 1.0] (défaut = plein)
        device       : 'cuda' ou 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_cars = n_cars

        self.fuel = torch.full(
            (n_cars,),
            float(max(0.0, min(1.0, initial_fuel))),
            dtype=torch.float32,
            device=self.device
        )

    # Pit stop
    def refuel(self, car_mask: torch.Tensor):
        """
        Remet le réservoir à plein pour les voitures sélectionnées par car_mask.

        car_mask : Tensor (N,) bool
        """
        self.fuel[car_mask] = 1.0

    # Interface physics - appelé par neuronal_env_v2 AVANT d'appeler CarPhysics
    def fuel_throttle_factor(self) -> torch.Tensor:
        """
        Facteur multiplicatif sur le throttle_gas dans [0, 1].

        Réservoir plein -> 1.0  (aucune restriction)
        Réservoir vide  -> 0.0  (plus d'accélération, inertie seulement)

        Usage dans neuronal_env_v2.step() :
            effective_throttle_gas = throttle_gas * self.fuel.fuel_throttle_factor()
            -> passer effective_throttle_gas à CarPhysics au lieu de throttle_gas
        """
        # Simple proportionnalité - on pourrait ajouter une non-linéarité plus tard
        return self.fuel.unsqueeze(1)   # (N, 1)

    # Consommation - appelé par neuronal_env_v2 après chaque step de physique
    def consume(self, speed: torch.Tensor):
        """
        Consomme du carburant proportionnellement à v².

        speed : Tensor (N, 1) - vitesse courante en px/s

        Formule :
            speed_norm   = v / BASE_MAX_SPEED dans [0, 1]
            consumption  = BASE_RATE × max(speed_norm², IDLE) × DT
        """
        speed_norm = (speed.squeeze(1) / BASE_MAX_SPEED).clamp(0.0, 1.0)   # (N,)

        # Consommation quadratique + minimum au ralenti
        load = torch.clamp(speed_norm ** CONSUMPTION_EXPONENT, min=IDLE_CONSUMPTION)

        consumption = BASE_CONSUMPTION_RATE * load * DT   # (N,)

        self.fuel = (self.fuel - consumption).clamp(min=0.0)

    # Utilitaires
    def reset(self, initial_fuel: float = 1.0):
        """Remet tous les réservoirs à la valeur initiale."""
        self.fuel.fill_(float(max(0.0, min(1.0, initial_fuel))))

    def is_empty(self) -> torch.Tensor:
        """Retourne un masque (N,) bool - True si le réservoir est vide."""
        return self.fuel <= 0.0

    def mean_fuel(self) -> float:
        """Niveau moyen de carburant sur la population (pour debug/logging)."""
        return self.fuel.mean().item()

    def __repr__(self):
        return (
            f"FuelSystemGPU(n_cars={self.n_cars}, "
            f"fuel_mean={self.mean_fuel():.3f})"
        )