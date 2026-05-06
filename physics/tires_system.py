"""
tire_system_gpu.py — Système de pneus vectorisé GPU
=====================================================

Version tenseur de tire_system.py, opère sur (N,) voitures simultanément.

Dépend de :
    envs/physics.py  -> DT, BASE_MAX_SPEED

Propriétés par composé (identiques à tire_system.py) :
    dry_grip: adhérence sur piste sèche [0, 1]
    wet_grip: adhérence sous pluie intense (rain = 1.0) [0, 1]
    base_wear_rate: usure par seconde à vitesse normalisée = 1.0 sur sec
    wet_wear_mult: multiplicateur d'usure sous pluie
    optimal_rain: plage [min, max] de pluie idéale pour ce composé

Rappel de la physique (voir tire_system.py pour la dérivation complète) :
    grip_fresh  = lerp(dry_grip, wet_grip, rain)
    wear_factor = lerp(1.0, WEAR_GRIP_FLOOR, wear)
    grip_final  = grip_fresh × wear_factor

    Wet/HeavyWet sur sec (rain < 0.10) -> surchauffe -> usure × WET_DRY_MULT
"""

import torch
from physics.physic import DT, BASE_MAX_SPEED

# -----------------------------------------------------------------------------
# Tables des propriétés — une ligne par composé, dans l'ordre COMPOUNDS
# -----------------------------------------------------------------------------

COMPOUNDS = ["Hard", "Medium", "Soft", "Wet", "HeavyWet"]
N_COMPOUNDS = len(COMPOUNDS)

# Index de chaque composé (pour construire les tenseurs de lookup)
HARD = 0
MEDIUM = 1
SOFT = 2
WET = 3
HEAVY_WET = 4

# Propriétés — shape (N_COMPOUNDS,) chacun
# On les stocke comme listes Python ici ; ils seront convertis en tenseurs GPU
# dans __init__ pour permettre le lookup vectorisé.

_DRY_GRIP = [0.82, 0.91, 1.00, 0.55, 0.40]
_WET_GRIP = [0.38, 0.48, 0.20, 0.92, 1.00]
_BASE_WEAR_RATE = [0.00030, 0.00065, 0.00140, 0.00090, 0.00060]  # par seconde
_WET_WEAR_MULT = [0.35, 0.50, 0.65, 0.90, 0.8]
_OPT_RAIN_LOW = [0.00, 0.00, 0.00, 0.25, 0.6]
_OPT_RAIN_HIGH = [0.15, 0.30, 0.08, 0.75, 1.0]

# Grip résiduel minimum quand wear = 1.0
WEAR_GRIP_FLOOR = 0.45

# Multiplicateur d'usure pour pneus pluie utilisés sur piste sèche (surchauffe)
WET_DRY_MULT = 4.0


class TireSystemGPU:
    """
    Gère les pneus de N voitures simultanément sur GPU.

    État interne (tous tenseurs sur self.device) :
        compound  (N,)    int32  — indice dans COMPOUNDS [0, 4]
        wear      (N,)    float  — usure [0.0, 1.0]

    Interface pit stop :
        queue_compound(car_mask, compound_idx) -> file un composé pour un sous-ensemble
        apply_pit_stop(car_mask)               -> applique + remet wear à 0
    """

    def __init__(self, n_cars: int, initial_compound: int = MEDIUM, device: str = 'cuda'):
        """
        n_cars: nombre de voitures dans la population
        initial_compound: indice du composé de départ (défaut = MEDIUM)
        device: 'cuda' ou 'cpu'
        """
        self.device  = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.n_cars  = n_cars

        # -- État des pneus ----------------------------------------------------
        self.compound = torch.full(
            (n_cars,), initial_compound,
            dtype=torch.int32, device=self.device
        )
        self.wear = torch.zeros(n_cars, device=self.device)

        # Composé en attente pour le prochain pit stop (−1 = aucun)
        self._queued = torch.full(
            (n_cars,), -1,
            dtype=torch.int32, device=self.device
        )

        # -- Tables de lookup GPU — shape (N_COMPOUNDS,) -----------------------
        # On les crée une seule fois ici pour éviter de les recréer à chaque step
        def _t(lst):
            return torch.tensor(lst, dtype=torch.float32, device=self.device)

        self._dry_grip       = _t(_DRY_GRIP)
        self._wet_grip       = _t(_WET_GRIP)
        self._base_wear_rate = _t(_BASE_WEAR_RATE)
        self._wet_wear_mult  = _t(_WET_WEAR_MULT)
        self._opt_rain_low   = _t(_OPT_RAIN_LOW) # borne basse de la plage optimale
        self._opt_rain_high  = _t(_OPT_RAIN_HIGH) # borne haute de la plage optimale

        # Masque booléen (N_COMPOUNDS,) : True si le composé est un pneu pluie
        _is_wet = torch.zeros(N_COMPOUNDS, dtype=torch.bool, device=self.device)
        _is_wet[WET]       = True
        _is_wet[HEAVY_WET] = True
        self._is_wet_compound = _is_wet

    # Pit stop
    def queue_compound(self, car_mask: torch.Tensor, compound_idx: int):
        """
        File le composé `compound_idx` pour les voitures sélectionnées par `car_mask`.

        car_mask    : Tensor (N,) bool — voitures concernées
        compound_idx: int ∈ [0, N_COMPOUNDS-1]
        """
        self._queued[car_mask] = compound_idx

    def apply_pit_stop(self, car_mask: torch.Tensor):
        """
        Applique le changement de composé en attente et remet l'usure à 0
        pour les voitures sélectionnées par `car_mask`.

        Si une voiture n'a pas de composé en file (_queued == -1),
        elle garde son composé actuel mais son usure est quand même remise à 0
        (pit stop = montage de pneus neufs du même composé).
        """
        # Voitures avec un composé en file ET dans le masque
        has_queued = (self._queued >= 0) & car_mask
        self.compound[has_queued] = self._queued[has_queued]
        self._queued[car_mask] = -1

        # Usure à 0 pour toutes les voitures au stand
        self.wear[car_mask] = 0.0

    # Grip — appelé par neuronal_env_v2 pour passer le grip à CarPhysics
    def grip_multiplier(self, rain_intensity: float) -> torch.Tensor:
        """
        Retourne l'adhérence disponible pour chaque voiture.

        rain_intensity : float scalaire [0, 1] — fourni par RainSystemGPU

        Retour : Tensor (N, 1) float — grip dans [WEAR_GRIP_FLOOR, 1.0]

        Formule :
            grip_fresh  = lerp(dry_grip[compound], wet_grip[compound], rain)
            wear_factor = 1.0 − (1.0 − WEAR_GRIP_FLOOR) × wear
            grip_final  = grip_fresh × wear_factor
        """
        # Lookup vectorisé : récupère la propriété du composé de chaque voiture
        # self.compound shape (N,) -> index dans les tables (N_COMPOUNDS,)
        dry  = self._dry_grip[self.compound] # (N,)
        wet  = self._wet_grip[self.compound] # (N,)

        # Interpolation linéaire sec ↔ mouillé selon la pluie
        grip_fresh = dry + (wet - dry) * rain_intensity   # (N,)

        # Dégradation par usure
        wear_factor = 1.0 - (1.0 - WEAR_GRIP_FLOOR) * self.wear # (N,)

        grip = grip_fresh * wear_factor # (N,)

        return grip.unsqueeze(1) # (N, 1) pour broadcaster avec les tenseurs physics

    # Usure — appelé par neuronal_env_v2 après chaque step de physique
    def update_wear(self, speed: torch.Tensor, rain_intensity: float):
        """
        Calcule et applique l'usure pour ce step.

        speed         : Tensor (N, 1) — vitesse courante en px/s
        rain_intensity: float scalaire [0, 1]

        Logique précise en 3 zones selon la plage optimale [opt_low, opt_high] :

          Zone 1 — Dans la plage optimale (opt_low ≤ rain ≤ opt_high) :
            wear_mult = 1.0  -> usure de base, conditions idéales

          Zone 2 — Trop mouillé (rain > opt_high) :
            frac_above = (rain - opt_high) / (1.0 - opt_high)   ∈ [0, 1]
            mult_above = lerp(1.0, _WET_WEAR_MULT, frac_above)
            Exemple Soft (opt_high=0.08, wet_mult=0.35, rain=0.50) :
              frac = (0.50-0.08)/(1.0-0.08) = 0.46
              mult = 1.0 + (0.35-1.0)×0.46 = 0.70  -> 30% moins d'usure (pas de chaleur)

          Zone 3 — Trop sec (rain < opt_low) :
            Uniquement pour Wet/HeavyWet qui surchauffent sur piste sèche.
            frac_below = (opt_low - rain) / opt_low              ∈ [0, 1]
            dry_penalty = lerp(1.0, WET_DRY_MULT, frac_below)
            Exemple Wet (opt_low=0.25, rain=0.05) :
              frac = (0.25-0.05)/0.25 = 0.80
              penalty = 1.0 + (4.0-1.0)×0.80 = 3.40  -> surchauffe massive

          wear_mult final = mult_above × dry_penalty
          wear_step = base_wear_rate × max(speed_norm, 0.10) × wear_mult × DT
        """
        speed_norm = (speed.squeeze(1) / BASE_MAX_SPEED).clamp(min=0.10) # (N,)

        # Lookup des propriétés du composé courant de chaque voiture
        base_rate = self._base_wear_rate[self.compound] # (N,)
        wet_mult  = self._wet_wear_mult[self.compound] # (N,)
        opt_low   = self._opt_rain_low[self.compound] # (N,)
        opt_high  = self._opt_rain_high[self.compound]  # (N,)

        # -- Zone 2 : trop mouillé (rain > opt_high) --------------------------
        # Fraction de dépassement par rapport à la borne haute [0, 1]
        # max_wet_range évite la division par zéro si opt_high = 1.0
        max_wet_range = (1.0 - opt_high).clamp(min=1e-6) # (N,)
        frac_above = ((rain_intensity - opt_high) / max_wet_range).clamp(0.0, 1.0)

        # Interpolation de 1.0 (dans la plage) vers wet_mult (complètement hors plage)
        mult_above = 1.0 + (wet_mult - 1.0) * frac_above # (N,)

        # -- Zone 3 : trop sec (rain < opt_low) — surchauffe pneus pluie ------
        # Fraction de dépassement par rapport à la borne basse [0, 1]
        max_dry_range = opt_low.clamp(min=1e-6) # (N,)
        frac_below = ((opt_low - rain_intensity) / max_dry_range).clamp(0.0, 1.0)

        # Surchauffe uniquement pour les pneus pluie (Wet, HeavyWet)
        # Les pneus secs sont déjà dans leur zone naturelle quand il fait sec
        is_wet_compound = self._is_wet_compound[self.compound] # (N,) bool
        dry_penalty = torch.where(
            is_wet_compound,
            1.0 + (WET_DRY_MULT - 1.0) * frac_below,   # surchauffe progressive
            torch.ones(self.n_cars, device=self.device)  # pas de pénalité pour pneus secs
        ) # (N,)

        # -- Multiplicateur final : produit des deux effets ---------------------
        # Si on est dans la plage optimale :
        #   frac_above = 0 -> mult_above = 1.0
        #   frac_below = 0 -> dry_penalty = 1.0
        #   -> wear_mult = 1.0 × 1.0 = 1.0  
        wear_mult = mult_above * dry_penalty # (N,)

        # -- Usure du step -----------------------------------------------------
        wear_step = base_rate * speed_norm * wear_mult * DT # (N,)
        self.wear = (self.wear + wear_step).clamp(max=1.0)

    # Utilitaires
    def reset(self, initial_compound: int = MEDIUM):
        """Remet tous les pneus à neuf avec le composé initial."""
        self.compound.fill_(initial_compound)
        self.wear.zero_()
        self._queued.fill_(-1)

    def is_destroyed(self) -> torch.Tensor:
        """Retourne un masque (N,) bool — True si le pneu est hors d'usage (wear ≥ 1.0)."""
        return self.wear >= 1.0

    def get_compound_name(self, car_idx: int) -> str:
        """Retourne le nom du composé d'une voiture spécifique (pour debug/affichage)."""
        return COMPOUNDS[self.compound[car_idx].item()]

    def __repr__(self):
        wear_mean = self.wear.mean().item()
        compounds = [COMPOUNDS[c] for c in self.compound[:5].tolist()]
        return (
            f"TireSystemGPU(n_cars={self.n_cars}, "
            f"wear_mean={wear_mean:.3f}, "
            f"first_5={compounds})"
        )