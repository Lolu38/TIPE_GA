"""
rain_system_gpu.py — Système de pluie GPU
==========================================

La pluie est un état GLOBAL à toutes les voitures — ce n'est pas un tenseur (N,)
mais un scalaire Python/float qui évolue au fil des steps.

TireSystemGPU et FuelSystemGPU le reçoivent directement comme float
via rain_system.intensity — pas besoin de le broadcaster en (N,).

Trois modes :
    "fixed"   : intensité constante pendant tout l'épisode
    "dynamic" : marche aléatoire avec mean-reversion (météo réaliste)
    "preset"  : suit une séquence prédéfinie (utile pour les tests et le TIPE)

Intensité [0.0, 1.0] :
    0.0  -> piste sèche
    0.3  -> bruine légère
    0.6  -> forte pluie
    1.0  -> déluge

Dépend de :
    Rien — aucune dépendance pour rester simple et réutilisable.
"""

import torch
import math

# -----------------------------------------------------------------------------
# Constantes

# Paramètres par défaut du mode dynamique
DEFAULT_DRIFT_SIGMA      = 0.0015   # bruit gaussien par step
DEFAULT_MEAN_REVERT_COEF = 0.001    # force de rappel vers l'intensité initiale


class RainSystemGPU:
    """
    Simule l'évolution de la pluie au cours d'un épisode.

    Utilisé dans neuronal_env_v2.step() :
        rain = self.rain.intensity          # float
        grip = self.tires.grip_multiplier(rain)
        self.tires.update_wear(speed, rain)
        self.fuel.consume(speed)
        self.rain.step()                    # fait évoluer l'intensité

    Paramètres
    ----------
    mode: "fixed" | "dynamic" | "preset"
    initial_intensity: intensité de départ [0, 1]
    drift_sigma: écart-type du bruit par step (mode dynamic)
    mean_revert_coef: force du rappel vers initial_intensity (mode dynamic)
    preset_sequence: liste de (n_steps, target_intensity) (mode preset)
        Exemple : [(200, 0.0), (100, 0.7), (300, 0.4)] -> sec pendant 200 steps, puis interpolation vers 0.7 sur 100, etc.
    device: utilisé uniquement pour le générateur aléatoire GPU
    """

    def __init__(
        self,
        mode               : str   = "dynamic",
        initial_intensity  : float = 0.0,
        drift_sigma        : float = DEFAULT_DRIFT_SIGMA,
        mean_revert_coef   : float = DEFAULT_MEAN_REVERT_COEF,
        preset_sequence    : list  = None,
        device             : str   = 'cuda',
    ):
        self.mode              = mode
        self.initial_intensity = float(max(0.0, min(1.0, initial_intensity)))
        self.intensity         = self.initial_intensity
        self.drift_sigma       = drift_sigma
        self.mean_revert_coef  = mean_revert_coef
        self.device            = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Générateur aléatoire GPU (évite de polluer le générateur global)
        self._generator = torch.Generator(device=self.device)

        # Mode preset
        self._preset         = preset_sequence or []
        self._preset_step    = 0
        self._preset_segment = 0

    # Interface principale
    def step(self):
        """
        Fait évoluer l'intensité d'un step selon le mode choisi.
        À appeler une fois par step dans neuronal_env_v2, après la physique.
        """
        if self.mode == "fixed":
            return

        if self.mode == "dynamic":
            self._dynamic_step()
        elif self.mode == "preset":
            self._preset_step_fn()

    def reset(self, intensity: float = None):
        """
        Remet la pluie à l'état initial (ou à une valeur donnée).
        À appeler dans neuronal_env_v2.reset().
        """
        self.intensity       = self.initial_intensity if intensity is None else float(
            max(0.0, min(1.0, intensity))
        )
        self._preset_step    = 0
        self._preset_segment = 0

    # Modes internes
    def _dynamic_step(self):
        """
        Marche aléatoire gaussienne avec mean-reversion douce vers initial_intensity.
        Utilise le générateur GPU pour la cohérence avec le reste du pipeline.

        La pluie tend à revenir vers initial_intensity sur le long terme,
        ce qui donne une météo réaliste avec des épisodes pluvieux et des éclaircies.
        """
        # Bruit gaussien sur GPU
        noise_tensor = torch.randn(1, generator=self._generator, device=self.device)
        noise = noise_tensor.item() * self.drift_sigma

        # Mean-reversion : rappel vers l'intensité initiale
        mean_rev = self.mean_revert_coef * (self.initial_intensity - self.intensity)
        new_val = self.intensity + noise + mean_rev
        self.intensity = max(0.0, min(1.0, new_val))

    def _preset_step_fn(self):
        """
        Suit la séquence _preset par interpolation linéaire.

        Chaque entrée (n_steps, target) définit une transition :
        on interpole linéairement de l'intensité précédente vers target sur n_steps.
        """
        if not self._preset or self._preset_segment >= len(self._preset):
            return

        n_steps, target = self._preset[self._preset_segment]

        # Intensité de départ de ce segment
        prev_target = (
            self._preset[self._preset_segment - 1][1]
            if self._preset_segment > 0
            else self.initial_intensity
        )

        t = self._preset_step / max(n_steps - 1, 1)
        self.intensity = prev_target + t * (target - prev_target)
        self.intensity = max(0.0, min(1.0, self.intensity))

        self._preset_step += 1
        if self._preset_step >= n_steps:
            self._preset_step = 0
            self._preset_segment += 1

    # Utilitaires
    @property
    def is_dry(self) -> bool:
        return self.intensity < 0.15

    @property
    def is_wet(self) -> bool:
        return 0.15 <= self.intensity < 0.60

    @property
    def is_heavy(self) -> bool:
        return self.intensity >= 0.60

    @property
    def label(self) -> str:
        """Étiquette lisible de la condition météo actuelle (pour affichage)."""
        if self.intensity < 0.15:
            return "Sec"
        elif self.intensity < 0.35:
            return "Bruine"
        elif self.intensity < 0.60:
            return "Pluie"
        elif self.intensity < 0.80:
            return "Forte pluie"
        else:
            return "Déluge"

    def __repr__(self):
        return (
            f"RainSystemGPU(mode={self.mode}, "
            f"intensity={self.intensity:.3f}, "
            f"label='{self.label}')"
        )