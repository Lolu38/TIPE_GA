"""
physics.py — Moteur physique vectorisé GPU
==========================================

Seul fichier du projet qui contient les constantes physiques.
Toutes les autres classes (neuronal_env_v2, systems) les importent d'ici.

Modèle physique
---------------
On sépare deux vecteurs qui peuvent diverger :

    heading       -> où la voiture POINTE  (direction du nez)
    velocity_angle -> où la voiture VA RÉELLEMENT

    slip_angle = velocity_angle - heading

Quand les pneus adhèrent  : slip_angle ≈ 0, la voiture suit exactement heading.
Quand les pneus lâchent   : slip_angle croît, la voiture part vers l'extérieur du virage.

Calibration — circuit Nascar
-----------------------------
    Rayon ligne de course ≈ 160 px  (entre inner=140 et outer=200)
    Dérive souhaitée à 77 % de la vitesse max avec pneu Medium neuf sur sec (grip=0.91)

    BASE_MAX_SPEED = périmètre / temps_tour = 1350 px / 15 s = 90 px/s
    déplacement max / step = 90 × 0.05 = 4.5 px

    F_lat_demanded au seuil = (0.77 × 90)² / 160 = 29.9 px/s²
    F_LAT_MAX_BASE = F_lat_demanded / grip_Medium = 29.9 / 0.91 ≈ 33.0 px/s²

Usage
-----
    from envs.physics import CarPhysics, BASE_MAX_SPEED, DT

    physics = CarPhysics(device='cuda')
    new_pos, new_speed, new_heading, new_slip = physics.step(
        pos, speed, heading, slip_angle, alive, steering, throttle, grip
    )
"""

import torch
import math

# -----------------------------------------------------------------------------
# Constantes physiques — TOUTES ICI, nulle part ailleurs dans le projet
# -----------------------------------------------------------------------------

DT = 0.05
# Pas de temps en secondes/step.
# Partagé avec tous les systems (tire_system_gpu, fuel_system_gpu...).

BASE_MAX_SPEED = 90.0
# Vitesse max théorique en px/s avec grip = 1.0 (Soft neuf sur sec).
# -> déplacement max par step = 90 × 0.05 = 4.5 px
# Calibré sur un tour Nascar en ~15 secondes (300 steps).

ACCELERATION = 55.0
# Accélération moteur MAX en px/s² (atteinte au pic de la courbe gaussienne).
# Réduit de 80 -> 55 pour un temps de montée en vitesse plus réaliste.
# Temps théorique pour atteindre vmax a plein gaz : ~4-5 secondes (vs ~1s avant).

BRAKE_FORCE = 130.0
# Décélération en px/s² quand throttle = -1.0.
# Volontairement plus fort que l'accélération — freiner est plus efficace qu'accélérer.

# Courbe de couple moteur (gaussienne)
# Simule la courbe de puissance d'un vrai moteur :
#   Démarrage lent (moteur pas encore en régime)
#   Pic d'accélération vers 35% de vmax (régime optimal)
#   Déclin progressif a haute vitesse (moteur a plein régime, plus de couple)
#
# rpm_factor = exp(-((speed_ratio - MU_RPM)² / (2 x SIGMA_RPM²)))
#
# Valeurs concrètes (MU=0.35, SIGMA=0.30) :
#   speed =   0% de vmax -> factor ≈ 0.51  (démarrage mou)
#   speed =  35% de vmax -> factor = 1.00  (pic moteur)
#   speed =  70% de vmax -> factor ≈ 0.51  (régime élevé)
#   speed = 100% de vmax -> factor ≈ 0.10  (quasi-impossible d'accélérer encore)

MU_RPM = 0.35   # Position du pic en fraction de vmax
SIGMA_RPM = 0.30   # Largeur de la cloche (plus grand = courbe plus plate)

FRICTION = 0.992
# Frottement passif appliqué chaque step (résistance de l'air + roulement).
# 0.992 -> perd 0.8% de vitesse par step à vitesse constante sans gaz.

STEERING_SENSITIVITY = 2.5
# Vitesse angulaire max du volant en rad/s à vitesse max.
# Exemple : 2.5 rad/s × 0.05 s = 0.125 rad/step à pleine vitesse.

F_LAT_MAX_BASE = 33.0
# Force latérale max supportable en px/s² avec grip = 1.0.
# Dérivé de la géométrie Nascar : (0.77×90)²/160 / 0.91 ≈ 33.0
# Multiplié par grip_multiplier dans le step pour donner la limite réelle.

K_DRIFT = 0.30
# Sensibilité au décrochage : rad par (px/s²) d'excès de force latérale par seconde.
# Augmenter -> la voiture part en dérive plus facilement.
# Diminuer -> la voiture est plus tolérante avant de glisser.

K_GRIP = 0.80
# Force de rappel du slip_angle vers 0, en 1/s, proportionnelle au grip.
# Représente la capacité du pneu à récupérer la dérive.
# K_GRIP/K_DRIFT = 0.80/0.30 ≈ 2.7 -> la dérive est récupérable si on lâche le gaz.

K_DRAG = 0.015
# Saignée de vitesse par radian de slip_angle, appliquée chaque step.
# Représente la chaleur et l'énergie dissipée dans les pneus lors de la dérive.

SLIP_ANGLE_MAX = math.pi / 2.5
# Slip angle max physiquement possible avant de clipper (~72°).
# Au-delà, la voiture est de côté — elle est de toute façon morte.


# Moteur physique
class CarPhysics:
    """
    Moteur physique vectorisé, 100% GPU, stateless.

    Fonctionne pour N=1 (agent unique) comme pour N=1000 (population génétique)
    sans aucune modification — c'est l'environnement qui stocke l'état et le passe
    à chaque appel.

    Tenseurs d'entrée (tous sur le même device) :
        pos         (N, 2)   — position (x, y)
        speed       (N, 1)   — vitesse scalaire en px/s
        heading     (N, 1)   — direction du nez en radians
        slip_angle  (N, 1)   — écart entre heading et velocity_angle en radians
        alive       (N,)     — masque booléen : False = voiture morte, ne bouge plus
        steering    (N, 1)   — commande volant dans [-1, 1]
        throttle    (N, 1)   — commande gaz/frein dans [-1, 1]
        grip        (N, 1)   — adhérence fournie par TireSystem dans [0, 1]

    Tenseurs de sortie :
        new_pos         (N, 2)
        new_speed       (N, 1)
        new_heading     (N, 1)
        new_slip_angle  (N, 1)
    """

    def __init__(self, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    def step(
        self,
        pos: torch.Tensor, # (N, 2)
        speed: torch.Tensor, # (N, 1)
        heading: torch.Tensor, # (N, 1)
        slip_angle: torch.Tensor, # (N, 1)
        alive: torch.Tensor, # (N,)  bool
        steering: torch.Tensor, # (N, 1)  dans [-1, 1]
        throttle: torch.Tensor, # (N, 1)  dans [-1, 1]
        grip: torch.Tensor, # (N, 1)  dans [0, 1]
    ):
        """
        Calcule le nouvel état physique pour un step.
        Retourne de nouveaux tenseurs — pas de modification in-place des inputs.
        """

        # -- 1. Vitesse max effective selon le grip du pneu --------------------
        # Un pneu Soft sec a grip=1.0 -> max = 90 px/s
        # Un Soft dans la pluie a grip=0.20 -> max = 18 px/s
        effective_max = BASE_MAX_SPEED * grip # (N, 1)

        # -- 2. Mise à jour du heading (où la voiture POINTE) -----------------
        # La vitesse angulaire dépend de la vitesse : à vitesse nulle,
        # tourner le volant ne fait rien (comme une vraie voiture).
        speed_norm   = speed / BASE_MAX_SPEED # (N, 1)
        omega        = steering * STEERING_SENSITIVITY * speed_norm # rad/s  (N, 1)
        new_heading  = heading + omega * DT # (N, 1)

        # Normalisation dans [-π, π] pour éviter l'accumulation numérique
        new_heading = torch.atan2(
            torch.sin(new_heading),
            torch.cos(new_heading)
        )

        # -- 3. Force latérale demandée vs force max disponible ---------------
        # F_lat_demanded = v × |ω|  (accélération centripète nécessaire en px/s²)
        # Si on va vite et qu'on tourne fort, cette force est élevée.
        # Si elle dépasse ce que le pneu peut encaisser -> décrochage.
        f_lat_demanded = speed * torch.abs(omega) # (N, 1)
        f_lat_max      = F_LAT_MAX_BASE * grip # (N, 1)

        excess = torch.clamp(f_lat_demanded - f_lat_max, min=0.0) # (N, 1)
        # excess > 0 -> le pneu est saturé, la dérive va croître

        # -- 4. Évolution du slip_angle ----------------------------------------
        # Croissance : l'excès de force fait glisser la voiture
        #   Le signe du steering indique vers quel côté elle part
        slip_growth = K_DRIFT * excess * torch.sign(steering) * DT # (N, 1)

        # Rappel : les pneus tirent toujours le slip vers 0
        #   Proportionnel au grip ET à l'amplitude du slip actuel
        #   Si grip = 0 (aquaplaning total), aucune récupération possible
        slip_recall = K_GRIP * grip * slip_angle * DT # (N, 1)

        new_slip = slip_angle + slip_growth - slip_recall

        # Clamp physique : au-delà de SLIP_ANGLE_MAX la voiture est de côté
        new_slip = torch.clamp(new_slip, -SLIP_ANGLE_MAX, SLIP_ANGLE_MAX)

        # -- 5. Mise à jour de la vitesse --------------------------------------
        # Gaz (throttle > 0) et frein (throttle < 0) séparés
        throttle_gas   = torch.clamp(throttle,  0.0, 1.0) # (N, 1)
        throttle_brake = torch.clamp(throttle, -1.0, 0.0) # (N, 1) négatif

        # Courbe gaussienne de couple moteur
        # rpm_factor varie de ~0.51 (arret) -> 1.0 (35% vmax) -> ~0.10 (vmax)
        # Cela simule : démarrage mou, pic de puissance, saturation en bout de ligne
        speed_ratio = (speed / effective_max.clamp(min=1e-6)).clamp(0.0, 1.0) # (N, 1)
        rpm_factor  = torch.exp(
            -((speed_ratio - MU_RPM) ** 2) / (2.0 * SIGMA_RPM ** 2)
        ) # (N, 1)

        effective_accel = ACCELERATION * grip * rpm_factor # (N, 1)
        effective_brake = BRAKE_FORCE  * grip # (N, 1)

        new_speed  = speed + throttle_gas   * effective_accel * DT
        new_speed  = new_speed + throttle_brake * effective_brake * DT   # soustrait car négatif

        # Frottement passif (résistance de l'air + roulement)
        new_speed *= FRICTION

        # Drag de dérive : l'énergie dissipée dans les pneus qui glissent saigne la vitesse
        # Plus le slip est grand, plus la voiture ralentit — incite à éviter la dérive
        new_speed *= (1.0 - K_DRAG * torch.abs(new_slip))

        # Clamp final [0, effective_max]
        new_speed = new_speed.clamp(min=0.0) # borne basse scalaire
        new_speed = torch.min(new_speed, effective_max)# borne haute tenseur (N, 1)

        # -- 6. Direction réelle du déplacement -------------------------------
        # velocity_angle = où la voiture VA, pas où elle POINTE
        velocity_angle = new_heading + new_slip # (N, 1)

        # -- 7. Mise à jour de la position -------------------------------------
        dx = new_speed * torch.cos(velocity_angle) * DT # (N, 1)
        dy = new_speed * torch.sin(velocity_angle) * DT # (N, 1)

        movement = torch.cat([dx, dy], dim=1) # (N, 2)

        # Les voitures mortes ne bougent plus
        movement = movement * alive.unsqueeze(1).float()

        new_pos = pos + movement # (N, 2)

        return new_pos, new_speed, new_heading, new_slip

    # Utilitaires
    @staticmethod
    def velocity_angle(heading: torch.Tensor, slip_angle: torch.Tensor) -> torch.Tensor:
        """
        Angle réel du déplacement.
        Utile pour le rendu (dessiner la trajectoire réelle vs le nez).
        """
        return heading + slip_angle

    @staticmethod
    def is_drifting(slip_angle: torch.Tensor, threshold: float = 0.05) -> torch.Tensor:
        """
        Masque booléen (N,) — True si la voiture est en dérive significative.

        threshold ≈ 0.05 rad ≈ 3° -> dérive perceptible mais pas catastrophique
        threshold ≈ 0.20 rad ≈ 11° -> dérive sévère

        Utilisé par :
          - le renderer (colorer la voiture en orange/rouge)
          - la reward function (pénaliser la dérive excessive)
        """
        return torch.abs(slip_angle).squeeze(-1) > threshold

    @staticmethod
    def drift_intensity(slip_angle: torch.Tensor) -> torch.Tensor:
        """
        Intensité normalisée de la dérive dans [0, 1].
        0 = aucune dérive, 1 = dérive maximale (SLIP_ANGLE_MAX).
        Utile pour une pénalité continue dans la reward function.
        """
        return torch.abs(slip_angle).squeeze(-1) / SLIP_ANGLE_MAX