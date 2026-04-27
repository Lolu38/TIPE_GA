import pygame
import math
import numpy as np

WIDTH, HEIGHT = 600, 600
SCALE = 0.6
OFFSET_X = 50
OFFSET_Y = 50

CAR_SIZE = 5


class VectorizedRenderer:
    """
    Renderer pour l'algo génétique : affiche N voitures simultanément.

    Paramètres
    ----------
    show_rays : bool
        Afficher ou non les rayons LIDAR
    show_dead : bool
        Afficher ou non les voitures crashées (en grisé)
    """

    def __init__(self, show_rays: bool = False, show_dead: bool = False):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Genetic Car Simulation")
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 16)

        self.show_rays = show_rays
        self.show_dead = show_dead

        # --- État interne pour render_step ---
        self._step = 0
        self._last_gen = -1        # Détection de changement de génération
        self._wall_surface = None  # Cache des murs (ne changent pas en cours de gen)

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def render_step(
        self,
        generation: int,
        render_data: dict,
        walls: list,
        ray_data: dict | None = None,
        stats: dict | None = None,
    ) -> bool:
        """
        À appeler à chaque step depuis TrainingLoop (quand render=True).
        Le filtrage par fréquence de génération est géré en amont par TrainingLoop.

        Paramètres
        ----------
        generation : int
            Numéro de la génération en cours
        render_data : dict
            get_render_data() → "pos" (N,2), "angle" (N,1), "alive" (N,)
        walls : list
            Liste de tuples ((x1,y1), (x2,y2))
        ray_data : dict | None
            Optionnel : "origins" (N,2), "directions" (N,R,2), "distances" (N,R)
        stats : dict | None
            Infos libres à afficher dans le HUD

        Retourne
        --------
        bool : False si l'utilisateur ferme la fenêtre
        """
        # --- Détection de début de nouvelle génération ---
        if generation != self._last_gen:
            self._step = 0
            self._last_gen = generation
            self._wall_surface = None  # Invalider le cache (circuit potentiellement différent)
        else:
            self._step += 1

        # --- Dessin ---
        self.screen.fill((20, 20, 20))

        # Murs en cache : tracés une seule fois par génération
        if self._wall_surface is None:
            self._wall_surface = self._build_wall_surface(walls)
        self.screen.blit(self._wall_surface, (0, 0))

        pos    = render_data["pos"]    # (N, 2)
        angles = render_data["angle"]  # (N, 1)
        alive  = render_data["alive"]  # (N,)  bool

        if self.show_rays and ray_data is not None:
            self._draw_rays(ray_data, alive)

        self._draw_cars(pos, angles, alive)
        self._draw_hud(generation, self._step, alive, stats)

        pygame.display.flip()
        self.clock.tick(60)

        return self._poll_events()

    # ------------------------------------------------------------------
    # Dessin interne
    # ------------------------------------------------------------------

    def _build_wall_surface(self, walls: list) -> pygame.Surface:
        """Pré-rend les murs dans une Surface dédiée (cache)."""
        surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 0))
        for wall in walls:
            (x1, y1), (x2, y2) = wall
            pygame.draw.line(
                surf,
                (200, 200, 200),
                self._to_screen(x1, y1),
                self._to_screen(x2, y2),
                2,
            )
        return surf

    def _draw_cars(self, pos: np.ndarray, angles: np.ndarray, alive: np.ndarray):
        angles_flat = angles.flatten()

        for i in range(len(pos)):
            is_alive = bool(alive[i])

            if not is_alive and not self.show_dead:
                continue

            x, y = pos[i]
            theta = float(angles_flat[i])
            color = (80, 220, 80) if is_alive else (120, 120, 120)
            px, py = self._to_screen(x, y)

            triangle = [
                (px + CAR_SIZE * math.cos(theta),       py + CAR_SIZE * math.sin(theta)),
                (px + CAR_SIZE * math.cos(theta + 2.4), py + CAR_SIZE * math.sin(theta + 2.4)),
                (px + CAR_SIZE * math.cos(theta - 2.4), py + CAR_SIZE * math.sin(theta - 2.4)),
            ]
            pygame.draw.polygon(self.screen, color, triangle)

    def _draw_rays(self, ray_data: dict, alive: np.ndarray):
        """
        ray_data doit contenir :
          - "origins"    : (N, 2)
          - "directions" : (N, R, 2)
          - "distances"  : (N, R)
        """
        origins    = ray_data["origins"]
        directions = ray_data["directions"]
        distances  = ray_data["distances"]

        for i, is_alive in enumerate(alive):
            if not is_alive:
                continue
            ox, oy = origins[i]
            for r in range(directions.shape[1]):
                dx, dy = directions[i, r]
                d = float(distances[i, r])
                pygame.draw.line(
                    self.screen,
                    (255, 220, 0),
                    self._to_screen(ox, oy),
                    self._to_screen(ox + dx * d, oy + dy * d),
                    1,
                )

    def _draw_hud(
        self,
        generation: int,
        step: int,
        alive: np.ndarray,
        stats: dict | None,
    ):
        n_alive = int(alive.sum())
        n_total = len(alive)
        pct = n_alive / max(n_total, 1)

        lines = [
            f"Génération : {generation}",
            f"Step       : {step}",
            f"Vivants    : {n_alive} / {n_total}",
        ]

        if stats:
            for k, v in stats.items():
                lines.append(f"{k:<12}: {v:.3f}" if isinstance(v, float) else f"{k:<12}: {v}")

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (10, 10 + i * 20))

        # Barre de survie
        bar_x, bar_y = 10, HEIGHT - 20
        bar_w = WIDTH - 20
        pygame.draw.rect(self.screen, (60, 60, 60),  (bar_x, bar_y, bar_w, 10))
        pygame.draw.rect(self.screen, (80, 220, 80), (bar_x, bar_y, int(bar_w * pct), 10))

    # ------------------------------------------------------------------
    # Utilitaires
    # ------------------------------------------------------------------

    def _to_screen(self, x: float, y: float) -> tuple[int, int]:
        return int(x * SCALE + OFFSET_X), int(y * SCALE + OFFSET_Y)

    def _poll_events(self) -> bool:
        """Retourne False si l'utilisateur ferme la fenêtre (Échap ou croix)."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self):
        pygame.quit()