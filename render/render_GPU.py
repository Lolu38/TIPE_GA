import pygame
import math
import numpy as np

WIDTH, HEIGHT = 1000, 1000
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
        self.show_dead = False

        self._step = 0
        self._last_gen = -1
        self._wall_surface = None
        self.finish_line = None   # Stockage de la ligne d'arrivée

    # ------------------------------------------------------------------
    # Interface principale
    # ------------------------------------------------------------------

    def render_step(
        self,
        generation: int,
        render_data: dict,
        walls: list,
        finish_line: tuple | None = None,
        ray_data: dict | None = None,
        stats: dict | None = None,
    ) -> bool:
        """
        À appeler à chaque step depuis TrainingLoop (quand render=True).

        Paramètres
        ----------
        generation: int
        render_data: dict       -> "pos" (N,2), "angle" (N,1), "alive" (N,)
        walls: list       -> liste de tuples ((x1,y1), (x2,y2))
        finish_line: tuple|None -> (A, B) retourné par get_render_finish_line()
        ray_data: dict|None  -> "origins", "directions", "distances"
        stats: dict|None  -> infos libres pour le HUD
        """
        if generation != self._last_gen:
            self._step = 0
            self._last_gen = generation
            self._wall_surface = None
        else:
            self._step += 1

        # Mémoriser la ligne d'arrivée dès qu'on la reçoit
        if finish_line is not None:
            self.finish_line = finish_line

        self.screen.fill((20, 20, 20))

        if self._wall_surface is None:
            self._wall_surface = self._build_wall_surface(walls)
        self.screen.blit(self._wall_surface, (0, 0))

        # Ligne d'arrivée dessinée directement sur self.screen (évite le bug SRCALPHA)
        if self.finish_line is not None:
            self._draw_finish_line(self.screen, self.finish_line)

        pos    = render_data["pos"]
        angles = render_data["angle"]
        alive  = render_data["alive"]

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

    def _draw_finish_line(
        self,
        surf: pygame.Surface,
        finish_line: tuple,
        n_cols: int = 10,
        n_rows: int = 2,
        epsilon: float = 8.0,
    ):
        A = np.array(finish_line[0], dtype=float)
        B = np.array(finish_line[1], dtype=float)

        AB = B - A
        ab_len = np.linalg.norm(AB)
        if ab_len < 1e-6:
            return

        ab_dir    = AB / ab_len
        track_dir = np.array([ab_dir[1], -ab_dir[0]])

        for i in range(n_cols):
            for j in range(n_rows):
                t0 = i       / n_cols
                t1 = (i + 1) / n_cols

                s0 = -epsilon + j       * (2 * epsilon / n_rows)
                s1 = -epsilon + (j + 1) * (2 * epsilon / n_rows)

                color = (255, 255, 255) if (i + j) % 2 == 0 else (0, 0, 0)

                p0 = A + t0 * AB + s0 * track_dir
                p1 = A + t1 * AB + s0 * track_dir
                p2 = A + t1 * AB + s1 * track_dir
                p3 = A + t0 * AB + s1 * track_dir

                quad = [
                    self._to_screen(p0[0], p0[1]),
                    self._to_screen(p1[0], p1[1]),
                    self._to_screen(p2[0], p2[1]),
                    self._to_screen(p3[0], p3[1]),
                ]
                pygame.draw.polygon(surf, color, quad)

    def _draw_cars(self, pos: np.ndarray, angles: np.ndarray, alive: np.ndarray):
        angles_flat = angles.flatten()

        for i in range(len(pos)):
            is_alive = bool(alive[i])
            if not is_alive and not self.show_dead:
                continue

            x, y  = pos[i]
            theta = float(angles_flat[i])
            color = (80, 220, 80) if is_alive else (200, 0, 0)
            px, py = self._to_screen(x, y)

            triangle = [
                (px + CAR_SIZE * math.cos(theta),       py + CAR_SIZE * math.sin(theta)),
                (px + CAR_SIZE * math.cos(theta + 2.4), py + CAR_SIZE * math.sin(theta + 2.4)),
                (px + CAR_SIZE * math.cos(theta - 2.4), py + CAR_SIZE * math.sin(theta - 2.4)),
            ]
            pygame.draw.polygon(self.screen, color, triangle)

    def _draw_rays(self, ray_data: dict, alive: np.ndarray):
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
                lines.append(
                    f"{k:<12}: {v:.3f}" if isinstance(v, float) else f"{k:<12}: {v}"
                )

        for i, line in enumerate(lines):
            surf = self.font.render(line, True, (220, 220, 220))
            self.screen.blit(surf, (10, 10 + i * 20))

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
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return False
        return True

    def close(self):
        pygame.quit()