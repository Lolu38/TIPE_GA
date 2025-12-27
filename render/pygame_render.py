import pygame
import math

WIDTH, HEIGHT = 600, 600
#SCALE = min(600/800, 600/600) Scale adapté pour pouvoir voir le circuit, sans rien de plus (collé à la fenêtre)
SCALE = 0.6 # Pour avoir le circuit plus petit que la fenêtre -> on voit plus autour
OFFSET_X = 50
OFFSET_Y = 50


class PygameRenderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("TIPE Car Environment")
        self.clock = pygame.time.Clock()

    def render(self, x, y, theta, track, walls=None, rays=None, ray_distance=None):
        self.screen.fill((30, 30, 30))

        # Bordures
        pygame.draw.rect(
            self.screen,
            (200, 200, 200),
            pygame.Rect(0, 0, WIDTH, HEIGHT),
            4
        )

        if track is not None:
            self.draw_inside_map(track)

        # Murs d'abord
        if walls is not None:
            self.draw_walls(walls)
            #print("Drawing", len(walls), "walls")

        # Position voiture
        px = int(x * SCALE + OFFSET_X)
        py = int(y * SCALE + OFFSET_Y)

        # Voiture (triangle)
        size = 6
        points = [
            (px + size * math.cos(theta),
            py + size * math.sin(theta)),
            (px + size * math.cos(theta + 2.5),
            py + size * math.sin(theta + 2.5)),
            (px + size * math.cos(theta - 2.5),
            py + size * math.sin(theta - 2.5))
        ]

        pygame.draw.polygon(self.screen, (255, 100, 100), points)

        # Rayons
        if rays is not None and ray_distance is not None:
            for ((ox, oy), (dx, dy)), d in zip(rays, ray_distance):
                ex = ox + dx * d
                ey = oy + dy * d
                pygame.draw.line(
                    self.screen,
                    (255, 255, 0),
                    (ox, oy),
                    (ex, ey),
                    1
                )

        # Affichage après murs
        pygame.display.flip()
        self.clock.tick(60)

    def close(self):
        pygame.quit()

    def draw_walls(self, walls):
        for (x1, y1), (x2, y2) in walls:
            pygame.draw.line(
                self.screen,
                (255, 255, 255),
                (int(x1 * SCALE + OFFSET_X), int(y1 * SCALE + OFFSET_X)),
                (int(x2 * SCALE + OFFSET_Y), int(y2 * SCALE + OFFSET_Y)),
                3
            )

    def draw_inside_map(self, track, step=5): #Objectif : distinguer le dehors du dedans en visuel, et vérification / débogage
        for x in range(0, WIDTH, step):
            for y in range(0, HEIGHT, step):
                wx = (x - OFFSET_X) / SCALE
                wy = (y - OFFSET_Y) / SCALE

                if track.is_inside(wx, wy):
                    color = (30, 15, 240)   # bleu pour le dedans
                else:
                    color = (40, 120, 40)   # vert pour le dehors

                self.screen.fill(color, (x, y, step, step))

