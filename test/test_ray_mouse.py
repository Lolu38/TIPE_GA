import math
import pygame
from learnings.ray_casting.intersections import ray_segment_intersection

wall = [
    ((350, 50), (350, 350))    
]

origin = (200, 300)

angle_ray = [-math.pi/4, 0, math.pi/4]

def unit_vector(angle): #création d'un vecteur directeur de mon rayon
    return (math.cos(angle), math.sin(angle))

# --- Pour observation/interaction avec pygame ---
drag_origin = False
drag_A = False
drag_B = False

# --- Pour avoir la distance ---
def dist(p, q):
    return ((p[0]-q[0])**2 + (p[1]-q[1])**2)**0.5


pygame.init()
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Ray casting test")

clock = pygame.time.Clock()

def draw():
    screen.fill((30, 30, 30))

    # Mur
    for A, B in wall:
        pygame.draw.line(screen, (255, 255, 255), A, B, 3)

    # Origine
    pygame.draw.circle(screen, (255, 0, 0), origin, 6)

    for angle in angle_ray:
        d = unit_vector(angle)

        min_t = None
        hit_point = None

        for A, B in wall:
            t = ray_segment_intersection(origin, d, A, B)
            if t is not None and (min_t is None or t < min_t):
                min_t = t

        # Dessin du rayon
        if min_t is not None:
            hit_point = (
                origin[0] + min_t * d[0],
                origin[1] + min_t * d[1],
            )
            pygame.draw.line(screen, (0, 255, 0), origin, hit_point, 2)
            pygame.draw.circle(screen, (255, 255, 0), hit_point, 5)
        else:
            # Rayon sans impact
            end = (
                origin[0] + 1000 * d[0],
                origin[1] + 1000 * d[1],
            )
            pygame.draw.line(screen, (255, 100, 100), origin, end, 1)

running = True
while running:
    clock.tick(60)

    mx, my = pygame.mouse.get_pos()

    for event in pygame.event.get():

        if event.type == pygame.MOUSEBUTTONDOWN:
            if dist((mx, my), origin) < 10:
                drag_origin = True
            elif dist((mx, my), wall[0][0]) < 10:
                drag_A = True
            elif dist((mx, my), wall[0][1]) < 10:
                drag_B = True

        if event.type == pygame.MOUSEBUTTONUP:
            drag_origin = drag_A = drag_B = False

        if event.type == pygame.MOUSEMOTION:
            if drag_origin:
                origin = (mx, my)
            elif drag_A:
                wall[0] = ((mx, my), wall[0][1])
            elif drag_B:
                wall[0] = (wall[0][0], (mx, my))

    draw()
    pygame.display.flip()
pygame.quit()