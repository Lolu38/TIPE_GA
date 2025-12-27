import math
import numpy as np

def generate_rays(position, angle, nbr_ray):
    """
    position: (x, y)
    angle: float (rad)
    ray_angles: liste d'angles relatifs (rad)
    """
    rays = []
    if nbr_ray is None:
        return rays
    ray_angles = np.linspace(-math.pi/2, math.pi/2, nbr_ray)

    x, y = position

    for alpha in ray_angles:
        theta = angle + alpha
        direction = (math.cos(theta), math.sin(theta))

        rays.append({
            "origin": (x, y),
            "direction": direction
        })

    return rays
