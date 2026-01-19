import numpy as np

def distance_to_centerline(position, centerline):
    mini = float("inf")
    x_car, ycar = position
    for (x,y) in centerline:
        mini = min(mini, np.sqrt((x_car - x)**2 + (y - ycar)**2))
    return mini

def closest_centerline_index(position, centerline):
    distances = [np.linalg.norm(position - p) for p in centerline]
    return int(np.argmin(distances))
