import numpy as np

def catmull_rom_segment(p0, p1, p2, p3, n_points=20):
    """
    Génère n_points sur la spline Catmull–Rom entre p1 et p2
    """
    p0, p1, p2, p3 = map(np.array, (p0, p1, p2, p3))
    points = []

    for t in np.linspace(0, 1, n_points):
        t2 = t * t
        t3 = t2 * t

        point = 0.5 * (
            (2 * p1)
            + (-p0 + p2) * t
            + (2*p0 - 5*p1 + 4*p2 - p3) * t2
            + (-p0 + 3*p1 - 3*p2 + p3) * t3
        )
        points.append(tuple(point))

    return points

def catmull_rom_spline(points, n_points_per_segment=20, closed=True):
    """
    Génère une spline Catmull–Rom complète
    """
    if closed:
        pts = [points[-1]] + points + [points[0], points[1]]
    else:
        pts = [points[0]] + points + [points[-1]]

    curve = []

    for i in range(1, len(pts) - 2):
        seg = catmull_rom_segment(
            pts[i-1],
            pts[i],
            pts[i+1],
            pts[i+2],
            n_points_per_segment
        )
        curve.extend(seg)

    return curve
