from learnings.ray_casting.ray_filtering import filter_walls
from learnings.ray_casting.spatial_grid import SpatialGrid

def ray_segment_intersection(origin, direction, A, B, eps=1e-9):
    """
    Calcule l'intersection entre un rayon et un segment.

    Paramètres: 
    origin : tuple(float, float) = Origine du rayon O, donc la voiture
    direction : tuple(float, float) = Direction unitaire du rayon d, donc dans quelle direction il va partir
    A, B : tuple(float, float) = Extrémités du segment AB, donc la tailles + positionnement du mur
    eps : float = Tolérance numérique, donc jusqu'où on pousse le calcul
    ----------
    Retour: 
    t : float | None
        Distance le long du rayon si intersection valide, sinon None
    """
    ox, oy = origin
    dx, dy = direction
    ax, ay = A
    bx, by = B
    # Vecteur du segment
    sx = bx - ax
    sy = by - ay
    # Déterminant (produit vectoriel 2D)
    """Voici à quoi ressemble la matrice dont on calcul le déterminant:
        (dx  sx)
        (dy  sy)
    """
    denom = dx * sy - dy * sx # Direction du rayon + sens du mur
    # Rayon et segment parallèles
    if abs(denom) < eps:
        return None
    # Vecteur AO, direction de la voiture au mur
    ao_x = ax - ox 
    ao_y = ay - oy
    # Paramètres d'intersection
    t = (ao_x * sy - ao_y * sx) / denom
    u = (ao_x * dy - ao_y * dx) / denom
    # Conditions physiques
    if t < 0:
        return None

    if u < 0 or u > 1:
        return None

    #print(t)
    return t#, u # On rajoute u afin de savoir où se truve l'intersection sur le mur et pouvoir le notifier dans l'affichage

def aabb_overlap(axmin, axmax, aymin, aymax, bxmin, bxmax, bymin, bymax):
    return not (
        axmax < bxmin or axmin > bxmax or
        aymax < bymin or aymin > bymax
    )

def ray_aabb(O, D, max_dist):
    ox, oy = O
    dx, dy = D
    x1 = ox + dx * max_dist
    y1 = oy + dy * max_dist
    return (
        min(ox, x1), max(ox, x1),
        min(oy, y1), max(oy, y1)
    )

def ray_distance(O, D, walls, max_dist=300):
    rxmin, rxmax, rymin, rymax = ray_aabb(O, D, max_dist)
    best = max_dist

    for A, B, xmin, xmax, ymin, ymax in walls:
        if not aabb_overlap(rxmin, rxmax, rymin, rymax,
                            xmin, xmax, ymin, ymax):
            continue

        t = ray_segment_intersection(O, D, A, B)
        if t is not None and t < best:
            best = t

    return best
