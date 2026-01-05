def point_segment_distance(px, py, ax, ay, bx, by):
    abx = bx - ax
    aby = by - ay
    apx = px - ax
    apy = py - ay

    denom = abx * abx + aby * aby
    if denom < 1e-12:
        dx = px - ax
        dy = py - ay
        return dx * dx + dy * dy

    t = (apx * abx + apy * aby) / denom
    if t < 0.0:
        dx = px - ax
        dy = py - ay
    elif t > 1.0:
        dx = px - bx
        dy = py - by
    else:
        projx = ax + t * abx
        projy = ay + t * aby
        dx = px - projx
        dy = py - projy

    return dx * dx + dy * dy


def filter_walls(O, D, walls, max_dist):
    ox, oy = O
    dx, dy = D

    max_dist_sq = max_dist * max_dist
    filtered = []

    for (A, B) in walls:
        ax, ay = A
        bx, by = B

        mx = 0.5 * (ax + bx)
        my = 0.5 * (ay + by)

        # Direction
        if (mx - ox) * dx + (my - oy) * dy < 0:
            continue

        # Distance
        if point_segment_distance(ox, oy, ax, ay, bx, by) > max_dist_sq:
            continue

        filtered.append((A, B))

    return filtered

# Cette méthode n'étais pas la bonne, elle ne nous permettait pas de 
# gagner du temps: le temps qu'on gagnait à ne pas vérifier était identique
# au temps perdu lors des calculs: il faut donc passer au hachage spatiale


