def closest_point_index(pos, centerline):
    x, y = pos
    best_i = 0
    best_d2 = float("inf")

    for i, (cx, cy) in enumerate(centerline):
        d2 = (x - cx)**2 + (y - cy)**2
        if d2 < best_d2:
            best_d2 = d2
            best_i = i

    return best_i
