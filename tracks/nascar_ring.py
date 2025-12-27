import math

def arc_points(cx, cy, r, a_start, a_end, n=64):
    pts = []
    for i in range(n + 1):
        a = a_start + (a_end - a_start) * i / n
        pts.append((
            cx + r * math.cos(a),
            cy + r * math.sin(a)
        ))
    return pts

def get_walls():
    cx_left, cx_right = 250, 550
    cy = 300
    r_outer = 200
    r_inner = 140

    # ---- OUTER POLYGON (sens trigo) ----
    outer = []

    # haut
    outer.append((cx_left,  cy - r_outer))
    outer.append((cx_right, cy - r_outer))

    # arc droit
    outer += arc_points(cx_right, cy, r_outer, -math.pi/2, math.pi/2)

    # bas
    outer.append((cx_left, cy + r_outer))

    # arc gauche
    outer += arc_points(cx_left, cy, r_outer, math.pi/2, 3*math.pi/2)

    # ---- INNER POLYGON (sens inverse !) ----
    inner = []

    # haut
    inner.append((cx_left, cy + r_inner))
    inner.append((cx_right, cy + r_inner))

    # arc droit
    inner += arc_points(cx_right, cy, r_inner, math.pi/2, -math.pi/2)

    # bas
    inner.append((cx_right, cy - r_inner))

    # arc gauche
    inner += arc_points(cx_left, cy, r_inner, 3*math.pi/2, math.pi/2)

    return outer, inner

def get_spawn():
    x = 400
    y = 145
    theta = 0.0
    return x,y,theta