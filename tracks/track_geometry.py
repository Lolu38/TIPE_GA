class RectangularTrack:
    def __init__(self, walls):
        self.outer_bounds = self._extract_bounds(walls[:4])
        self.inner_bounds = self._extract_bounds(walls[4:])

    def _extract_bounds(self, rect_walls):
        xs = []
        ys = []
        for (x1, y1), (x2, y2) in rect_walls:
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        return min(xs), max(xs), min(ys), max(ys)

    def is_inside(self, x, y):
        ox_min, ox_max, oy_min, oy_max = self.outer_bounds
        ix_min, ix_max, iy_min, iy_max = self.inner_bounds

        # dehors du rectangle extérieur
        if not (ox_min <= x <= ox_max and oy_min <= y <= oy_max):
            return False

        # dans la zone intérieure interdite
        if ix_min <= x <= ix_max and iy_min <= y <= iy_max:
            return False

        return True

class AngularTrack:
    def __init__(self, outer_walls, inner_walls):
        self.outer = outer_walls
        self.inner = inner_walls

    def is_inside(self, x, y):
        return (
            self._inside_polygon(x, y, self.outer)
            and not self._inside_polygon(x, y, self.inner)
        )

    def _inside_polygon(self, x, y, polygon):
        inside = False
        n = len(polygon)
        for i in range(n-1):
            (x1,y1) = polygon[i]
            (x2,y2) = polygon[i + 1]
            if ((y1 > y) != (y2 > y)):
                xinters = (y - y1) * (x2 - x1) / (y2 - y1 + 1e-9) + x1
                if x < xinters:
                    inside = not inside
        return inside