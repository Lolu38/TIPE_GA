# Cette nouvelle idée va concister à découper notre map en carrés à l'aide 
# d'une grille, et de ne vérifier que les murs qui sont dans 
# le carré de notre voiture

class SpatialGrid:
    def __init__(self, walls, cell_size):
        self.cell_size = cell_size
        self.grid = {}

        for wall in walls:
            self.insert_wall(wall)


    def _cell_id(self, x, y):
        # On refait un nouveau plan, avec des coordonnées plus grandes
        return int(x // self.cell_size), int(y // self.cell_size)
    

    def insert_wall(self, wall):
        (ax, ay), (bx, by) = wall

        # On récupère les dimmensions du mur
        min_x = min(ax, bx)
        min_y = min(ay, by)
        max_x = max(ax, bx)
        max_y = max(ay, by)

        # On crée la boite qui contient le mur
        cx0, cy0 = self._cell_id(min_x, min_y)
        cx1, cy1 = self._cell_id(max_x, max_y)

        # On ajoute notre mur à toutes les cellules par lesquelles il passe
        for cx in range(cx0, cx1 + 1):
            for cy in range(cy0, cy1 + 1):
                # Si la cellule existe déjà, avec des murs dedans, 
                # on la récupère sinon on la créé
                self.grid.setdefault((cx, cy), []).append(wall)

    def query(self, x, y, max_dist):
        cx, cy = self._cell_id(x, y)
        r = int(max_dist // self.cell_size) + 1

        # Création d'une collection d'el unique
        walls = set()

        for dx in range(-r, r + 1):
            for dy in range(-r, r + 1):
                cell = (cx + dx, cy + dy)
                if cell in self.grid:
                    walls.update(self.grid[cell])

        return list(walls)
    
# De nouveau, c'est un echec, on ne filtre pas assez