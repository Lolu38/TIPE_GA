import torch

def compute_ray_intersections(ray_origins, ray_directions, wall_starts, wall_ends, max_dist=1000.0):
    """
    Calcule les intersections pour N rayons contre M murs en parallèle sur GPU.
    Args:
        ray_origins: Tensor (N_rays, 2) -> (x, y) de départ
        ray_directions: Tensor (N_rays, 2) -> (dx, dy) vecteur unitaire
        wall_starts: Tensor (N_walls, 2) -> Point A des murs
        wall_ends: Tensor (N_walls, 2) -> Point B des murs
        max_dist: Distance max si pas d'intersection
    
    Returns:
        distances: Tensor (N_rays,) -> Distance vers le mur le plus proche
    """
    
    # --- 1. Préparation des dimensions pour le Broadcasting ---
    # On veut comparer chaque rayon (i) avec chaque mur (j).
    # Dimensions cibles : (N_rays, N_walls, 2)
    
    # Rayons : (N_rays, 1, 2)
    ro = ray_origins.unsqueeze(1)
    rd = ray_directions.unsqueeze(1)
    
    # Murs : (1, N_walls, 2)
    p1 = wall_starts.unsqueeze(0)
    p2 = wall_ends.unsqueeze(0)
    
    # --- 2. Mathématiques d'intersection (Cramer/Determinant) ---
    # Mur vecteur : v = p2 - p1
    v = p2 - p1  # (1, M, 2)
    
    # On cherche t (distance rayon) et u (position sur le segment mur) tel que :
    # ro + t * rd = p1 + u * v
    # Produit en croix 2D (determinant) : x1*y2 - y1*x2
    def cross_product(a, b):
        return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]

    # Différence entre origine rayon et début mur
    d_ro_p1 = p1 - ro # (N, M, 2)
    
    # Denominateur = cross(rd, v)
    denom = cross_product(rd, v) # (N, M)
    
    # Eviter la division par zéro (rayons parallèles aux murs)
    # On remplace 0 par epsilon
    denom = torch.where(torch.abs(denom) < 1e-9, torch.tensor(1e-9, device=denom.device), denom)
    
    # t = cross(p1 - ro, v) / denom
    t = cross_product(d_ro_p1, v) / denom # (N, M)
    
    # u = cross(p1 - ro, rd) / denom
    # Attention au signe : p1 - ro = - (ro - p1), d'où l'ordre inversé ou signe
    # La formule standard est (ro - p1) cross rd / cross(v, rd)
    # Ici on adapte pour coller aux tenseurs
    u = cross_product(d_ro_p1, rd) / denom # (N, M)
    
    # --- 3. Filtrage des intersections valides ---
    # Condition 1 : t > 0 (le mur est devant)
    # Condition 2 : 0 <= u <= 1 (on tape bien dans le segment du mur, pas la droite infinie)
    mask = (t > 0) & (u >= 0) & (u <= 1)
    
    # On met une distance infinie là où il n'y a pas d'intersection valide
    valid_t = torch.where(mask, t, torch.tensor(max_dist, device=t.device))
    
    # --- 4. Réduction : Trouver le mur le plus proche pour chaque rayon ---
    min_dist, _ = torch.min(valid_t, dim=1) # (N_rays,)
    
    return min_dist