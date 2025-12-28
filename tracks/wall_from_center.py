import numpy as np
import matplotlib.pyplot as plt


def compute_tangents(centerline: np.ndarray) -> np.ndarray:
    """
    Calcule les vecteurs tangents par différences centrées.
    """
    N = len(centerline)
    tangents = np.zeros_like(centerline)

    for i in range(N):
        p_prev = centerline[i - 1]
        p_next = centerline[(i + 1) % N]  # périodique (circuit fermé)
        tangents[i] = p_next - p_prev

    return tangents


def compute_normals(tangents) -> np.ndarray:
    """
    Calcule les normales unitaires à partir des tangentes.
    """
    normals = np.zeros_like(tangents)

    for i, t in enumerate(tangents):
        norm = np.linalg.norm(t)
        if norm == 0:
            raise ValueError(f"Tangente nulle au point {i}")
        normals[i] = np.array([-t[1], t[0]]) / norm

    return normals


def generate_walls(centerline: np.ndarray, width: float):
    """
    Génère les murs gauche et droit à partir de la ligne centrale.
    """
    tangents = compute_tangents(centerline)
    normals = compute_normals(tangents)

    left_wall = centerline + width * normals
    right_wall = centerline - width * normals

    return left_wall, right_wall


def plot_track(centerline, left_wall, right_wall):
    """
    Visualisation de contrôle géométrique.
    """
    plt.figure(figsize=(8, 8))
    plt.plot(centerline[:, 0], centerline[:, 1], 'k--', label="Ligne centrale")
    plt.plot(left_wall[:, 0], left_wall[:, 1], 'r', label="Mur gauche")
    plt.plot(right_wall[:, 0], right_wall[:, 1], 'b', label="Mur droit")
    plt.axis('equal')
    plt.legend()
    plt.title("Génération des murs du circuit")
    plt.show()
