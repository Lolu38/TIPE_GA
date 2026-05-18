import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from pathlib import Path
import matplotlib.pyplot as plt

# Usage:
# python -m Simulations.show_comparison --nascar checkpoints/nascar_20260510_115319/adaptation_w_track.pt --gt checkpoints/speed_ring_gt_20260510_120005/adaptation_w_track.pt --multi checkpoints/nascar_rectangle_speed_ring_gt_20260510_123644/adaptation_w_track.pt



COLORS = {
    "NASCAR"        : "#e74c3c",
    "Speed Ring GT" : "#3498db",
    "Multi-circuit" : "#2ecc71",
}


# ---------------------------------------------------------------------------
# Chargement (identique à show_improve.py)
# ---------------------------------------------------------------------------

def access_to_file(filepath):
    checkpoint = torch.load(filepath, map_location="cpu")
    all_avg = [round(v) for v in checkpoint["avg_fitness_history"]]
    best    = [round(v) for v in checkpoint["best_fitness_history"]]
    return all_avg, best


# ---------------------------------------------------------------------------
# Affichage
# ---------------------------------------------------------------------------

def plot_individual(ax_avg, ax_best, label, color, avg, best):
    """Graphe individuel : avg à gauche, best à droite."""
    gens = list(range(1, len(best) + 1))
    ax_avg.plot(gens, avg,  color=color, linewidth=2, marker="s")
    ax_best.plot(gens, best, color=color, linewidth=2, marker="o")

    for ax, subtitle in zip([ax_avg, ax_best], ["Fitness moyenne", "Meilleure fitness"]):
        ax.set_title(f"{label} — {subtitle}")
        ax.set_ylabel("Fitness")
        ax.grid(True, alpha=0.3)


def plot_all(agents_data: dict):
    """
    n_agents graphes individuels (avg + best côte à côte)
    + 1 graphe de comparaison superposant les 3 agents.
    """
    n_agents = len(agents_data)
    n_rows   = n_agents + 1   # 3 individuels + 1 comparaison

    fig, axes = plt.subplots(n_rows, 2, figsize=(13, 5 * n_rows))
    fig.suptitle("Adaptation sur W-track — Comparaison des agents",
                 fontsize=14, fontweight="bold")

    # --- Graphes individuels -------------------------------------------------
    for row, (label, data) in enumerate(agents_data.items()):
        plot_individual(
            axes[row][0], axes[row][1],
            label, data["color"],
            data["avg"], data["best"]
        )

    # --- Graphe de comparaison -----------------------------------------------
    ax_cmp_avg  = axes[n_agents][0]
    ax_cmp_best = axes[n_agents][1]

    for label, data in agents_data.items():
        color = data["color"]
        gens  = list(range(1, len(data["avg"]) + 1))

        ax_cmp_avg.plot(gens,  data["avg"],  label=label, color=color, linewidth=2, marker="s", linestyle="--")
        ax_cmp_best.plot(gens, data["best"], label=label, color=color, linewidth=2, marker="o")

    for ax, subtitle in zip(
        [ax_cmp_avg, ax_cmp_best],
        ["Comparaison — Fitness moyenne", "Comparaison — Meilleure fitness"]
    ):
        ax.set_title(subtitle)
        ax.set_ylabel("Fitness")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Résumé terminal
# ---------------------------------------------------------------------------

def print_summary(agents_data: dict):
    print("\n" + "=" * 65)
    print(f"{'Agent':<22} {'Best (gen.1)':>14} {'Best (finale)':>14} {'Δ':>8}")
    print("-" * 65)
    for label, data in agents_data.items():
        b0 = data["best"][0]
        bf = data["best"][-1]
        print(f"{label:<22} {b0:>14.0f} {bf:>14.0f} {bf - b0:>+8.0f}")
    print("=" * 65)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(nascar_path, gt_path, multi_path):
    configs = [
        ("NASCAR",        COLORS["NASCAR"],        nascar_path),
        ("Speed Ring GT", COLORS["Speed Ring GT"], gt_path),
        ("Multi-circuit", COLORS["Multi-circuit"], multi_path),
    ]

    agents_data = {}
    for label, color, path in configs:
        if path is None:
            print(f"[{label}] Ignoré (pas de chemin fourni)")
            continue
        if not Path(path).exists():
            print(f"[{label}] Fichier introuvable : {path}")
            continue

        avg, best = access_to_file(path)
        agents_data[label] = {"color": color, "avg": avg, "best": best}
        print(f"[{label}] Chargé — {len(best)} génération(s)")

    if not agents_data:
        print("Aucun agent chargé.")
        return

    print_summary(agents_data)
    plot_all(agents_data)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--nascar", type=str, default=None)
    parser.add_argument("--gt",     type=str, default=None)
    parser.add_argument("--multi",  type=str, default=None)
    args = parser.parse_args()

    main(args.nascar, args.gt, args.multi)