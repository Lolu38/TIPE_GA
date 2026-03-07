import torch
from pathlib import Path
import matplotlib as mpl

def access_to_file(filepath):
    checkpoint = torch.load(filepath, map_location="cpu") #On load le fichier

    #On va s'en servir de partout afin de récupérer les données qu'on veut
    all_avg = checkpoint["all_avg"]
    avg_of_all = checkpoint["avg_fitness_history"]
    best_of_all_10 = checkpoint["best_fitness_history"]

    return [all_avg, avg_of_all, best_of_all_10]

def visualize_stat(stats):
    # On s'occupe de faire tout les affichages en print
    print(f"")


def open_mpl():
    # Ici on va s'occuper d'afficher le graphe avec toutes les moyennes et tout les best de toutes les 10 gen
    pass


def main():
    # ---------- Chef d'orchestre ----------
    # On donne le chemin (On se place comme si on était à l'endroit où on le lance, TIPE_genetic de mon côté)
    dossier = Path("checkpoints")
    tab_avg_tot = []

    # Print d'intro
    print("=" * 100)
    print("Voici quelque stats")
    print("= * 30")

    # On va faire des affichages pour chaque dossier et récupérer toutes les avg pour le graphique
    for fichier in dossier.glob("*.pt"):
        tab_stats = access_to_file(fichier)
        visualize_stat(tab_stats)
        print("=" * 50)

        # Ici on récupère tout les avg pour le graphique
        for avg in tab_stats[0]:
            tab_avg_tot.append(avg)
        



if __name__ == "__main__":
    main()