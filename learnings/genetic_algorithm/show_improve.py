import torch
from pathlib import Path
import matplotlib as mpl
from itertools import islice

# Usage: python -m learnings.genetic_algorithm.show_improve

def access_to_file(filepath):
    checkpoint = torch.load(filepath, map_location="cpu") #On load le fichier

    #On va s'en servir de partout afin de récupérer les données qu'on veut
    all_avg = checkpoint["avg_fitness_history"]
    best = checkpoint["best_fitness_history"]

    for i in range(len(all_avg)):
        all_avg[i] = round(all_avg[i])
    for i in range(len(best)):
        best[i] = round(best[i])

    return [all_avg, best]

def get_info(file):
    with open(file, "r", encoding="utf-8") as fichier:
        ligne = next(islice(fichier, 1, None)) # On va à la ligne 2
        nbr_pop = ligne.split(":")[1].strip() # On récupère les données qui se trouvent après les :
        ligne = next(islice(fichier, 0, None))
        nbr_gen = ligne.split(":")[1].strip()

    return (nbr_pop, nbr_gen)


def visualize_stat(stats, i):
    # On s'occupe de faire tout les affichages en print
    tab_avg = stats[0][(10 * i):(10 * (i+1))] # Obligatoire car on rajoutes
    tab_best = stats[1][(10 * i):(10 * (i+1))]
    print(f"Les moyennes de chaque générations : {tab_avg}")
    improve_rate = (tab_avg[-1]  / tab_avg[0]) - 1
    print(f"Taux d'amélioration au cours des 10 : {improve_rate}")


def visualize_stat_all(last_file):
   all_avg, best = access_to_file(last_file)

   


def open_mpl(all_avg, all_best):
    # Ici on va s'occuper d'afficher le graphe avec toutes les moyennes et tout les best de toutes les 10 gen
    pass


def main(path):
    # ---------- Chef d'orchestre ----------
    # On donne le chemin (On se place comme si on était à l'endroit où on le lance, TIPE_genetic de mon côté)
    file = Path(path)
    context_file = f"{file}\config.txt"
    tab_avg_tot = []
    tab_best = []

    # Print d'intro
    print("=" * 100)
    print("Voici quelque stats")
    nbr_pop, nbr_gen = get_info(context_file)
    print(f"Nombre de génération: {nbr_gen} | Nombre d'individu par génération: {nbr_pop}")
    print("=" * 30)
    i = 0

    # On va faire des affichages pour chaque dossier et récupérer toutes les avg pour le graphique
    for i in range(10):
        fichier = f"{file}\gen_{10 * (i + 1)}.pt"
        tab_stats = access_to_file(fichier)
        print("\n")
        print(f"Pour les génération de {10 * i} à {10 * (i + 1)}")
        visualize_stat(tab_stats, i)
        print("\n")
        print("=" * 50)

        i += 1

    # Ici on va s'occuper de faire un affichage qui prend tout le monde
    print("=" * 100)
    print("Voici donc les stats finales, une fois que tout est regroupé :")
    print("-" * 30)
    visualize_stat_all(f"{file}\gen_100.pt")

    # Désormais on va ouvrir matplotlib avec une fenêtre afin d'avoir un graphique pour nos stats
    open_mpl(tab_avg_tot, tab_best)
        

if __name__ == "__main__":
    main("checkpoints/nascar_20260307_232547")