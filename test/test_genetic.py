"""
Script d'entraînement principal pour l'algorithme génétique sur GPU

Ce script coordonne:
- L'environnement (neuronal_env_improved.py)
- La population (population_manager.py)
- Le fitness tracker (fitness_tracker.py)

Usage:
    python -m test.test_genetic  --generations 100 --population 100 --frequency_showgen 2 --random_train 2
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
from datetime import datetime

# Importer les modules créés
from envs.neuronal_env import VectorizedCarEnv
from learnings.genetic_algorithm.fitness_tracker import FitnessTracker
from learnings.genetic_algorithm.pop_manager import PopulationManager, TrainingLoop
from tracks.circuits_loader import load_circuits


def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Entraînement AG pour voitures autonomes')        
    parser.add_argument('--generations', type=int, default=100,help='Nombre de générations à entraîner')    
    parser.add_argument('--population', type=int, default=1000,help='Taille de la population')    
    parser.add_argument('--n_rays', type=int, default=9,help='Nombre de rayons de détection')    
    parser.add_argument('--max_steps', type=int, default=1000,help='Steps max par génération')    
    parser.add_argument('--save_every', type=int, default=10,help='Sauvegarder tous les N générations')    
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'],help='Device à utiliser')    
    parser.add_argument('--mutation_start', type=float, default=0.3,help='Taux de mutation initial (0.3 = 30%)')    
    parser.add_argument('--mutation_end', type=float, default=0.05,help='Taux de mutation final (0.001 = 0.1%)')    
    parser.add_argument('--mutation_decay', type=float, default=0.975,help='Facteur de décroissance de la mutation')    
    parser.add_argument('--checkpoint', type=str, default=None,help='Chemin vers un checkpoint à reprendre')
    parser.add_argument('--frequency_showgen', type=int, default = -1, help='Fréquence à laquelle on va chercher à afficher nos générations')  
    parser.add_argument('--random_train', type=int, default=1, help='Le nombre de circuit avec lesquels on veut entrainer nos agents, classé par difficulté')
    parser.add_argument('--circuit', type=str, default='nascar',choices=['nascar', 'rectangle', 'high_speed_ring'],help='Circuit à utiliser si on ne choisit pas un nombre juste avant')
    return parser.parse_args()


def main():
    """Fonction principale d'entraînement"""
    args = parse_args()
    
    # --- 1. Vérification GPU ---
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA non disponible, passage en mode CPU")
        args.device = 'cpu'
    
    if args.device == 'cuda':
        print(f"GPU détecté: {torch.cuda.get_device_name(0)}")
        print(f"VRAM disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # --- 2. Création de l'environnement ---

    configs = load_circuits(
        n = args.random_train if args.random_train > 0 else None,
        names = args.circuits     if args.random_train == 0 else None,
        n_cars = args.population,
        n_rays = args.n_rays,
        device = args.device,
    )

    # configs est une liste de dicts :
    # [{"name": str, "difficulty": int, "env": VectorizedCarEnv,
    #   "checkpoints": list, "walls": list}, ...]
    # Si un seul circuit → on passe directement les objets comme avant
    # Si plusieurs       → on passe la liste entière à TrainingLoop

    if len(configs) == 1:
        cfg = configs[0]
        env = cfg["env"]
        checkpoints = cfg["checkpoints"]
        walls = cfg["walls"]
        print(f"\nCircuit unique : {cfg['name']} (difficulté {cfg['difficulty']})")
    else:
        # On prend l'env du premier circuit pour initialiser FitnessTracker
        # (TrainingLoop recevra la liste complète et changera d'env à chaque génération)
        cfg = configs[0]
        env = cfg["env"]
        checkpoints = cfg["checkpoints"]
        walls = cfg["walls"]
        print(f"\n{len(configs)} circuits chargés :")
        for c in configs:
            print(f" - {c['name']} (difficulté {c['difficulty']})")
 
    print(f"Voitures: {args.population}")
    print(f"Rayons: {args.n_rays}")
    print(f"Checkpoints : {len(checkpoints)}")
    
    # --- 3. Création du FitnessTracker ---
    fitness_tracker = FitnessTracker(
        checkpoints = checkpoints,
        spawn_point = (env.spawn_x, env.spawn_y, env.spawn_angle),
        n_cars = args.population,
        track_width = env.track_width,
        device = args.device
    )
    
    # --- 4. Création du PopulationManager ---
    population_manager = PopulationManager(
        n_population = args.population,
        n_rays = args.n_rays,
        initial_mutation_rate = args.mutation_start,
        final_mutation_rate = args.mutation_end,
        mutation_decay = args.mutation_decay,
        device = args.device
    )
    
    # --- 5. Charger un checkpoint si demandé ---
    if args.checkpoint:
        print(f"\nChargement du checkpoint: {args.checkpoint}")
        population_manager.load_population_from_file(args.checkpoint)
    
    # --- 6. Création de la boucle d'entraînement ---
    training_loop = TrainingLoop(
        env = env,
        population_manager= population_manager,
        fitness_tracker = fitness_tracker,
        frequency_show = args.frequency_showgen,
        walls = walls,
        all_configs = configs
    )
    
    # --- 7. Dossier de sauvegarde ---
    circuit_names = "_".join(c["name"] for c in configs)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"checkpoints/{circuit_names}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSauvegardes dans : {save_dir}")
    
    # Sauvegarder la config (pas nécessaire pour l'instant, permettra de déboguer plus tard)
    """config_path = os.path.join(save_dir, "config.txt") 
    with open(config_path, 'w') as f:
        f.write(f"Circuit: {args.circuit}\n")
        f.write(f"Population: {args.population}\n")
        f.write(f"Generations: {args.generations}\n")
        f.write(f"Mutation: {args.mutation_start} -> {args.mutation_end} (decay: {args.mutation_decay})\n")
        f.write(f"Device: {args.device}\n")"""
    
    print(f"\nSauvegardes dans: {save_dir}")
    
    # --- 8. ENTRAÎNEMENT ---
    #print(f"\n{'='*100}")
    print(f"DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*100}\n")
    
    training_loop.train(
        n_generations = args.generations,
        save_every = args.save_every,
        save_path = save_dir
    )
    
    # --- 9. Statistiques finales ---
    stats = population_manager.get_statistics()
    print(f"\n{'='*100}")
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*100}")
    do_visu = input("Voulez vous un visuel des données récupérée (stats, moyenne, best...)? Y/N")
    if do_visu == "y" | do_visu == "Y":
        main(save_dir)
        pass
    
    print(f"\nFichiers sauvegardés dans: {save_dir}")


if __name__ == "__main__":
    main()

