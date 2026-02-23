"""
Script d'entraînement principal pour l'algorithme génétique sur GPU

Ce script coordonne:
- L'environnement (neuronal_env_improved.py)
- La population (population_manager.py)
- Le fitness tracker (fitness_tracker.py)

Usage:
    python train_genetic.py --circuit nascar --generations 100 --population 1000
"""

import torch
import argparse
import os
from datetime import datetime

# Importer les modules créés
from envs.neuronal_env import VectorizedCarEnv, build_env_from_track_config
from learnings.genetic_algorithm.fitness_tracker import FitnessTracker
from learnings.genetic_algorithm.pop_manager import PopulationManager, TrainingLoop


def parse_args():
    """Parse les arguments de ligne de commande"""
    parser = argparse.ArgumentParser(description='Entraînement AG pour voitures autonomes')    
    parser.add_argument('--circuit', type=str, default='nascar',choices=['nascar', 'rectangle', 'high_speed_ring'],help='Circuit à utiliser')    
    parser.add_argument('--generations', type=int, default=100,help='Nombre de générations à entraîner')    
    parser.add_argument('--population', type=int, default=1000,help='Taille de la population')    
    parser.add_argument('--n_rays', type=int, default=9,help='Nombre de rayons de détection')    
    parser.add_argument('--max_steps', type=int, default=1000,help='Steps max par génération')    
    parser.add_argument('--save_every', type=int, default=10,help='Sauvegarder tous les N générations')    
    parser.add_argument('--device', type=str, default='cuda',choices=['cuda', 'cpu'],help='Device à utiliser')    
    parser.add_argument('--mutation_start', type=float, default=0.3,help='Taux de mutation initial (0.3 = 30%)')    
    parser.add_argument('--mutation_end', type=float, default=0.01,help='Taux de mutation final (0.001 = 0.1%)')    
    parser.add_argument('--mutation_decay', type=float, default=0.95,help='Facteur de décroissance de la mutation')    
    parser.add_argument('--checkpoint', type=str, default=None,help='Chemin vers un checkpoint à reprendre')    
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
    print(f"\nChargement du circuit: {args.circuit}")
    env, checkpoints = build_env_from_track_config(
        track_name=args.circuit,
        n_cars=args.population,
        n_rays=args.n_rays,
        device=args.device
    )
    print(f"Voitures: {args.population}")
    print(f"Rayons: {args.n_rays}")
    print(f"Checkpoints: {len(checkpoints)}")
    
    # --- 3. Création du FitnessTracker ---
    fitness_tracker = FitnessTracker(
        checkpoints=checkpoints,
        spawn_point=(env.spawn_x, env.spawn_y, env.spawn_angle),
        n_cars=args.population,
        track_width= env.track_width,
        device=args.device
    )
    
    # --- 4. Création du PopulationManager ---
    population_manager = PopulationManager(
        n_population=args.population,
        n_rays=args.n_rays,
        initial_mutation_rate=args.mutation_start,
        final_mutation_rate=args.mutation_end,
        mutation_decay=args.mutation_decay,
        device=args.device
    )
    
    # --- 5. Charger un checkpoint si demandé ---
    if args.checkpoint:
        print(f"\nChargement du checkpoint: {args.checkpoint}")
        population_manager.load_population_from_file(args.checkpoint)
    
    # --- 6. Création de la boucle d'entraînement ---
    training_loop = TrainingLoop(
        env=env,
        population_manager=population_manager,
        fitness_tracker=fitness_tracker
    )
    
    # --- 7. Dossier de sauvegarde ---
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"checkpoints/{args.circuit}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Sauvegarder la config
    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Circuit: {args.circuit}\n")
        f.write(f"Population: {args.population}\n")
        f.write(f"Generations: {args.generations}\n")
        f.write(f"Mutation: {args.mutation_start} -> {args.mutation_end} (decay: {args.mutation_decay})\n")
        f.write(f"Device: {args.device}\n")
    
    print(f"\nSauvegardes dans: {save_dir}")
    
    # --- 8. ENTRAÎNEMENT ---
    print(f"\n{'='*60}")
    print(f"DÉBUT DE L'ENTRAÎNEMENT")
    print(f"{'='*60}\n")
    
    training_loop.train(
        n_generations=args.generations,
        save_every=args.save_every,
        save_path=save_dir
    )
    
    # --- 9. Sauvegarde finale ---
    final_path = os.path.join(save_dir, "final_population.pt")
    population_manager.save_population(final_path)
    
    # Sauvegarder le meilleur agent
    best_agent_path = os.path.join(save_dir, "best_agent.pt")
    obs = env.reset()
    fitness_tracker.reset()
    
    # Simuler une dernière génération pour obtenir le fitness
    for step in range(args.max_steps):
        actions = population_manager.get_actions(obs)
        obs, _, dones = env.step(actions)
        fitness_tracker.update(env.pos, env.speed, env.alive)
        if env.is_all_dead():
            break
    
    fitness_scores = fitness_tracker.compute_fitness()
    population_manager.save_best_agent(best_agent_path, fitness_scores)
    
    # --- 10. Statistiques finales ---
    stats = population_manager.get_statistics()
    print(f"\n{'='*60}")
    print(f"ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*60}")
    print(f"Générations: {stats['generation']}")
    print(f"Meilleur fitness final: {stats['best_fitness_history'][-1]:.2f}")
    print(f"Fitness moyen final: {stats['avg_fitness_history'][-1]:.2f}")
    print(f"\nProgression:")
    print(f"Génération 1:   Best={stats['best_fitness_history'][0]:.2f}, Avg={stats['avg_fitness_history'][0]:.2f}")
    print(f"Génération {stats['generation']}: Best={stats['best_fitness_history'][-1]:.2f}, Avg={stats['avg_fitness_history'][-1]:.2f}")
    
    # Amélioration
    improvement = stats['avg_fitness_history'][-1] / max(stats['avg_fitness_history'][0], 0.01)
    print(f"\nAmélioration: x{improvement:.2f}")
    
    print(f"\nFichiers sauvegardés dans: {save_dir}")


if __name__ == "__main__":
    main()