"""
adaptation_wtrack.py — Fine-tuning des 3 agents pré-entraînés sur W-track
==========================================================================

Sauvegarde le résultat dans le même dossier que le checkpoint source,
sous le nom adaptation_w_track.pt (même architecture que les gen_N.pt).

Usage :
    python -m simulations.adaptation_wtrack --nascar  checkpoints/nascar_.../gen_50.pt --gt      checkpoints/speed_ring_gt_.../gen_50.pt --multi   checkpoints/nascar_rectangle_speed_ring_gt_.../gen_50.pt  --n_adapt_gen 10
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
from pathlib import Path

from learnings.genetic_algorithm.fitness_tracker import FitnessTracker
from learnings.genetic_algorithm.pop_manager_physic import PopulationManager, TrainingLoop
from tracks.circuits_loader import load_circuits


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Adaptation des agents pré-entraînés sur W-track"
    )

    parser.add_argument('--nascar', type=str, required=True)
    parser.add_argument('--gt', type=str, required=True)
    parser.add_argument('--multi', type=str, required=True)

    parser.add_argument('--n_adapt_gen', type=int,   default=10)
    parser.add_argument('--population', type=int,   default=1000)
    parser.add_argument('--n_rays', type=int,   default=9)
    parser.add_argument('--nb_steps', type=int,   default=2000)
    parser.add_argument('--nb_laps', type=int,   default=1)
    parser.add_argument('--device', type=str,   default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--frequency_show', type=int, default = 0)

    parser.add_argument('--mutation_start', type=float, default=0.05)
    parser.add_argument('--mutation_end',   type=float, default=0.005)
    parser.add_argument('--mutation_decay', type=float, default=0.90)

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Adaptation d'un seul agent
# ---------------------------------------------------------------------------

def run_adaptation(label, checkpoint_path, env, checkpoints, walls, args):
    """
    Charge un checkpoint, lance N générations sur W-track,
    sauvegarde adaptation_w_track.pt dans le même dossier que le checkpoint.
    Retourne le chemin du fichier produit.
    """
    print(f"\n{'='*70}")
    print(f"  ADAPTATION : {label}")
    print(f"  Checkpoint : {checkpoint_path}")
    print(f"{'='*70}")

    # -- Population -----------------------------------------------------------
    pop_manager = PopulationManager(
        n_population          = args.population,
        n_rays                = args.n_rays,
        initial_mutation_rate = args.mutation_start,
        final_mutation_rate   = args.mutation_end,
        mutation_decay        = args.mutation_decay,
        device                = args.device,
    )
    pop_manager.load_population_from_file(checkpoint_path)
    pop_manager.generation = 0

    # -- Fitness tracker ------------------------------------------------------
    fitness_tracker = FitnessTracker(
        checkpoints = checkpoints,
        spawn_point = (env.spawn_x, env.spawn_y, env.spawn_angle),
        n_cars = args.population,
        track_width = env.track_width,
        walls = walls,
        device = args.device,
    )

    # -- Boucle d'adaptation --------------------------------------------------
    training_loop = TrainingLoop(
        env                = env,
        population_manager = pop_manager,
        fitness_tracker    = fitness_tracker,
        frequency_show     = args.frequency_show,
        walls              = walls,
        max_laps           = args.nb_laps,
        max_steps          = args.nb_steps,
        phase2_gen         = None,
        phase2_fitness     = None,
    )

    for gen in range(args.n_adapt_gen):
        print(f"\n  Génération {gen + 1}/{args.n_adapt_gen}")
        stats = training_loop.run_generation(generation=gen)
        if stats is None:
            break
        print(
            f"  Best={stats['best_fitness']:.1f} | "
            f"Avg={stats['avg_fitness']:.1f} | "
            f"Mutation={stats['mutation_rate']:.2%}"
        )

    # -- Sauvegarde dans le même dossier que le checkpoint source -------------
    save_path = Path(checkpoint_path).parent / "adaptation_w_track.pt"
    pop_manager.save_population(str(save_path))
    print(f"\n  Sauvegardé : {save_path}")
    return str(save_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA non disponible, passage en CPU")
        args.device = 'cpu'

    # -- W-track chargé une seule fois, partagé entre les 3 agents -----------
    print("\nChargement du circuit W-track...")
    configs     = load_circuits(
        names  = ['w_track'],
        n_cars = args.population,
        n_rays = args.n_rays,
        device = args.device,
    )
    cfg         = configs[0]
    env         = cfg["env"]
    checkpoints = cfg["checkpoints"]
    walls       = cfg["walls"]
    print(f"  Checkpoints : {len(checkpoints)}")

    # -- 3 adaptations séquentielles ------------------------------------------
    agents = [
        ("NASCAR",        args.nascar),
        ("Speed Ring GT", args.gt),
        ("Multi-circuit", args.multi),
    ]

    result_paths = {}
    for label, ckpt in agents:
        result_paths[label] = run_adaptation(
            label, ckpt, env, checkpoints, walls, args
        )

    # -- Commande show_comparison ---------------------------------------------
    print(f"\n{'='*70}")
    print("  ADAPTATION TERMINÉE — commande pour la comparaison :")
    print(f"{'='*70}")
    print(
        f"\npython -m simulations.show_comparison \\\n"
        f"    --nascar  {result_paths['NASCAR']} \\\n"
        f"    --gt      {result_paths['Speed Ring GT']} \\\n"
        f"    --multi   {result_paths['Multi-circuit']}"
    )


if __name__ == "__main__":
    main()