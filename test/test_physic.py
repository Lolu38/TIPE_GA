"""
test_genetic_v2.py — Entraînement AG avec physique réaliste
============================================================

Coordonne :
  - neuronal_env_v2     (physique réaliste, pneus, carburant, pluie)
  - neural_network_v2   (4 sorties, entrées étendues)
  - pop_manager_v2      (curriculum learning phases 1 & 2)
  - fitness_tracker     (inchangé)
  - circuits_loader_v2  (instancie neuronal_env_v2)

Usage :
    # Entraînement basique (piste sèche, Medium)
    python -m test.test_physic  --generations 100 --population 100 --frequency_showgen 0 --random_train 0 --circuit speed_ring_gt --nb_laps 1 --nb_steps 2000

    # Pluie dynamique, démarrage en Wet, phase 2 à la génération 20
    python -m test.test_genetic_v2 --generations 100 --rain_mode dynamic --initial_rain 0.5 --compound wet --phase2_gen 20

    # Phase 2 débloquée par seuil de fitness plutôt que par génération
    python -m test.test_genetic_v2 --generations 200 --phase2_gen -1 --phase2_fitness 5000

    # Multi-circuit (les 2 plus faciles), affichage toutes les 5 générations
    python -m test.test_genetic_v2 --random_train 2 --frequency_showgen 5

    # Reprendre un checkpoint
    python -m test.test_genetic_v2 --checkpoint checkpoints/nascar_.../gen_50.pt
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import argparse
from datetime import datetime

from learnings.genetic_algorithm.fitness_tracker import FitnessTracker
from learnings.genetic_algorithm.pop_manager_physic  import PopulationManager, TrainingLoop
from tracks.circuits_loader import load_circuits
from physics.tires_system import (
    COMPOUNDS, HARD, MEDIUM, SOFT, WET, HEAVY_WET
)

# Mapping nom -> indice pour --compound
_COMPOUND_MAP = {
    "hard"     : HARD,
    "medium"   : MEDIUM,
    "soft"     : SOFT,
    "wet"      : WET,
    "heavywet" : HEAVY_WET,
}


# -----------------------------------------------------------------------------
# Arguments
# -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Entraînement AG v2 — physique réaliste (pneus / carburant / pluie)"
    )

    # -- Entraînement ----------------------------------------------------------
    parser.add_argument('--generations',       type=int,   default=100)
    parser.add_argument('--population',        type=int,   default=1000)
    parser.add_argument('--n_rays',            type=int,   default=9)
    parser.add_argument('--save_every',        type=int,   default=10,
                        help="Sauvegarder tous les N générations")
    parser.add_argument('--device',            type=str,   default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--checkpoint',        type=str,   default=None,
                        help="Chemin vers un checkpoint à reprendre")
    parser.add_argument('--frequency_showgen', type=int,   default=-1,
                        help="Afficher le rendu toutes les N générations (-1 = jamais)")

    # -- Circuits --------------------------------------------------------------
    parser.add_argument('--random_train', type=int, default=1,
                        help="Nombre de circuits (triés par difficulté). 0 = utiliser --circuit")
    parser.add_argument('--circuit',      type=str, nargs='+', default=['nascar'],
                        choices=['nascar', 'rectangle', 'speed_ring_gt', 'w_track'],
                        help="Circuit(s) à utiliser si --random_train 0")
    parser.add_argument('--nb_laps',  type=int, default=1,
                        help="Tours pour terminer une génération (prioritaire sur nb_steps)")
    parser.add_argument('--nb_steps', type=int, default=2000,
                        help="Steps max par génération (fallback si nb_laps non atteint)")

    # -- Mutation --------------------------------------------------------------
    parser.add_argument('--mutation_start', type=float, default=0.3)
    parser.add_argument('--mutation_end',   type=float, default=0.05)
    parser.add_argument('--mutation_decay', type=float, default=0.975)

    # -- Systèmes (nouveaux en v2) ------------------------------------------
    parser.add_argument('--rain_mode',    type=str,   default='fixed',
                        choices=['fixed', 'dynamic', 'preset'],
                        help="Mode d'évolution de la pluie")
    parser.add_argument('--initial_rain', type=float, default=0.0,
                        help="Intensité de pluie initiale [0.0=sec, 1.0=déluge]")
    parser.add_argument('--compound',     type=str,   default='medium',
                        choices=list(_COMPOUND_MAP.keys()),
                        help="Composé de départ pour tous les agents")

    # -- Curriculum learning (nouveaux en v2) ----------------------------------
    parser.add_argument('--phase2_gen',     type=int,   default=10,
                        help="Génération de déclenchement de la phase 2 (-1 = désactivé)")
    parser.add_argument('--phase2_fitness', type=float, default=None,
                        help="Seuil avg_fitness pour déclencher la phase 2 (None = désactivé)")

    return parser.parse_args()


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main():
    args = parse_args()

    # -- 1. GPU ----------------------------------------------------------------
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA non disponible, passage en CPU")
        args.device = 'cpu'

    if args.device == 'cuda':
        print(f"GPU : {torch.cuda.get_device_name(0)}")
        print(f"VRAM : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    initial_compound = _COMPOUND_MAP[args.compound]
    phase2_gen       = args.phase2_gen if args.phase2_gen >= 0 else None

    print(
        f"\nSystèmes : pluie={args.rain_mode} (init={args.initial_rain:.2f}) | "
        f"composé={args.compound.capitalize()}"
    )
    print(
        f"Curriculum : phase 2 à gen≥{phase2_gen} "
        f"ou avg_fitness≥{args.phase2_fitness}\n"
    )

    # -- 2. Circuits -----------------------------------------------------------
    configs = load_circuits(
        n                = args.random_train if args.random_train > 0 else None,
        names            = args.circuit      if args.random_train == 0 else None,
        n_cars           = args.population,
        n_rays           = args.n_rays,
        device           = args.device,
        rain_mode        = args.rain_mode,
        initial_rain     = args.initial_rain,
        initial_compound = initial_compound,
    )

    if len(configs) == 1:
        cfg = configs[0]
        print(f"Circuit unique : {cfg['name']} (difficulté {cfg['difficulty']})")
    else:
        print(f"{len(configs)} circuits chargés :")
        for c in configs:
            print(f"  - {c['name']} (difficulté {c['difficulty']})")

    cfg         = configs[0]
    env         = cfg["env"]
    checkpoints = cfg["checkpoints"]
    walls       = cfg["walls"]

    print(f"Voitures : {args.population} | Rayons : {args.n_rays} | Checkpoints : {len(checkpoints)}")

    # -- 3. FitnessTracker ----------------------------------------------------
    fitness_tracker = FitnessTracker(
        checkpoints = checkpoints,
        spawn_point = (env.spawn_x, env.spawn_y, env.spawn_angle),
        n_cars      = args.population,
        track_width = env.track_width,
        device      = args.device,
    )

    # -- 4. PopulationManager -------------------------------------------------
    population_manager = PopulationManager(
        n_population          = args.population,
        n_rays                = args.n_rays,
        initial_mutation_rate = args.mutation_start,
        final_mutation_rate   = args.mutation_end,
        mutation_decay        = args.mutation_decay,
        device                = args.device,
    )

    # -- 5. Checkpoint ---------------------------------------------------------
    if args.checkpoint:
        print(f"\nChargement checkpoint : {args.checkpoint}")
        population_manager.load_population_from_file(args.checkpoint)

    # -- 6. TrainingLoop -------------------------------------------------------
    training_loop = TrainingLoop(
        env                = env,
        population_manager = population_manager,
        fitness_tracker    = fitness_tracker,
        frequency_show     = args.frequency_showgen,
        walls              = walls,
        all_configs        = configs,
        max_laps           = args.nb_laps,
        max_steps          = args.nb_steps,
        phase2_gen         = phase2_gen,
        phase2_fitness     = args.phase2_fitness,
    )

    # -- 7. Dossier de sauvegarde ----------------------------------------------
    circuit_names = "_".join(c["name"] for c in configs)
    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir      = f"checkpoints/{circuit_names}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Sauvegarder la config lisiblement
    config_path = os.path.join(save_dir, "config.txt")
    with open(config_path, 'w') as f:
        f.write(f"Circuits       : {circuit_names}\n")
        f.write(f"Population     : {args.population}\n")
        f.write(f"Générations    : {args.generations}\n")
        f.write(f"Rayons         : {args.n_rays}\n")
        f.write(f"Mutation       : {args.mutation_start} -> {args.mutation_end} (decay={args.mutation_decay})\n")
        f.write(f"Device         : {args.device}\n")
        f.write(f"Pluie          : {args.rain_mode} (init={args.initial_rain})\n")
        f.write(f"Composé        : {args.compound}\n")
        f.write(f"Phase 2 gen    : {phase2_gen}\n")
        f.write(f"Phase 2 fitness: {args.phase2_fitness}\n")

    print(f"\nSauvegardes dans : {save_dir}")

    # -- 8. Entraînement -------------------------------------------------------
    print(f"\nDÉBUT DE L'ENTRAÎNEMENT")
    print("=" * 100 + "\n")

    training_loop.train(
        n_generations = args.generations,
        save_every    = args.save_every,
        save_path     = save_dir,
    )

    # -- 9. Statistiques finales -----------------------------------------------
    print(f"\n{'='*100}")
    print("ENTRAÎNEMENT TERMINÉ")
    print(f"{'='*100}")
    print(f"Fichiers sauvegardés dans : {save_dir}")

    do_visu = input("\nAfficher les courbes de progression ? (Y/N) : ").strip().upper()
    if do_visu == "Y":
        from learnings.genetic_algorithm.show_improve import main as show_main
        show_main(save_dir)


if __name__ == "__main__":
    main()