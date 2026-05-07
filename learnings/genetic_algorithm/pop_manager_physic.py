"""
pop_manager_v2.py - Gestionnaire de population avec curriculum learning
========================================================================

Changements par rapport à pop_manager.py :

  1. Import depuis neural_network_v2
       -> VectorizedNeuralPopulation avec input_size = n_rays + 1 + n_system_obs
       -> 4 sorties au lieu de 2 (steering, throttle, pit_tire, pit_full)

  2. PopulationManager accepte n_system_obs, hidden_size1, hidden_size2

  3. TrainingLoop - Curriculum Learning en 2 phases :

       Phase 1 - Conduite pure
         actions[:, 2:4] = 0.0  -> pit_tire et pit_full masqués (toujours < seuil 0.5)
         L'agent apprend à conduire sans se préoccuper des pit stops.

       Phase 2 - Stratégie pit stop débloquée
         Déclenchée à la génération `phase2_gen` OU quand avg_fitness ≥ `phase2_fitness`
         (le premier critère atteint gagne).
         actions[:, 2:4] actives UNIQUEMENT pour les voitures dans la zone de pit
         (env.in_pit_zone). Hors zone -> masqué à 0.0.

     Le réseau a toujours 4 sorties dès le départ - génome compatible entre les deux phases.
     Transition instantanée, sans rechargement ni modification d'architecture.

Usage :
    from learnings.genetic_algorithm.pop_manager_v2 import PopulationManager, TrainingLoop
"""

import os
import torch
from learnings.genetic_algorithm.neural_network_physic import (
    NeuralAgent,
    VectorizedNeuralPopulation,
    N_SYSTEM_OBS,
)
from render.render_GPU import VectorizedRenderer

# Seuil de décision pour les actions pit (sigmoid > PIT_THRESHOLD -> pit déclenché)
PIT_THRESHOLD = 0.5

class PopulationManager:
    """
    Gère la population, la sélection, le crossover et la mutation.
    Identique à v1 sauf les paramètres de construction de VectorizedNeuralPopulation.
    """

    def __init__(
        self,
        n_population: int   = 1000,
        n_rays: int   = 9,
        n_system_obs: int   = N_SYSTEM_OBS,
        hidden_size1: int   = 32,
        hidden_size2: int   = 16,
        initial_mutation_rate: float = 0.3,
        final_mutation_rate: float = 0.001,
        mutation_decay: float = 0.995,
        mutation_strength: float = 0.3,
        device: str   = 'cuda',
    ):
        self.n_population = n_population
        self.n_rays = n_rays
        self.n_system_obs = n_system_obs
        self.hidden_size1 = hidden_size1
        self.hidden_size2 = hidden_size2
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.mutation_decay = mutation_decay
        self.mutation_strength = mutation_strength

        self.population = VectorizedNeuralPopulation(
            n_agents = n_population,
            n_rays = n_rays,
            n_system_obs = n_system_obs,
            hidden_size1 = hidden_size1,
            hidden_size2 = hidden_size2,
            device = str(self.device),
        )

        self.generation            = 0
        self.best_fitness_history  = []
        self.avg_fitness_history   = []

    # Actions
    def get_actions(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass de toute la population.

        Args:
            observations : Tensor (N, n_rays + 1 + n_system_obs)

        Returns:
            actions : Tensor (N, 4)  [steering, throttle, pit_tire, pit_full]
        """
        with torch.no_grad():
            return self.population.forward_vectorized(observations)

    # Sélection / reproduction
    def select_top_percent(
        self,
        fitness_scores : torch.Tensor,
        top_percent    : float = 0.1, # Le top 10%
    ):
        n_keep        = max(2, int(self.n_population * top_percent))
        sorted_idx    = torch.argsort(fitness_scores, descending=True)
        top_idx       = sorted_idx[:n_keep]
        elite_genomes = self.population.genomes[top_idx].clone()

        print(
            f"Sélection : top {n_keep} agents "
            f"(fitness max : {fitness_scores[top_idx[0]].item():.2f})\n"
        )
        return elite_genomes, top_idx

    def reproduce(self, elite_genomes: torch.Tensor):
        """Crossover + mutation entièrement vectorisés - zéro boucle Python."""
        n_elite     = len(elite_genomes)
        genome_size = elite_genomes.shape[1]
        device      = self.device

        print(f"Reproduction : {n_elite} parents -> {self.n_population} enfants\n")

        new_genomes = torch.empty((self.n_population, genome_size), device=device)
        new_genomes[:n_elite] = elite_genomes   # élitisme

        n_children = self.n_population - n_elite
        if n_children > 0:
            idx1 = torch.randint(n_elite, (n_children,), device=device)
            idx2 = torch.randint(n_elite, (n_children,), device=device)
            p1   = elite_genomes[idx1]
            p2   = elite_genomes[idx2]

            # Crossover uniforme
            mask     = torch.rand(n_children, genome_size, device=device) < 0.5
            children = torch.where(mask, p1, p2)

            # Mutation
            mut_mask = torch.rand_like(children) < self.mutation_rate
            noise    = (torch.rand_like(children) * 2.0 - 1.0) * self.mutation_strength
            children += noise * mut_mask.float()

            new_genomes[n_elite:] = children

        self.population.genomes = new_genomes

    def evolve(self, fitness_scores: torch.Tensor) -> dict:
        elite_genomes, _ = self.select_top_percent(fitness_scores, top_percent=0.1)
        self.reproduce(elite_genomes)

        self.mutation_rate = max(
            self.final_mutation_rate,
            self.mutation_rate * self.mutation_decay
        )
        self.generation += 1

        best = fitness_scores.max().item()
        avg  = fitness_scores.mean().item()
        self.best_fitness_history.append(best)
        self.avg_fitness_history.append(avg)

        return {
            'generation'  : self.generation,
            'best_fitness': best,
            'avg_fitness' : avg,
            'mutation_rate': self.mutation_rate,
            'elite_size'  : int(self.n_population * 0.1),
        }

    # Sauvegarde / chargement
    def save_best_agent(self, filepath: str, fitness_scores: torch.Tensor):
        best_idx   = torch.argmax(fitness_scores).item()
        best_agent = self.population.get_agent(best_idx)
        best_agent.save_to_file(filepath)
        print(
            f"Meilleur agent sauvegardé : {filepath} "
            f"(fitness : {fitness_scores[best_idx].item():.2f})\n"
        )

    def save_population(self, filepath: str):
        torch.save({
            'genomes'             : self.population.genomes,
            'generation'          : self.generation,
            'mutation_rate'       : self.mutation_rate,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history' : self.avg_fitness_history,
            'n_rays'              : self.n_rays,
            'n_system_obs'        : self.n_system_obs,
            'hidden_size1'        : self.hidden_size1,
            'hidden_size2'        : self.hidden_size2,
        }, filepath)
        print(f"Population sauvegardée : {filepath}")

    def load_population_from_file(self, filepath: str):
        ckpt = torch.load(filepath, map_location=self.device)
        if 'genomes' not in ckpt:
            print("Format invalide (clé 'genomes' manquante)")
            return

        self.population.genomes = ckpt['genomes'].to(self.device)
        self.generation          = ckpt.get('generation',   0)
        self.mutation_rate       = ckpt.get('mutation_rate', self.mutation_rate)
        self.n_system_obs        = ckpt.get('n_system_obs',  N_SYSTEM_OBS)
        self.hidden_size1        = ckpt.get('hidden_size1',  32)
        self.hidden_size2        = ckpt.get('hidden_size2',  16)
        print(f"Population chargée : génération {self.generation}")

    def get_statistics(self) -> dict:
        return {
            'generation'          : self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history' : self.avg_fitness_history,
            'current_mutation_rate': self.mutation_rate,
        }

    def reset_population(self):
        self.population = VectorizedNeuralPopulation(
            n_agents     = self.n_population,
            n_rays       = self.n_rays,
            n_system_obs = self.n_system_obs,
            hidden_size1 = self.hidden_size1,
            hidden_size2 = self.hidden_size2,
            device       = str(self.device),
        )
        self.generation    = 0
        self.mutation_rate = 0.3
        print("Population réinitialisée")


class TrainingLoop:
    """
    Boucle d'entraînement avec curriculum learning en 2 phases.

    Phase 1 - Conduite pure
        Les sorties pit_tire et pit_full sont masquées à 0.0 avant d'être
        envoyées à l'environnement. L'agent n'a aucun moyen de déclencher
        un pit stop, il apprend uniquement à conduire vite.

    Phase 2 - Stratégie débloquée
        Déclenchée dès que l'un de ces critères est atteint :
          - génération >= phase2_gen          (critère temporel)
          - avg_fitness >= phase2_fitness     (critère qualitatif)
        Les pit stops deviennent accessibles, mais uniquement si la voiture
        est dans la zone de pit (env.in_pit_zone - défini dans neuronal_env_v2).

    Paramètres de terminaison de génération (priorité dans l'ordre) :
        max_laps  -> s'arrête dès qu'une voiture boucle N tours
        max_steps -> fallback si max_laps non défini
        Si aucun -> tourne jusqu'à ce que toutes les voitures soient mortes
    """

    def __init__(
        self,
        env,
        population_manager,
        fitness_tracker,
        frequency_show: int = 0,
        walls: list = None,
        all_configs: list = None,
        max_laps: int | None = None,
        max_steps: int | None = None,
        phase2_gen: int | None = 10,
        phase2_fitness: float | None = None,
    ):
        self.env = env
        self.pop_manager = population_manager
        self.fitness_tracker = fitness_tracker
        self.frequency_show = frequency_show
        self.walls = walls
        self.renderer = VectorizedRenderer(show_dead=True) if frequency_show != 0 else None

        # Multi-circuit
        self.all_configs = all_configs if all_configs and len(all_configs) > 1 else None
        self._circuit_idx = 0

        # Conditions d'arrêt de génération
        self.max_laps = max_laps
        self.max_steps = max_steps

        # Curriculum learning
        self.phase = 1           # démarre toujours en phase 1
        self.phase2_gen = phase2_gen  # génération de déclenchement (None = désactivé)
        self.phase2_fitness = phase2_fitness  # seuil fitness (None = désactivé)

    # Curriculum learning - masquage des actions
    def _apply_action_mask(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Applique le masque de curriculum sur les sorties pit.

        Phase 1 : pit_tire et pit_full -> 0.0 pour tout le monde
        Phase 2 : pit_tire et pit_full -> 0.0 uniquement pour les voitures hors pit zone

        Args:
            actions : Tensor (N, 4)  - sortie brute de forward_vectorized

        Returns:
            actions : Tensor (N, 4)  - actions filtrées
        """
        if self.phase == 1:
            # Masque total : aucun pit possible
            actions[:, 2:4] = 0.0

        elif self.phase == 2:
            # Masque partiel : pit interdit hors zone
            # env.in_pit_zone est un Tensor (N,) bool mis à jour dans neuronal_env_v2.step()
            if hasattr(self.env, 'in_pit_zone'):
                out_of_pit = ~self.env.in_pit_zone          # (N,) bool
                actions[out_of_pit, 2:4] = 0.0
            else:
                # Sécurité : si l'env ne définit pas in_pit_zone, on masque tout
                actions[:, 2:4] = 0.0

        return actions

    def _check_phase_transition(self, stats: dict):
        """
        Vérifie si les critères de passage en phase 2 sont atteints.
        Affiche un message une seule fois au moment de la transition.
        """
        if self.phase != 1:
            return   # déjà en phase 2, rien à faire

        triggered = False
        reason = ""

        if self.phase2_gen is not None:
            if self.pop_manager.generation >= self.phase2_gen:
                triggered = True
                reason = f"génération {self.pop_manager.generation} ≥ {self.phase2_gen}"

        if self.phase2_fitness is not None and not triggered:
            if stats.get('avg_fitness', 0.0) >= self.phase2_fitness:
                triggered = True
                reason = (
                    f"avg_fitness {stats['avg_fitness']:.2f} "
                    f"≥ seuil {self.phase2_fitness:.2f}"
                )

        if triggered:
            self.phase = 2
            print(f"\n{'='*60}")
            print(f"  PHASE 2 DÉBLOQUÉE - Pit stops activés ({reason})")
            print(f"{'='*60}\n")

    # Rotation de circuit 
    def _rotate_circuit(self):
        cfg = self.all_configs[self._circuit_idx]
        self._circuit_idx = (self._circuit_idx + 1) % len(self.all_configs)

        self.env   = cfg["env"]
        self.walls = cfg["walls"]

        new_checkpoints = cfg["checkpoints"]
        n_new = len(new_checkpoints)

        self.fitness_tracker.checkpoints = torch.tensor(
            new_checkpoints, dtype=torch.float32,
            device=self.fitness_tracker.device
        )
        self.fitness_tracker.n_checkpoints = n_new
        self.fitness_tracker.spawn_point = (
            self.env.spawn_x, self.env.spawn_y, self.env.spawn_angle
        )
        self.fitness_tracker.spawn_tensor = torch.tensor(
            [self.env.spawn_x, self.env.spawn_y],
            dtype=torch.float32, device=self.fitness_tracker.device
        )
        self.fitness_tracker.checkpoint_status = torch.zeros(
            (self.fitness_tracker.n_cars, n_new),
            dtype=torch.bool, device=self.fitness_tracker.device
        )
        self.fitness_tracker.treshold = self.env.track_width

        print(f"  Circuit : {cfg['name']} (difficulté {cfg['difficulty']})")

    # Condition d'arrêt de génération (identique à v1)
    def _is_generation_done(self, step: int) -> bool:
        if not self.env.alive.any():
            return True
        if self.max_laps is not None:
            if bool((self.fitness_tracker.laps_completed >= self.max_laps).any()):
                return True
        if self.max_steps is not None:
            if step >= self.max_steps:
                return True
        return False

    # Boucle de génération
    def run_generation(self, render: bool = False, generation: int = 0) -> dict | None:
        observations = self.env.reset()
        self.fitness_tracker.reset()

        step = 0
        while not self._is_generation_done(step):

            # Forward pass réseau
            actions = self.pop_manager.get_actions(observations)

            # -- Curriculum : masquage des pit stops selon la phase ------------
            actions = self._apply_action_mask(actions)

            # Step environnement
            observations, rewards, dones = self.env.step(actions)

            self.fitness_tracker.update(
                positions  = self.env.pos,
                speeds     = self.env.speed,
                alive_mask = self.env.alive,
            )

            if render:
                render_data   = self.env.get_render_data()
                still_running = self.renderer.render_step(generation, render_data, self.walls)
                if not still_running:
                    return None

            step += 1

        best_laps = int(self.fitness_tracker.laps_completed.max().item())
        if best_laps > 0:
            print(f"Meilleur tour bouclé : {best_laps} (sur {step} steps)")
        else:
            print(f"Génération terminée après {step} steps (aucun tour complet)")

        fitness_scores = self.fitness_tracker.compute_fitness()
        stats          = self.pop_manager.evolve(fitness_scores)
        stats.update(self.fitness_tracker.get_statistics())

        # -- Vérification transition de phase ---------------------------------
        self._check_phase_transition(stats)

        return stats

    # Boucle d'entraînement principale
    def train(
        self,
        n_generations : int = 100,
        save_every    : int = 10,
        save_path     : str = "checkpoints",
    ):
        os.makedirs(save_path, exist_ok=True)
        print(
            f"Début : {n_generations} générations | "
            f"Phase 2 : gen≥{self.phase2_gen} ou fitness≥{self.phase2_fitness}\n"
            + "=" * 100
        )

        for gen in range(n_generations):
            print(f"\nGÉNÉRATION {gen + 1}/{n_generations}  [Phase {self.phase}]")

            if self.all_configs is not None:
                self._rotate_circuit()

            render = (
                self.frequency_show != 0
                and (gen + 1) % self.frequency_show == 0
            )
            stats = self.run_generation(render=render, generation=gen)

            if stats is None:
                print("Fenêtre fermée, arrêt de l'entraînement.")
                return

            print(
                f"Best={stats['best_fitness']:.2f} | "
                f"Avg={stats['avg_fitness']:.2f} | "
                f"Mutation={stats['mutation_rate']:.1%} | "
                f"Meilleurs tours={stats['max_laps']}"
            )
            print("-" * 60)

            if (gen + 1) % save_every == 0:
                self.pop_manager.save_population(
                    os.path.join(save_path, f"gen_{gen + 1}.pt")
                )