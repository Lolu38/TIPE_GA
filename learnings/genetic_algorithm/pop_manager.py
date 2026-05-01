import torch
from learnings.genetic_algorithm.neural_network import NeuralAgent, VectorizedNeuralPopulation
from render.render_GPU import VectorizedRenderer
import os



class PopulationManager:

    def __init__(
        self,
        n_population=1000,
        n_rays=7,
        initial_mutation_rate=0.3,
        final_mutation_rate=0.001,
        mutation_decay=0.995,
        mutation_strength=0.3,
        device='cuda'
    ):
        self.n_population = n_population
        self.n_rays = n_rays
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        self.mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.mutation_decay = mutation_decay
        self.mutation_strength = mutation_strength

        # Remplace la liste de NeuralAgent par une instance vectorisée
        self.population = VectorizedNeuralPopulation(
            n_agents=n_population,
            n_rays=n_rays,
            device=str(self.device)
        )

        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def get_actions(self, observations):
        """1 seul appel GPU pour toute la population."""
        with torch.no_grad():
            # self.population EST l'instance — plus de problème de self manquant
            return self.population.forward_vectorized(observations)

    def select_top_percent(self, fitness_scores, top_percent=0.1):
        n_keep = max(2, int(self.n_population * top_percent))
        sorted_indices = torch.argsort(fitness_scores, descending=True)
        top_indices = sorted_indices[:n_keep]

        # Retourne des tenseurs de génomes, plus des objets NeuralAgent
        elite_genomes = self.population.genomes[top_indices].clone()

        print(f"Sélection: Top {n_keep} agents "
              f"(fitness max: {fitness_scores[top_indices[0]].item():.2f})\n")

        return elite_genomes, top_indices

    def reproduce(self, elite_genomes):
        """Crossover + mutation entièrement vectorisés — zéro boucle Python."""
        n_elite = len(elite_genomes)
        genome_size = elite_genomes.shape[1]
        device = self.device

        print(f"Reproduction: {n_elite} parents → {self.n_population} enfants\n")

        new_genomes = torch.empty((self.n_population, genome_size), device=device)

        # Élitisme : garder les élites telles quelles
        new_genomes[:n_elite] = elite_genomes

        # Crossover + mutation pour le reste
        n_children = self.n_population - n_elite
        if n_children > 0:
            idx1 = torch.randint(n_elite, (n_children,), device=device)
            idx2 = torch.randint(n_elite, (n_children,), device=device)

            p1 = elite_genomes[idx1]
            p2 = elite_genomes[idx2]

            # Crossover uniforme vectorisé
            crossover_mask = torch.rand(n_children, genome_size, device=device) < 0.5
            children = torch.where(crossover_mask, p1, p2)

            # Mutation vectorisée
            mutation_mask = torch.rand_like(children) < self.mutation_rate
            noise = (torch.rand_like(children) * 2.0 - 1.0) * self.mutation_strength
            children += noise * mutation_mask.float()

            new_genomes[n_elite:] = children

        # Mise à jour directe du tenseur GPU
        self.population.genomes = new_genomes

    def evolve(self, fitness_scores):
        elite_genomes, _ = self.select_top_percent(fitness_scores, top_percent=0.1)
        self.reproduce(elite_genomes)

        self.mutation_rate = max(
            self.final_mutation_rate,
            self.mutation_rate * self.mutation_decay
        )

        self.generation += 1
        best_fitness = fitness_scores.max().item()
        avg_fitness = fitness_scores.mean().item()

        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)

        return {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'mutation_rate': self.mutation_rate,
            'elite_size': int(self.n_population * 0.1),
        }

    def save_best_agent(self, filepath, fitness_scores):
        best_idx = torch.argmax(fitness_scores).item()
        best_agent = self.population.get_agent(best_idx)
        best_agent.save_to_file(filepath)
        print(f"Meilleur agent: {filepath} "
              f"(fitness: {fitness_scores[best_idx].item():.2f})\n")

    def save_population(self, filepath):
        torch.save({
            'genomes': self.population.genomes,
            'generation': self.generation,
            'mutation_rate': self.mutation_rate,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'n_rays': self.n_rays,
        }, filepath)
        print(f"Population sauvegardée: {filepath}")

    def load_population_from_file(self, filepath):
        checkpoint = torch.load(filepath, map_location=self.device)
        if 'genomes' in checkpoint:
            self.population.genomes = checkpoint['genomes'].to(self.device)
            self.generation = checkpoint.get('generation', 0)
            self.mutation_rate = checkpoint.get('mutation_rate', self.mutation_rate)
            print(f"Population chargée: génération {self.generation}")
        else:
            print("Format invalide (clé 'genomes' manquante)")

    def get_statistics(self):
        return {
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'current_mutation_rate': self.mutation_rate,
        }

    def reset_population(self):
        self.population = VectorizedNeuralPopulation(
            n_agents=self.n_population,
            n_rays=self.n_rays,
            device=str(self.device)
        )
        self.generation = 0
        self.mutation_rate = 0.3
        print("Population réinitialisée")


class TrainingLoop:
    """
    Boucle d'entraînement principale.

    Paramètres de terminaison de génération (priorité dans l'ordre) :
      - max_laps  : la génération s'arrête dès qu'une voiture boucle ce nombre de tours.
      - max_steps : fallback utilisé seulement si max_laps n'est pas défini ;
                    la génération s'arrête après ce nombre de steps.
      Si aucun n'est fourni, la génération tourne jusqu'à ce que toutes les
      voitures soient mortes.

    La logique de tours, checkpoints et détection de raccourcis est entièrement
    gérée par FitnessTracker (_check_lap_completion, _check_checkpoints).
    """

    def __init__(
        self,
        env,
        population_manager,
        fitness_tracker,
        frequency_show: int = 0,
        walls=None,
        all_configs=None,
        max_laps: int | None = None,
        max_steps: int | None = None,
    ):
        self.env             = env
        self.pop_manager     = population_manager
        self.fitness_tracker = fitness_tracker
        self.frequency_show  = frequency_show
        self.walls           = walls
        self.renderer        = VectorizedRenderer(show_dead=True) if frequency_show != 0 else None

        # Multi-circuit
        self.all_configs  = all_configs if all_configs and len(all_configs) > 1 else None
        self._circuit_idx = 0

        # Condition d'arrêt de génération
        self.max_laps  = max_laps
        self.max_steps = max_steps

    #  Rotation de circuit
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
        self.fitness_tracker.spawn_point   = (
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
        # Mettre à jour le rayon de détection selon le nouveau circuit
        self.fitness_tracker.treshold = self.env.track_width

        print(f"  Circuit : {cfg['name']} (difficulté {cfg['difficulty']})")

    #  Condition d'arrêt de génération
    def _is_generation_done(self, step: int) -> bool:
        """
        Priorité :
          1. Toutes les voitures mortes → True.
          2. max_laps défini → True dès qu'une voiture atteint max_laps tours.
          3. max_steps défini (fallback, ignoré si max_laps est défini) → True quand step >= max_steps.
          4. Aucun critère → False.
        """
        if not self.env.alive.any():
            return True

        if self.max_laps is not None:
            if bool((self.fitness_tracker.laps_completed >= self.max_laps).any()):
                return True

        if self.max_steps is not None:
            if step >= self.max_steps:
                return True

        return False

    #  Boucle de génération
    def run_generation(self, render: bool = False, generation: int = 0):
        observations = self.env.reset()
        self.fitness_tracker.reset()

        step = 0
        while not self._is_generation_done(step):
            actions = self.pop_manager.get_actions(observations)
            observations, rewards, dones = self.env.step(actions)

            self.fitness_tracker.update(
                positions = self.env.pos,
                speeds = self.env.speed,
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
        return stats

    #  Boucle d'entraînement
    def train(self, n_generations: int = 100, save_every: int = 10, save_path: str = "checkpoints"):
        os.makedirs(save_path, exist_ok=True)
        print(f"Début : {n_generations} générations\n" + "=" * 100)

        for gen in range(n_generations):
            print(f"\nGÉNÉRATION {gen + 1}/{n_generations}")

            if self.all_configs is not None:
                self._rotate_circuit()

            render = (self.frequency_show != 0 and (gen + 1) % self.frequency_show == 0)
            stats  = self.run_generation(render=render, generation=gen)

            if stats is None:
                print("Fenêtre fermée, arrêt de l'entraînement.")
                return

            print(
                f"Avg = {stats['avg_fitness']:.2f} | "
                f"Mutation = {stats['mutation_rate']:.1%} | "
                f"Meilleurs tours = {stats['max_laps']}"
            )
            print("-" * 60)

            if (gen + 1) % save_every == 0:
                self.pop_manager.save_population(
                    os.path.join(save_path, f"gen_{gen + 1}.pt")
                )