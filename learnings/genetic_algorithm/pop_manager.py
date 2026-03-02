import torch
import numpy as np
from learnings.genetic_algorithm.neural_network import NeuralAgent, VectorizedNeuralPopulation


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

        # ✅ Remplace la liste de NeuralAgent par une instance vectorisée
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
            # ✅ self.population EST l'instance — plus de problème de self manquant
            return self.population.forward_vectorized(observations)

    def select_top_percent(self, fitness_scores, top_percent=0.1):
        n_keep = max(2, int(self.n_population * top_percent))
        sorted_indices = torch.argsort(fitness_scores, descending=True)
        top_indices = sorted_indices[:n_keep]

        # ✅ Retourne des tenseurs de génomes, plus des objets NeuralAgent
        elite_genomes = self.population.genomes[top_indices].clone()

        print(f"Sélection: Top {n_keep} agents "
              f"(fitness max: {fitness_scores[top_indices[0]].item():.2f})")

        return elite_genomes, top_indices

    def reproduce(self, elite_genomes):
        """Crossover + mutation entièrement vectorisés — zéro boucle Python."""
        n_elite = len(elite_genomes)
        genome_size = elite_genomes.shape[1]
        device = self.device

        print(f"Reproduction: {n_elite} parents → {self.n_population} enfants "
              f"(mutation: {self.mutation_rate:.1%})")

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

            # ✅ Crossover uniforme vectorisé
            crossover_mask = torch.rand(n_children, genome_size, device=device) < 0.5
            children = torch.where(crossover_mask, p1, p2)

            # ✅ Mutation vectorisée
            mutation_mask = torch.rand_like(children) < self.mutation_rate
            noise = (torch.rand_like(children) * 2.0 - 1.0) * self.mutation_strength
            children += noise * mutation_mask.float()

            new_genomes[n_elite:] = children

        # ✅ Mise à jour directe du tenseur GPU
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
              f"(fitness: {fitness_scores[best_idx].item():.2f})")

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

    def __init__(self, env, population_manager, fitness_tracker):
        self.env = env
        self.pop_manager = population_manager
        self.fitness_tracker = fitness_tracker

    def run_generation(self, max_steps=1000, render=False):
        observations = self.env.reset()
        self.fitness_tracker.reset()

        for step in range(max_steps):
            actions = self.pop_manager.get_actions(observations)
            observations, rewards, dones = self.env.step(actions)

            self.fitness_tracker.update(
                positions=self.env.pos,
                speeds=self.env.speed,
                alive_mask=self.env.alive
            )

            if not self.env.alive.any():
                print(f"Toute la population est morte au step {step}")
                break

        fitness_scores = self.fitness_tracker.compute_fitness()
        stats = self.pop_manager.evolve(fitness_scores)
        stats.update(self.fitness_tracker.get_statistics())
        return stats

    def train(self, n_generations=100, save_every=10, save_path='checkpoints'):
        import os
        os.makedirs(save_path, exist_ok=True)
        print(f"Début: {n_generations} générations\n" + "=" * 60)

        for gen in range(n_generations):
            print(f"\nGÉNÉRATION {gen + 1}/{n_generations}")
            stats = self.run_generation(max_steps=1000)
            print(f"   Best: {stats['best_fitness']:.2f} | "
                  f"Avg: {stats['avg_fitness']:.2f} | "
                  f"Mutation: {stats['mutation_rate']:.1%}")

            if (gen + 1) % save_every == 0:
                self.pop_manager.save_population(
                    os.path.join(save_path, f'gen_{gen+1}.pt')
                )

        print("\n" + "=" * 60 + "\nEntraînement terminé !")