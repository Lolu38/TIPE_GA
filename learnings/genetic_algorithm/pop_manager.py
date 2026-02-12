import torch
import numpy as np
from neural_network import NeuralAgent


class PopulationManager:
    """
    Gère l'évolution d'une population d'agents par algorithme génétique.
    
    Processus:
    1. Évaluation: toute la population joue, on calcule leur fitness
    2. Sélection: on garde les top 10%
    3. Reproduction: crossover + mutation pour créer la nouvelle génération
    4. Répéter
    """
    
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
        """
        Args:
            n_population: Taille de la population (ex: 1000)
            n_rays: Nombre de rayons de détection
            initial_mutation_rate: Taux de mutation au départ (ex: 0.3 = 30%)
            final_mutation_rate: Taux de mutation minimum (ex: 0.001 = 0.1%)
            mutation_decay: Facteur de décroissance par génération (ex: 0.995)
            mutation_strength: Amplitude des mutations [-strength, +strength]
            device: 'cuda' ou 'cpu'
        """
        self.n_population = n_population
        self.n_rays = n_rays
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # Paramètres de mutation
        self.mutation_rate = initial_mutation_rate
        self.final_mutation_rate = final_mutation_rate
        self.mutation_decay = mutation_decay
        self.mutation_strength = mutation_strength
        
        # Créer la population initiale
        self.population = [
            NeuralAgent(n_rays=n_rays).to(self.device) 
            for _ in range(n_population)
        ]
        
        # Statistiques
        self.generation = 0
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
    def select_top_percent(self, fitness_scores, top_percent=0.1):
        """
        Sélectionne les meilleurs agents (top 10% par défaut)
        
        Args:
            fitness_scores: Tensor (n_population,) - scores de chaque agent
            top_percent: Fraction à garder (0.1 = 10%)
        
        Returns:
            list of NeuralAgent - les meilleurs agents
        """
        # Nombre d'agents à garder
        n_keep = max(1, int(self.n_population * top_percent))
        
        # Indices des meilleurs (triés par fitness décroissante)
        sorted_indices = torch.argsort(fitness_scores, descending=True)
        top_indices = sorted_indices[:n_keep].cpu().numpy()
        
        # Extraire les meilleurs agents
        elite = [self.population[i].clone() for i in top_indices]
        
        print(f"Sélection: Top {n_keep} agents gardés (fitness max: {fitness_scores[top_indices[0]].item():.2f})")
        
        return elite, top_indices
    
    def crossover(self, parent1, parent2):
        """
        Crée un enfant par croisement génétique entre deux parents
        
        Args:
            parent1, parent2: NeuralAgent
        
        Returns:
            NeuralAgent enfant
        """
        return parent1.crossover(parent2, crossover_rate=0.5)
    
    def reproduce(self, elite):
        """
        Génère une nouvelle population à partir des élites
        
        Stratégie: Les top 10% se reproduisent aléatoirement pour créer 1000 enfants
        
        Args:
            elite: list of NeuralAgent - les meilleurs agents
        
        Returns:
            list of NeuralAgent - nouvelle population complète
        """
        new_population = []
        n_elite = len(elite)
        
        print(f"Reproduction: {n_elite} parents → {self.n_population} enfants (mutation: {self.mutation_rate:.1%})")
        
        # 1. Garder les élites telles quelles (élitisme)
        for agent in elite:
            new_population.append(agent.clone())
        
        # 2. Créer les enfants par crossover aléatoire
        while len(new_population) < self.n_population:
            # Sélectionner deux parents aléatoirement
            parent1 = elite[np.random.randint(n_elite)]
            parent2 = elite[np.random.randint(n_elite)]
            
            # Crossover
            child = self.crossover(parent1, parent2)
            
            # Mutation
            child.mutate(
                mutation_rate=self.mutation_rate,
                mutation_strength=self.mutation_strength
            )
            
            new_population.append(child)
        
        return new_population
    
    def evolve(self, fitness_scores):
        """
        Fait évoluer la population: sélection + reproduction
        
        Args:
            fitness_scores: Tensor (n_population,) - fitness de chaque agent
        
        Returns:
            dict avec statistiques de la génération
        """
        # 1. Sélection des meilleurs
        elite, top_indices = self.select_top_percent(fitness_scores, top_percent=0.1)
        
        # 2. Reproduction
        self.population = self.reproduce(elite)
        
        # 3. Décroissance de la mutation
        self.mutation_rate = max(
            self.final_mutation_rate,
            self.mutation_rate * self.mutation_decay
        )
        
        # 4. Statistiques
        self.generation += 1
        best_fitness = fitness_scores.max().item()
        avg_fitness = fitness_scores.mean().item()
        
        self.best_fitness_history.append(best_fitness)
        self.avg_fitness_history.append(avg_fitness)
        
        stats = {
            'generation': self.generation,
            'best_fitness': best_fitness,
            'avg_fitness': avg_fitness,
            'mutation_rate': self.mutation_rate,
            'elite_size': len(elite)
        }
        
        return stats
    
    def get_actions(self, observations):
        """
        Calcule les actions pour toute la population
        
        Args:
            observations: Tensor (n_population, n_rays + 1)
        
        Returns:
            actions: Tensor (n_population, 2) - [steering, throttle]
        """
        actions = []
        
        with torch.no_grad():  # Pas besoin de gradients pour l'inférence
            for i, agent in enumerate(self.population):
                action = agent(observations[i:i+1])
                actions.append(action)
        
        return torch.cat(actions, dim=0)
    
    def save_best_agent(self, filepath, fitness_scores):
        """
        Sauvegarde le meilleur agent de la génération
        
        Args:
            filepath: Chemin du fichier de sauvegarde
            fitness_scores: Tensor (n_population,)
        """
        best_idx = torch.argmax(fitness_scores).item()
        best_agent = self.population[best_idx]
        best_agent.save_to_file(filepath)
        print(f"Meilleur agent sauvegardé: {filepath} (fitness: {fitness_scores[best_idx].item():.2f})")
    
    def load_population_from_file(self, filepath):
        """
        Charge une population sauvegardée
        
        Args:
            filepath: Chemin du fichier
        """
        checkpoint = torch.load(filepath)
        
        if 'population' in checkpoint:
            self.population = checkpoint['population']
            self.generation = checkpoint.get('generation', 0)
            self.mutation_rate = checkpoint.get('mutation_rate', self.mutation_rate)
            print(f"Population chargée: génération {self.generation}")
        else:
            print("Format de fichier invalide")
    
    def save_population(self, filepath):
        """
        Sauvegarde toute la population + état de l'évolution
        
        Args:
            filepath: Chemin du fichier de sauvegarde
        """
        checkpoint = {
            'population': self.population,
            'generation': self.generation,
            'mutation_rate': self.mutation_rate,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history
        }
        
        torch.save(checkpoint, filepath)
        print(f"Population complète sauvegardée: {filepath}")
    
    def get_statistics(self):
        """
        Retourne les statistiques d'évolution
        
        Returns:
            dict avec historique des fitness
        """
        return {
            'generation': self.generation,
            'best_fitness_history': self.best_fitness_history,
            'avg_fitness_history': self.avg_fitness_history,
            'current_mutation_rate': self.mutation_rate
        }
    
    def reset_population(self):
        """Réinitialise la population (nouveaux poids aléatoires)"""
        self.population = [
            NeuralAgent(n_rays=self.n_rays).to(self.device) 
            for _ in range(self.n_population)
        ]
        self.generation = 0
        self.mutation_rate = 0.3  # Reset au taux initial
        print("Population réinitialisée")


class TrainingLoop:
    """
    Boucle d'entraînement complète qui coordonne:
    - Environnement (neuronal_env)
    - Population (population_manager)
    - Fitness (fitness_tracker)
    """
    
    def __init__(self, env, population_manager, fitness_tracker):
        """
        Args:
            env: VectorizedCarEnv - environnement de simulation
            population_manager: PopulationManager
            fitness_tracker: FitnessTracker
        """
        self.env = env
        self.pop_manager = population_manager
        self.fitness_tracker = fitness_tracker
    
    def run_generation(self, max_steps=1000, render=False):
        """
        Exécute une génération complète
        
        Args:
            max_steps: Nombre max de steps par génération
            render: Si True, affiche la simulation (ralentit beaucoup)
        
        Returns:
            dict avec statistiques de la génération
        """
        # Reset environnement et fitness
        observations = self.env.reset()
        self.fitness_tracker.reset()
        
        # Simulation
        for step in range(max_steps):
            # Obtenir les actions de toute la population
            actions = self.pop_manager.get_actions(observations)
            
            # Step dans l'environnement
            observations, rewards, dones = self.env.step(actions)
            
            # Mise à jour fitness
            self.fitness_tracker.update(
                positions=self.env.pos,
                speeds=self.env.speed,
                alive_mask=self.env.alive
            )
            
            # Affichage optionnel
            if render and step % 10 == 0:
                # Appeler ton renderer PyGame ici si besoin
                pass
            
            # Arrêt si tout le monde est mort
            if not self.env.alive.any():
                print(f"Toute la population est morte au step {step}")
                break
        
        # Calcul fitness finale
        fitness_scores = self.fitness_tracker.compute_fitness()
        
        # Évolution (sélection + reproduction)
        stats = self.pop_manager.evolve(fitness_scores)
        
        # Ajouter les stats de fitness
        fitness_stats = self.fitness_tracker.get_statistics()
        stats.update(fitness_stats)
        
        return stats
    
    def train(self, n_generations=100, save_every=10, save_path='checkpoints'):
        """
        Boucle d'entraînement complète
        
        Args:
            n_generations: Nombre de générations à entraîner
            save_every: Sauvegarder tous les N générations
            save_path: Dossier de sauvegarde
        """
        import os
        os.makedirs(save_path, exist_ok=True)
        
        print(f"Début de l'entraînement: {n_generations} générations")
        print("=" * 60)
        
        for gen in range(n_generations):
            print(f"\nGÉNÉRATION {gen + 1}/{n_generations}")
            
            # Exécuter une génération
            stats = self.run_generation(max_steps=1000)
            
            # Affichage
            print(f"   Best fitness: {stats['best_fitness']:.2f}")
            print(f"   Avg fitness:  {stats['avg_fitness']:.2f}")
            print(f"   Max laps:     {stats['max_laps']}")
            print(f"   Mutation:     {stats['mutation_rate']:.1%}")
            
            # Sauvegarde périodique
            if (gen + 1) % save_every == 0:
                filepath = os.path.join(save_path, f'gen_{gen+1}.pt')
                self.pop_manager.save_population(filepath)
        
        print("\n" + "=" * 60)
        print("Entraînement terminé !")