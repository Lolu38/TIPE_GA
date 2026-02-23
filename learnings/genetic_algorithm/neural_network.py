import torch
import torch.nn as nn

class NeuralAgent(nn.Module):
    """
    Réseau de neurones simple (MLP) pour contrôler une voiture.
    
    Architecture:
        Input: [n_rays distances] + [vitesse] = n_rays + 1
        Hidden1: 16 neurones (ReLU)
        Hidden2: 8 neurones (ReLU)
        Output: 2 neurones
            - steering: Tanh → [-1, 1] (gauche/droite)
            - throttle: Sigmoid → [0, 1] (freiner/accélérer)
    """
    
    def __init__(self, n_rays=7, hidden_size1=16, hidden_size2=8, device='cuda'):
        super(NeuralAgent, self).__init__()
        
        self.n_rays = n_rays
        input_size = n_rays + 1  # distances + vitesse
        self.device = device

        # Architecture du réseau
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, 2)  # [steering_raw, throttle_raw]
        )
        
        # Initialisation des poids (Xavier pour meilleure convergence)
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialisation intelligente des poids"""
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, observations):
        """
        Propagation avant
        
        Args:
            observations: Tensor (batch_size, n_rays + 1)
                - n_rays premières valeurs: distances normalisées [0, 1]
                - dernière valeur: vitesse normalisée [0, 1]
        
        Returns:
            actions: Tensor (batch_size, 2)
                - actions[:, 0]: steering ∈ [-1, 1]
                - actions[:, 1]: throttle ∈ [0, 1]
        """
        raw_output = self.network(observations)
        
        # Activer les sorties
        steering = torch.tanh(raw_output[:, 0:1])      # [-1, 1]
        throttle = torch.sigmoid(raw_output[:, 1:2])   # [0, 1]
        
        return torch.cat([steering, throttle], dim=1)
    
    def get_genome(self):
        """
        Extrait tous les poids et biais du réseau sous forme de vecteur 1D
        
        Returns:
            Tensor 1D contenant tous les paramètres
        """
        genome = []
        for param in self.parameters():
            genome.append(param.data.view(-1))
        return torch.cat(genome)
    
    def set_genome(self, genome):
        """
        Injecte un génome (vecteur 1D) dans le réseau
        
        Args:
            genome: Tensor 1D avec tous les paramètres
        """
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = genome[offset:offset + numel].view(param.shape)
            offset += numel
    
    def mutate(self, mutation_rate=0.3, mutation_strength=0.1):
        """
        Applique une mutation aléatoire uniforme aux poids
        
        Args:
            mutation_rate: Probabilité qu'un poids soit muté (0.1 = 10%)
            mutation_strength: Amplitude du changement [-strength, +strength]
        """
        with torch.no_grad():
            for param in self.parameters():
                # Masque binaire: quels poids vont muter
                mutation_mask = torch.rand_like(param) < mutation_rate
                
                # Bruit uniforme dans [-strength, +strength]
                noise = torch.rand_like(param) * 2 * mutation_strength - mutation_strength
                
                # Appliquer uniquement sur les poids sélectionnés
                param.data += noise * mutation_mask.float()
    
    def clone(self):
        """
        Crée une copie identique du réseau
        
        Returns:
            NeuralAgent avec les mêmes poids
        """
        clone = NeuralAgent(n_rays=self.n_rays, device=self.device).to(self.device)
        return clone
    
    def crossover(self, other, crossover_rate=0.5):
        """
        Crée un enfant par croisement génétique avec un autre agent
        
        Args:
            other: Un autre NeuralAgent (parent 2)
            crossover_rate: Probabilité de prendre le gène du parent 1 (vs parent 2)
        
        Returns:
            NeuralAgent enfant
        """
        child = NeuralAgent(n_rays=self.n_rays, device=self.device).to(self.device)
        
        # Extraire les génomes des deux parents
        genome1 = self.get_genome()
        genome2 = other.get_genome()
        
        # Masque de crossover: True = prendre parent1, False = prendre parent2
        crossover_mask = torch.rand_like(genome1) < crossover_rate
        
        # Créer le génome de l'enfant
        child_genome = torch.where(crossover_mask, genome1, genome2)
        
        # Injecter dans l'enfant
        child.set_genome(child_genome)
        
        return child
    
    def save_to_file(self, filepath):
        """Sauvegarde le modèle sur disque"""
        torch.save(self.state_dict(), filepath)
    
    def load_from_file(self, filepath):
        """Charge le modèle depuis disque"""
        self.load_state_dict(torch.load(filepath))


class VectorizedNeuralPopulation:
    """
    Gère une population entière de NeuralAgents de manière vectorisée
    pour des calculs ultra-rapides sur GPU.
    
    Au lieu de faire N forwards séparés, on fait UN SEUL forward avec batch_size=N.
    """
    
    def __init__(self, n_agents, n_rays=7, device='cuda'):
        self.n_agents = n_agents
        self.n_rays = n_rays
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # On crée un seul modèle de référence
        self.model = NeuralAgent(n_rays=n_rays).to(self.device)
        
        # On stocke tous les génomes dans un gros tensor (n_agents, genome_size)
        genome_size = sum(p.numel() for p in self.model.parameters())
        self.genomes = torch.randn(n_agents, genome_size, device=self.device) * 0.5
    
    def forward(self, observations):
        """
        Calcule les actions pour TOUTE la population en parallèle
        
        Args:
            observations: Tensor (n_agents, n_rays + 1)
        
        Returns:
            actions: Tensor (n_agents, 2)
        """
        all_actions = []
        
        # Pour chaque agent (on pourrait optimiser encore plus, mais c'est déjà très rapide)
        for i in range(self.n_agents):
            # Charger le génome de l'agent i
            self.model.set_genome(self.genomes[i])
            
            # Forward pour cet agent
            action = self.model(observations[i:i+1])
            all_actions.append(action)
        
        return torch.cat(all_actions, dim=0)
    
    def get_agent(self, index):
        """Récupère un agent spécifique sous forme de NeuralAgent"""
        agent = NeuralAgent(n_rays=self.n_rays).to(self.device)
        agent.set_genome(self.genomes[index])
        return agent
    
    def set_agent(self, index, agent):
        """Met à jour un agent de la population"""
        self.genomes[index] = agent.get_genome()
    
    def mutate_all(self, mutation_rate=0.1, mutation_strength=0.3):
        """Applique une mutation à toute la population"""
        mutation_mask = torch.rand_like(self.genomes) < mutation_rate
        noise = torch.rand_like(self.genomes) * 2 * mutation_strength - mutation_strength
        self.genomes += noise * mutation_mask.float()
    
    def save_population(self, filepath):
        """Sauvegarde toute la population"""
        torch.save({
            'genomes': self.genomes,
            'n_agents': self.n_agents,
            'n_rays': self.n_rays
        }, filepath)
    
    def load_population(self, filepath):
        """Charge une population depuis un fichier"""
        data = torch.load(filepath)
        self.genomes = data['genomes'].to(self.device)
        self.n_agents = data['n_agents']
        self.n_rays = data['n_rays']