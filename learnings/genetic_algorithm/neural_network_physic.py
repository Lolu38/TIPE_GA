"""
neural_network_v2.py — Réseau de neurones avec physique réaliste
================================================================

Changements par rapport à neural_network.py :

  Entrées : n_rays + 1  ->  n_rays + 1 + n_system_obs
    Nouvelles observations systèmes (par défaut 4) :
      - usure pneu      [0, 1]
      - niveau fuel     [0, 1]
      - intensité pluie [0, 1]
      - composé actuel  [0, 1]  (normalisé : indice / N_COMPOUNDS)

  Sorties : 2  ->  4
    - steering  : tanh  -> [-1, 1]   (inchangé)
    - throttle  : tanh  -> [-1, 1]   (sigmoid -> tanh pour autoriser le freinage)
    - pit_tire  : sigmoid -> [0, 1]  (> 0.5 + in_pit_zone -> pit pneus seulement)
    - pit_full  : sigmoid -> [0, 1]  (> 0.5 + in_pit_zone -> pit complet)

  Taille du génome (n_rays=9, n_system_obs=4, h1=32, h2=16) :
    W1 : 32 × 14 = 448   b1 : 32
    W2 : 16 × 32 = 512   b2 : 16
    W3 :  4 × 16 = 64    b3 :  4
    Total = 1076

  Note sur les hidden sizes :
    h1 et h2 ont été augmentés (16->32, 8->16) pour compenser
    l'augmentation du nombre d'entrées et de sorties.
    Si les temps d'entraînement sont trop longs, on peut revenir à 16/8.

  Compatibilité :
    pop_manager_v2.py doit importer depuis neural_network_v2
    Les checkpoints de neural_network.py sont INCOMPATIBLES (taille génome différente)

Usage :
    from learnings.genetic_algorithm.neural_network_v2 import (
        NeuralAgent, VectorizedNeuralPopulation
    )
"""

import torch
import torch.nn as nn

# Nombre d'observations système ajoutées aux rayons + vitesse
# (wear, fuel, rain, compound) -> 4
N_SYSTEM_OBS = 4

# Nombre d'actions de sortie
# (steering, throttle, pit_tire, pit_full) -> 4
N_ACTIONS = 4


class NeuralAgent(nn.Module):
    """
    Réseau MLP pour contrôler une voiture avec gestion des systèmes.

    Architecture :
        Input  : n_rays + 1 (vitesse) + n_system_obs
        Hidden1: h1 neurones (ReLU)
        Hidden2: h2 neurones (ReLU)
        Output : 4 neurones
          [0] steering : tanh  -> [-1, 1]
          [1] throttle : tanh  -> [-1, 1]  (négatif = freinage)
          [2] pit_tire : sigmoid -> [0, 1]
          [3] pit_full : sigmoid -> [0, 1]
    """

    def __init__(
        self,
        n_rays        : int = 9,
        n_system_obs  : int = N_SYSTEM_OBS,
        hidden_size1  : int = 32,
        hidden_size2  : int = 16,
        device        : str = 'cuda'
    ):
        super(NeuralAgent, self).__init__()

        self.n_rays       = n_rays
        self.n_system_obs = n_system_obs
        self.device       = device

        input_size = n_rays + 1 + n_system_obs   # rayons + vitesse + systèmes

        self.network = nn.Sequential(
            nn.Linear(input_size,   hidden_size1),
            nn.ReLU(),
            nn.Linear(hidden_size1, hidden_size2),
            nn.ReLU(),
            nn.Linear(hidden_size2, N_ACTIONS),   # 4 sorties
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Propagation avant.

        Args:
            observations : Tensor (batch, n_rays + 1 + n_system_obs)
              [:, :n_rays]          distances lidar normalisées [0, 1]
              [:, n_rays]           vitesse normalisée          [0, 1]
              [:, n_rays+1]         usure pneu                  [0, 1]
              [:, n_rays+2]         niveau fuel                 [0, 1]
              [:, n_rays+3]         intensité pluie             [0, 1]
              [:, n_rays+4]         composé normalisé           [0, 1]

        Returns:
            actions : Tensor (batch, 4)
              [:, 0] steering  ∈ [-1, 1]
              [:, 1] throttle  ∈ [-1, 1]
              [:, 2] pit_tire  ∈ [0, 1]
              [:, 3] pit_full  ∈ [0, 1]
        """
        raw = self.network(observations)                     # (batch, 4)

        steering = torch.tanh(raw[:, 0:1])                  # [-1, 1]
        throttle = torch.tanh(raw[:, 1:2])                  # [-1, 1]  ← freinage possible
        pit_tire = torch.sigmoid(raw[:, 2:3])               # [0, 1]
        pit_full = torch.sigmoid(raw[:, 3:4])               # [0, 1]

        return torch.cat([steering, throttle, pit_tire, pit_full], dim=1)  # (batch, 4)

    # --------------------------------------------------------------------------
    # Interface génome — identique à l'ancienne
    def get_genome(self) -> torch.Tensor:
        """Extrait tous les poids sous forme de vecteur 1D."""
        return torch.cat([p.data.view(-1) for p in self.parameters()])

    def set_genome(self, genome: torch.Tensor):
        """Injecte un génome (vecteur 1D) dans le réseau."""
        offset = 0
        for param in self.parameters():
            numel = param.numel()
            param.data = genome[offset:offset + numel].view(param.shape)
            offset += numel

    def mutate(self, mutation_rate: float = 0.3, mutation_strength: float = 0.1):
        """Mutation aléatoire uniforme des poids."""
        with torch.no_grad():
            for param in self.parameters():
                mask  = torch.rand_like(param) < mutation_rate
                noise = torch.rand_like(param) * 2 * mutation_strength - mutation_strength
                param.data += noise * mask.float()

    def clone(self) -> 'NeuralAgent':
        c = NeuralAgent(
            n_rays=self.n_rays, n_system_obs=self.n_system_obs, device=self.device
        ).to(self.device)
        c.load_state_dict(self.state_dict())
        return c

    def crossover(self, other: 'NeuralAgent', crossover_rate: float = 0.5) -> 'NeuralAgent':
        child = NeuralAgent(
            n_rays=self.n_rays, n_system_obs=self.n_system_obs, device=self.device
        ).to(self.device)
        g1   = self.get_genome()
        g2   = other.get_genome()
        mask = torch.rand_like(g1) < crossover_rate
        child.set_genome(torch.where(mask, g1, g2))
        return child

    def save_to_file(self, filepath: str):
        torch.save(self.state_dict(), filepath)

    def load_from_file(self, filepath: str):
        self.load_state_dict(torch.load(filepath))


class VectorizedNeuralPopulation:
    """
    Population entière gérée en un seul forward GPU via bmm.

    Identique à v1 sauf :
      - input_size  = n_rays + 1 + n_system_obs
      - output_size = 4  (au lieu de 2)
      - h1=32, h2=16     (au lieu de 16, 8)
      - throttle -> tanh,  pit_tire / pit_full -> sigmoid
    """

    def __init__(
        self,
        n_agents     : int = 1000,
        n_rays       : int = 9,
        n_system_obs : int = N_SYSTEM_OBS,
        hidden_size1 : int = 32,
        hidden_size2 : int = 16,
        device       : str = 'cuda'
    ):
        self.n_agents     = n_agents
        self.n_rays       = n_rays
        self.n_system_obs = n_system_obs
        self.h1           = hidden_size1
        self.h2           = hidden_size2
        self.device       = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Taille du génome calculée depuis un modèle de référence
        ref = NeuralAgent(n_rays, n_system_obs, hidden_size1, hidden_size2).to(self.device)
        self.genome_size = sum(p.numel() for p in ref.parameters())
        self.input_size  = n_rays + 1 + n_system_obs

        # Génomes initiaux (distribution gaussienne centrée, std=0.5)
        self.genomes = torch.randn(
            n_agents, self.genome_size, device=self.device
        ) * 0.5

    def forward_vectorized(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass pour toute la population en une seule passe GPU.

        Args:
            observations : Tensor (N, input_size)

        Returns:
            actions : Tensor (N, 4)
              [:, 0] steering  ∈ [-1, 1]
              [:, 1] throttle  ∈ [-1, 1]
              [:, 2] pit_tire  ∈ [0, 1]
              [:, 3] pit_full  ∈ [0, 1]

        Dimensions des tenseurs de poids (bmm = batch matrix multiply) :
            W1 (N, h1, input) × x (N, input, 1) -> (N, h1, 1)
            W2 (N, h2, h1)    × x (N, h1,    1) -> (N, h2, 1)
            W3 (N,  4, h2)    × x (N, h2,    1) -> (N,  4, 1)
        """
        N  = self.n_agents
        i  = self.input_size
        h1 = self.h1
        h2 = self.h2

        # -- Découpage des génomes en matrices de poids ------------------------
        offset = 0

        W1 = self.genomes[:, offset:offset + h1*i ].view(N, h1, i );  offset += h1 * i
        b1 = self.genomes[:, offset:offset + h1   ].view(N, h1, 1 );  offset += h1
        W2 = self.genomes[:, offset:offset + h2*h1].view(N, h2, h1);  offset += h2 * h1
        b2 = self.genomes[:, offset:offset + h2   ].view(N, h2, 1 );  offset += h2
        W3 = self.genomes[:, offset:offset + 4*h2 ].view(N,  4, h2);  offset += 4  * h2
        b3 = self.genomes[:, offset:offset + 4    ].view(N,  4, 1 )

        # -- Forward pass vectorisé --------------------------------------------
        x = observations.unsqueeze(-1)            # (N, input, 1)
        x = torch.relu(torch.bmm(W1, x) + b1)    # (N, h1, 1)
        x = torch.relu(torch.bmm(W2, x) + b2)    # (N, h2, 1)
        x = torch.bmm(W3, x) + b3                # (N,  4, 1)
        x = x.squeeze(-1)                         # (N,  4)

        # -- Activations par sortie --------------------------------------------
        steering = torch.tanh(x[:, 0:1])          # (N, 1)  [-1, 1]
        throttle = torch.tanh(x[:, 1:2])          # (N, 1)  [-1, 1]
        pit_tire = torch.sigmoid(x[:, 2:3])       # (N, 1)  [0, 1]
        pit_full = torch.sigmoid(x[:, 3:4])       # (N, 1)  [0, 1]

        return torch.cat([steering, throttle, pit_tire, pit_full], dim=1)  # (N, 4)

    # Interface population — identique à v1
    def get_agent(self, index: int) -> NeuralAgent:
        agent = NeuralAgent(
            self.n_rays, self.n_system_obs, self.h1, self.h2
        ).to(self.device)
        agent.set_genome(self.genomes[index])
        return agent

    def set_agent(self, index: int, agent: NeuralAgent):
        self.genomes[index] = agent.get_genome()

    def mutate_all(self, mutation_rate: float = 0.1, mutation_strength: float = 0.3):
        mask  = torch.rand_like(self.genomes) < mutation_rate
        noise = (torch.rand_like(self.genomes) * 2.0 - 1.0) * mutation_strength
        self.genomes += noise * mask.float()

    def save_population(self, filepath: str):
        torch.save({
            'genomes'     : self.genomes,
            'n_agents'    : self.n_agents,
            'n_rays'      : self.n_rays,
            'n_system_obs': self.n_system_obs,
            'h1'          : self.h1,
            'h2'          : self.h2,
        }, filepath)

    def load_population(self, filepath: str):
        data = torch.load(filepath, map_location=self.device)
        self.genomes      = data['genomes'].to(self.device)
        self.n_agents     = data['n_agents']
        self.n_rays       = data['n_rays']
        self.n_system_obs = data.get('n_system_obs', N_SYSTEM_OBS)
        self.h1           = data.get('h1', 32)
        self.h2           = data.get('h2', 16)