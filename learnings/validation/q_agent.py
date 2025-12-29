import numpy as np
import random

class QAgent:
    def __init__(
        self,
        obs_space,
        action_space,
        alpha=0.1,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
        alpha_decay=0.999
    ):
        self.obs_space = obs_space
        self.action_space = action_space

        # Dimensions de la Q-table
        self.q_shape = tuple(obs_space.nvec) + (action_space.n,)
        self.Q = np.zeros(self.q_shape)

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha_decay = alpha_decay

    def select_action(self, state):
        """Politique epsilon-greedy"""
        if random.random() < self.epsilon:
            return self.action_space.sample()
        return np.argmax(self.Q[tuple(state)])

    def update(self, state, action, reward, next_state, terminated):
        s = tuple(state)
        s_next = tuple(next_state)

        q_sa = self.Q[s + (action,)]

        if terminated:
            target = reward
        else:
            target = reward + self.gamma * np.max(self.Q[s_next])

        self.Q[s + (action,)] += self.alpha * (target - q_sa)

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.alpha *= self.alpha_decay
