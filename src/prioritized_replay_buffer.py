import numpy as np
from state import State
import random

OFFSET = 0.005
ALPHA_INCREMENT = 1e-4

class PrioritizedReplayBuffer:
    def __init__(self, mem_size: int, alpha: int):
        """Alpha [0, 1]: 0 means pure random, 1 means pure greedy"""
        self.mem_size = mem_size
        self.alpha = alpha

        self.old_state_memory = np.zeros((self.mem_size, 6), dtype=np.float32)
        self.state_memory = np.zeros((self.mem_size, 6), dtype=np.float32)
        self.action_memory = np.zeros(self.mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.mem_size, dtype=np.float32)

        self.terminal_states = np.zeros(self.mem_size, dtype=bool)

        self.priorities = np.ones(self.mem_size, dtype=np.float32)
        self.probabilities = np.zeros(self.mem_size, dtype=np.float32)

        self.cursor = 0

    def __len__(self):
        return self.cursor

    def add(self, old_state: State, action: int, reward: np.float32, state: State, is_terminal: bool):
        i = self.cursor % self.mem_size

        self.old_state_memory[i] = old_state.convert_to_vector()
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.state_memory[i] = state.convert_to_vector()

        self.terminal_states[i] = is_terminal

        self.priorities[i] = max(self.priorities)  # Set initial priority to max

        self.cursor += 1

    def update_probs(self):
        current_size = min(self.cursor, self.mem_size)
        total_priorities = np.sum(self.priorities[:current_size] ** self.alpha)
        self.probabilities[:current_size] = (self.priorities[:current_size] ** self.alpha) / total_priorities

    def set_priorities(self, idxs, td_errors):
        self.priorities[idxs] = np.abs(td_errors) + OFFSET

    def get_importance(self, probabilities):
        importance = 1/self.cursor * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size: int):
        sample_size = min(self.cursor, batch_size)
        i = self.cursor % self.mem_size

        current_size = min(self.cursor, self.mem_size)

        self.update_probs()

        sample_probs = self.probabilities[:i]
        sample_indices = random.choices(range(current_size), k=sample_size, weights=sample_probs)

        samples = (self.old_state_memory[sample_indices], self.action_memory[sample_indices],
                   self.reward_memory[sample_indices], self.state_memory[sample_indices],
                   self.terminal_states[sample_indices])

        importance = self.get_importance(sample_probs[sample_indices])

        # alpha annealing to gradually shift to prioritized sampling
        self.alpha += ALPHA_INCREMENT
        return samples, sample_indices, importance

