from collections import defaultdict
from state import State
from deep_q_network import DeepQNetwork
import random
import numpy as np
import torch

GAMMA = 0.9
ALPHA = 0.01
EPSILON = 1.0
EPSILON_DECAY = 0.05

LAMBDA = 0.9

INITIAL_ACTION_VALUE = 3.0
MEM_SIZE = 10000
BATCH_SIZE = 64
    # left, right
MOVES = [1, 2]


class Agent:
    def __init__(self):
        self.Q_func = DeepQNetwork(learning_rate=0.01, input_dims=6, fc1_dims=128, fc2_dims=128, n_actions=2)

        self.old_state_memory = np.zeros((MEM_SIZE, 6), dtype=np.float32)
        self.state_memory = np.zeros((MEM_SIZE, 6), dtype=np.float32)

        self.action_memory = np.zeros(MEM_SIZE, dtype=np.int32)
        self.reward_memory = np.zeros(MEM_SIZE, dtype=np.float32)

        self.terminal_states = np.zeros(MEM_SIZE, dtype=bool)

        self.memory_cursor = 0

        self.epsilon = EPSILON

    def act(self, state: State):
        if np.random.random() < self.epsilon:  # explore moves according to epsilon greedy
            action = np.random.choice(MOVES)
        else:
            network_input = torch.Tensor([state.convert_to_vector()]).to(self.Q_func.device)
            action_values = self.Q_func.forward(network_input)
            action = torch.argmax(action_values).item()

        return action

    def learn(self):
        if self.memory_cursor < BATCH_SIZE:
            return  # Not enough samples to learn from yet

        # Randomly sample a batch of transitions
        max_mem = min(self.memory_cursor, MEM_SIZE)
        batch_indices = np.random.choice(max_mem, BATCH_SIZE, replace=False)

        old_states = torch.tensor(self.old_state_memory[batch_indices]).to(self.Q_func.device)
        actions = torch.tensor(self.action_memory[batch_indices]).to(self.Q_func.device)
        rewards = torch.tensor(self.reward_memory[batch_indices]).to(self.Q_func.device)
        states = torch.tensor(self.state_memory[batch_indices]).to(self.Q_func.device)
        terminal = torch.tensor(self.terminal_states[batch_indices]).to(self.Q_func.device)

        # Compute predicted Q-values
        q_eval = self.Q_func.forward(old_states).gather(1, actions.unsqueeze(-1).long()).squeeze(-1)

        # Compute target Q-values using the same Q-function
        q_next = self.Q_func.forward(states).max(dim=1)[0]
        q_next[terminal] = 0.0  # Terminal states have 0 future value
        q_target = rewards + GAMMA * q_next

        # Calculate the loss
        loss = self.Q_func.loss(q_eval, q_target.detach())

        # Backpropagation and update Q-network
        self.Q_func.optimizer.zero_grad()
        loss.backward()
        self.Q_func.optimizer.step()

        self.epsilon = self.epsilon - EPSILON_DECAY

    def update(self, old_state: State, action: int, reward: np.float32, state: State, is_terminal: bool):
        i = self.memory_cursor % MEM_SIZE

        self.old_state_memory[i] = old_state.convert_to_vector()
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.state_memory[i] = state.convert_to_vector()

        self.terminal_states[i] = is_terminal

        self.memory_cursor += 1


if __name__ == '__main__':
    a = Agent()

    print(a.q["B"])

