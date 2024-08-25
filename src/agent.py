from collections import defaultdict
from state import State
from deep_q_network import DeepQNetwork
import random
import numpy as np
import torch
import copy

GAMMA = 0.99
EPSILON = 1.0
EPSILON_DECAY = 1e-4

MEM_SIZE = 100000
BATCH_SIZE = 64
    # left, right
MOVES = [0, 1]

TARGET_UPDATE_INTERVAL = 1000

torch.autograd.set_detect_anomaly(True)

class Agent:
    def __init__(self):
        self.Q_func = DeepQNetwork(learning_rate=0.01, input_dims=6, fc1_dims=128, fc2_dims=128, n_actions=2)
        self.target_network = copy.deepcopy(self.Q_func)

        self.old_state_memory = np.zeros((MEM_SIZE, 6), dtype=np.float32)
        self.state_memory = np.zeros((MEM_SIZE, 6), dtype=np.float32)

        self.action_memory = np.zeros(MEM_SIZE, dtype=np.int32)
        self.reward_memory = np.zeros(MEM_SIZE, dtype=np.float32)

        self.terminal_states = np.zeros(MEM_SIZE, dtype=bool)

        self.memory_cursor = 0
        self.steps = 0

        self.epsilon = EPSILON

    def act(self, state: State):
        if np.random.random() < self.epsilon:  # explore moves according to epsilon greedy
            action = np.random.choice(MOVES)
        else:
            network_input = torch.Tensor([state.convert_to_vector()]).to(self.Q_func.device)
            action_values = self.Q_func.forward(network_input)
            action = torch.argmax(action_values).item()
            # print("network determined action: ", action, action_values.softmax(-1).tolist()[0])

        return action

    def learn(self):
        if self.memory_cursor < BATCH_SIZE:
            return  # Not enough samples to learn from yet

        # Randomly sample a batch of transitions
        max_mem = min(self.memory_cursor, MEM_SIZE)
        batch_indices = np.random.choice(max_mem, BATCH_SIZE, replace=False)

        batch_index = np.arange(BATCH_SIZE, dtype=np.int32)

        old_states = torch.tensor(self.old_state_memory[batch_indices]).to(self.Q_func.device)
        rewards = torch.tensor(self.reward_memory[batch_indices]).to(self.Q_func.device)
        states = torch.tensor(self.state_memory[batch_indices]).to(self.Q_func.device)
        terminal = torch.tensor(self.terminal_states[batch_indices]).to(self.Q_func.device)

        actions = self.action_memory[batch_indices]

        # Compute predicted Q-values
        q_eval = self.Q_func.forward(old_states)[batch_index, actions]

        # Compute target Q-values using the same Q-function
        q_next = self.target_network.forward(states).detach()
        q_next[terminal] = 0.0  # Terminal states have 0 future value
        q_target = rewards + GAMMA * torch.max(q_next, dim=1)[0]

        # Calculate the loss
        self.Q_func.optimizer.zero_grad()
        loss = self.Q_func.loss(q_target, q_eval).to(self.Q_func.device)

        # Backpropagation and update Q-network
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.Q_func.parameters(), max_norm=1.0)
        self.Q_func.optimizer.step()

        if self.epsilon > 0.05:
            self.epsilon = self.epsilon - EPSILON_DECAY
        self.steps += 1

        if self.steps > TARGET_UPDATE_INTERVAL:
            self.target_network.load_state_dict(self.Q_func.state_dict())

    def update(self, old_state: State, action: int, reward: np.float32, state: State, is_terminal: bool):
        i = self.memory_cursor % MEM_SIZE

        # print(self.old_state_memory, self.action_memory, self.reward_memory, self.state_memory)

        self.old_state_memory[i] = old_state.convert_to_vector()
        self.action_memory[i] = action
        self.reward_memory[i] = reward
        self.state_memory[i] = state.convert_to_vector()

        self.terminal_states[i] = is_terminal

        self.memory_cursor += 1


if __name__ == '__main__':
    a = Agent()

    print(a.q["B"])

