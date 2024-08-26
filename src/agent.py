from collections import defaultdict
from state import State
from deep_q_network import DeepQNetwork
from prioritized_replay_buffer import PrioritizedReplayBuffer
import random
import numpy as np
import torch
import copy

GAMMA = 0.95
EPSILON = .8925
EPSILON_DECAY = 1e-5
TAU = 0.005

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

        self.memory = PrioritizedReplayBuffer(MEM_SIZE, alpha=0.6)

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
        if len(self.memory) < BATCH_SIZE:
            return  # Not enough samples to learn from yet

        # Randomly sample a batch of transitions
        batch, idxs, is_weights = self.memory.sample(BATCH_SIZE)
        old_states, actions, rewards, states, terminals = batch

        old_states = torch.tensor(old_states).to(self.Q_func.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.Q_func.device)
        rewards = torch.tensor(rewards).to(self.Q_func.device)
        states = torch.tensor(states).to(self.Q_func.device)
        terminals = torch.tensor(terminals).to(self.Q_func.device)
        importance_weights = torch.tensor(is_weights).to(self.Q_func.device)

        # Compute predicted Q-values
        q_eval = self.Q_func.forward(old_states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

        # Compute target Q-values using the same Q-function
        q_next = self.target_network.forward(states).detach()
        q_next[terminals] = 0.0  # Terminal states have 0 future value
        q_target = rewards + GAMMA * torch.max(q_next, dim=1)[0]

        # Calculate the loss
        self.Q_func.optimizer.zero_grad()
        loss = (importance_weights * self.Q_func.loss(q_target, q_eval).to(self.Q_func.device)).mean()

        # Update priorities
        td_errors = (q_target - q_eval).detach().cpu().numpy()
        self.memory.set_priorities(idxs, td_errors)

        # Backpropagation and update Q-network
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.Q_func.parameters(), max_norm=1.0)
        self.Q_func.optimizer.step()

        if self.epsilon > 0.05:
            self.epsilon = self.epsilon - EPSILON_DECAY
        self.steps += 1

        if self.steps > TARGET_UPDATE_INTERVAL:
            # Soft update of the target network
            for target_param, local_param in zip(self.target_network.parameters(), self.Q_func.parameters()):
                target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

    def update(self, old_state: State, action: int, reward: np.float32, state: State, is_terminal: bool):
        self.memory.add(old_state, action, reward, state, is_terminal)

    def save(self):
        torch.save(self.Q_func.state_dict(), "model.txt")

    def load(self):
        self.Q_func.load_state_dict(torch.load("model.txt"))


