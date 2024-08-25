import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DeepQNetwork(nn.Module):
    def __init__(self, learning_rate, input_dims, fc1_dims, fc2_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        # input and output dims for each layer of the network
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.n_actions = n_actions

        # create the layers of the network
        self.fc1 = nn.Linear(self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps")
        self.to(self.device)

    def forward(self, state):
        """forward propagation for the network"""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        actions = F.relu(self.fc3(x))

        return actions


