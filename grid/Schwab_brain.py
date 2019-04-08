import numpy as np
from collections import deque
import copy

# # --Machine Learning--
import torch
from torch import autograd
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# # -Optimizer
import torch.optim as optim



class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__() # Initialize superclass

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Layers
        self.ff1 = nn.Linear(input_dim, hidden_dim)
#        self.ff2 = nn.Linear(hidden_dim, hidden_dim)
        self.ff3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.ff1(x)) # Input -> 1st hidden
#        x = F.relu(self.ff2(x)) # 1st hidden -> 2nd hidden
        x = F.softmax(self.ff3(x), dim=0)         # 2nd hidden -> output, deprecation along dim=0 (there should only be one)
        return x



class Agent:
    def __init__(self, id, loc, in_dim, hid_dim = 10, out_dim = 4, lr = 0.01, glee = 1, baseline = 0):
        self.id = id    # Agent id (in agent_dict)
        self.loc = loc  # Location (r,c coordinate [r,c])

        self.has_key = False
	
        self.reward = 0     # Keep track of reward
        self.glee = glee    # Number of points gained from opening A door
        self.baseline = baseline

        # Create brain
        self.PolNet = Net(in_dim, hid_dim, out_dim)        # Personal neural network
        self.optimizer = optim.Adam(self.PolNet.parameters())

    # State formation
    def observe(self, map):
        state = copy.deepcopy(map)
        state[self.loc[0]][self.loc[1]] = 0            # Own location = 0 on map
        state = np.reshape(state, [1,-1]).squeeze()    # Convert to 1D array
        return state

    # Policy Update
    # This function can go outside the class
    def REINFORCE(self, probability, reward, baseline):
        rl = -torch.log(probability) * (reward-baseline)
        return rl
