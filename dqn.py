import torch
from torch import nn
import torch.nn.functional as F
import flappy_bird_gymnasium
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_layer=256): 
        super(DQN, self).__init__()  

        self.fc1 = nn.Linear(state_dim, hidden_layer)             
        self.fc2 = nn.Linear(hidden_layer, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        y = self.fc2(x)
        return y
    
if __name__ == '__main__':
    state_dim = 12         # Number of input features
    action_dim = 2         # Number of possible actions
    
    net = DQN(state_dim, action_dim)  # Create the DQN instance

    state = torch.randn(1, state_dim) # Dummy input state
    output = net(state)               # Forward pass
    print(output)                     # Print Q-values
