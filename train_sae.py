import torch
import torch.nn as nn
import torch.optim as optim



class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, input_dim),
            torch.nn.ReLU()
        )
    
    def forward(self, x):
        return self.layers(x)
