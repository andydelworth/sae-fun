import torch
import torch.nn as nn



'''
TODO:
- top K
- 
'''
class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, l1_lambda=1):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU()
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim),
        )

        self.decoder[0].weight.data.copy_(self.encoder[0].weight.data.t())

        self.hidden_dim = hidden_dim
        self.l1_lambda = l1_lambda
    
    def forward(self, x):
        sparse_representation = self.encoder(x - self.decoder[0].bias)
        reconstruction = self.decoder(sparse_representation)
        return reconstruction, sparse_representation