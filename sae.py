import torch
import torch.nn as nn



'''
TODO:
- top K
- 
'''
class SAE(nn.Module):
    def __init__(self, input_dim, hidden_dim):
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
    
    def forward(self, x):
        sparse_representation = self.encoder(x - self.decoder[0].bias)
        reconstruction = self.decoder(sparse_representation)
        return reconstruction, sparse_representation

'''
based on https://github.com/bartbussmann/BatchTopK/blob/main/sae.py#L74
it looks like topk can be done per gpu
'''

class BatchTopKSAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, k, alpha=0.999):
        super().__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
        )
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(hidden_dim, input_dim),
        )

        self.decoder[0].weight.data.copy_(self.encoder[0].weight.data.t())

        self.hidden_dim = hidden_dim
        self.k = k
        self.alpha = alpha
        self.min_activation_value = None
    
    def forward(self, x):
        hidden_state = self.encoder(x - self.decoder[0].bias) # BS, HD
        top_k_acts = torch.topk(hidden_state, self.k * x.shape[0])
        if self.training:
            self.update_min_act_value(torch.min(top_k_acts.values))
        sparse_representation = torch.zeros_like(hidden_state).scatter(-1, top_k_acts.indices, top_k_acts.values)
        reconstruction = self.decoder(sparse_representation)
        return reconstruction, sparse_representation
    
    def update_min_act_value(self, new_val):
        if self. min_activation_value is None:
            self.min_activation_value = new_val
        else:
            self.min_activation_value = self.min_activation_value * self.alpha + (1 - self.alpha) * new_val