import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedLayerFusion(nn.Module):
    def __init__(self, num_layers=25):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(num_layers))

    def forward(self, all_layers):
        normalized_weights = F.softmax(self.weights, dim=0)
        return torch.sum(all_layers * normalized_weights.view(1, -1, 1), dim=1)

class MERModel(nn.Module):
    def __init__(self, mode='hybrid', input_dim=1024):
        super().__init__()
        self.mode = mode
        if mode == 'hybrid':
            self.fusion = WeightedLayerFusion()
        
        self.regressor = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        if self.mode == 'hybrid':
            x = self.fusion(x)
        else:
            x = x[:, 24, :]
        
        # --- CRITICAL FIX START ---
        # We normalize the latent features so the dot-product doesn't explode
        latent = F.normalize(x, p=2, dim=1) 
        # --- CRITICAL FIX END ---
        
        output = self.regressor(x)
        return output, latent