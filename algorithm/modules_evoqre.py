import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ResidualBlock(nn.Module):
    """Residual block for deeper networks."""
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.fc = nn.Sequential(
            layer_init(nn.Linear(dim, dim)),
            nn.LayerNorm(dim),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(dim, dim)),
            nn.LayerNorm(dim),
        )
        
    def forward(self, x):
        return F.relu(x + self.fc(x))

class MixturePolicy(nn.Module):
    """
    Mixture of Gaussians Policy Network for EvoQRE.
    
    Output:
    - weights: Mixing probabilities (M)
    - means: Means of each component (M x action_dim)
    - scales: Standard deviations of each component (M x action_dim)
    """
    def __init__(self, state_dim, hidden_dim, action_dim=2, num_components=10):
        super(MixturePolicy, self).__init__()
        self.num_components = num_components
        self.action_dim = action_dim
        
        # Input projection
        self.input_layer = nn.Sequential(
            layer_init(nn.Linear(state_dim, hidden_dim)),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Residual blocks
        self.res_block1 = ResidualBlock(hidden_dim)
        self.res_block2 = ResidualBlock(hidden_dim)
        
        # Heads
        # 1. Mixture weights (logits)
        self.weight_head = nn.Sequential(
            layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.ReLU(inplace=True),
            layer_init(nn.Linear(hidden_dim // 2, num_components), 0.01)
        )
        
        # 2. Means (M * action_dim)
        self.mean_head = nn.Sequential(
             layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
             nn.ReLU(inplace=True),
             layer_init(nn.Linear(hidden_dim // 2, num_components * action_dim), 0.01)
        )
        
        # 3. Scales (M * action_dim) - output log_scales or raw scales
        self.scale_head = nn.Sequential(
             layer_init(nn.Linear(hidden_dim, hidden_dim // 2)),
             nn.ReLU(inplace=True),
             layer_init(nn.Linear(hidden_dim // 2, num_components * action_dim), 0.01)
        )

    def forward(self, state):
        x = self.input_layer(state)
        x = self.res_block1(x)
        x = self.res_block2(x)
        
        # Weights
        logits = self.weight_head(x)
        weights = F.softmax(logits, dim=-1) # (batch, M)
        
        # Means
        means = self.mean_head(x)
        means = means.view(-1, self.num_components, self.action_dim)
        
        # Scales
        raw_scales = self.scale_head(x)
        raw_scales = raw_scales.view(-1, self.num_components, self.action_dim)
        scales = F.softplus(raw_scales) + 1e-4 # Ensure positive
        
        return weights, means, scales
