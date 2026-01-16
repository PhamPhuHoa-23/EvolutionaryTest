"""
Q-Network implementations for EvoQRE v2.

This module provides:
- ConcaveQNetwork: Q-network with guaranteed α-strong concavity
- SpectralNormEncoder: Encoder with Lipschitz bound via spectral normalization
- create_qnetwork: Factory function for creating Q-networks

References:
- Lemma 4.6: Concavity from Quadratic Head
- Lemma 4.7: Coupling Bound from Spectral Normalization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from typing import Optional, Tuple


def layer_init(layer: nn.Module, std: float = np.sqrt(2), bias_const: float = 0.0) -> nn.Module:
    """Initialize layer with orthogonal weights."""
    if hasattr(layer, 'weight'):
        torch.nn.init.orthogonal_(layer.weight, std)
    if hasattr(layer, 'bias') and layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class SpectralNormEncoder(nn.Module):
    """
    Encoder with spectral normalization on all linear layers.
    
    Guarantees ||f_θ||_Lip ≤ 1 (Lemma 4.7).
    
    The Lipschitz constant of a composition of layers is bounded by
    the product of individual spectral norms:
        ||f_θ||_Lip ≤ ∏_l ||W_l||_σ ≤ 1
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden layer dimension
        output_dim: Output feature dimension
        num_layers: Number of hidden layers (default: 2)
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2
    ):
        super().__init__()
        
        layers = []
        
        # Input layer
        layers.append(spectral_norm(nn.Linear(input_dim, hidden_dim)))
        layers.append(nn.ReLU())
        
        # Hidden layers
        for _ in range(num_layers - 1):
            layers.append(spectral_norm(nn.Linear(hidden_dim, hidden_dim)))
            layers.append(nn.ReLU())
        
        # Output layer
        layers.append(spectral_norm(nn.Linear(hidden_dim, output_dim)))
        
        self.network = nn.Sequential(*layers)
        self.output_dim = output_dim
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor of shape (batch, input_dim)
            
        Returns:
            Output tensor of shape (batch, output_dim)
        """
        return self.network(x)


class ConcaveQNetwork(nn.Module):
    """
    Q-network with guaranteed α-strong concavity in actions.
    
    Architecture:
        Q(s, a) = f_θ(s)ᵀa - ½aᵀ(LLᵀ + εI)a
    
    where:
        - f_θ: State encoder (optionally with spectral norm)
        - L: Learnable lower-triangular matrix
        - ε: Minimum eigenvalue guarantee (concavity strength)
    
    This guarantees:
        ∇²_aa Q = -(LLᵀ + εI) ⪯ -εI
    
    i.e., Q is ε-strongly concave in actions (Lemma 4.6).
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension
        epsilon: Concavity strength (minimum eigenvalue of -Hessian)
        use_spectral_norm: Whether to apply spectral norm to encoder
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        epsilon: float = 0.1,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.epsilon = epsilon
        
        # State encoder f_θ(s)
        if use_spectral_norm:
            self.encoder = SpectralNormEncoder(
                input_dim=state_dim,
                hidden_dim=hidden_dim,
                output_dim=action_dim,  # Output matches action dim for dot product
                num_layers=2
            )
        else:
            self.encoder = nn.Sequential(
                layer_init(nn.Linear(state_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, hidden_dim)),
                nn.ReLU(),
                layer_init(nn.Linear(hidden_dim, action_dim))
            )
        
        # Lower-triangular matrix L for quadratic term
        # LLᵀ is always positive semi-definite
        # Initialize L as identity for stable training start
        self.L = nn.Parameter(torch.eye(action_dim) * 0.1)
        
        # Register epsilon as buffer (not trainable)
        self.register_buffer('_epsilon', torch.tensor(epsilon))
        
    def forward(
        self, 
        state: torch.Tensor, 
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Q-value.
        
        Q(s, a) = f_θ(s)ᵀa - ½aᵀPa
        
        where P = LLᵀ + εI is the precision matrix.
        
        Args:
            state: State tensor of shape (batch, state_dim) or (state_dim,)
            action: Action tensor of shape (batch, action_dim) or (action_dim,)
            
        Returns:
            Q-value tensor of shape (batch, 1) or (1,)
        """
        # Handle unbatched input
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            squeeze_output = True
            
        # Linear term: f_θ(s)ᵀa
        f_s = self.encoder(state)  # (batch, action_dim)
        linear_term = (f_s * action).sum(dim=-1, keepdim=True)  # (batch, 1)
        
        # Compute precision matrix P = LLᵀ + εI
        # Use lower triangular part of L
        L_tril = torch.tril(self.L)
        P = L_tril @ L_tril.T + self._epsilon * torch.eye(
            self.action_dim, device=self.L.device
        )
        
        # Quadratic term: -½aᵀPa
        # Pa: (batch, action_dim)
        Pa = action @ P  # (batch, action_dim)
        quadratic_term = -0.5 * (action * Pa).sum(dim=-1, keepdim=True)  # (batch, 1)
        
        q_value = linear_term + quadratic_term
        
        if squeeze_output:
            q_value = q_value.squeeze(0)
            
        return q_value
    
    def get_action_gradient(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ∇_a Q(s, a) analytically.
        
        ∇_a Q = f_θ(s) - Pa
        
        This is more efficient than autograd for Langevin sampling.
        
        Args:
            state: State tensor
            action: Action tensor
            
        Returns:
            Gradient tensor of same shape as action
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)
            action = action.unsqueeze(0)
            
        f_s = self.encoder(state)
        
        L_tril = torch.tril(self.L)
        P = L_tril @ L_tril.T + self._epsilon * torch.eye(
            self.action_dim, device=self.L.device
        )
        
        grad = f_s - action @ P
        return grad
    
    def get_hessian(self) -> torch.Tensor:
        """
        Compute ∇²_aa Q analytically.
        
        ∇²_aa Q = -P = -(LLᵀ + εI)
        
        Returns:
            Hessian matrix of shape (action_dim, action_dim)
        """
        L_tril = torch.tril(self.L)
        P = L_tril @ L_tril.T + self._epsilon * torch.eye(
            self.action_dim, device=self.L.device
        )
        return -P
    
    def get_alpha(self) -> float:
        """
        Get strong concavity parameter α.
        
        α = min eigenvalue of -∇²_aa Q = min eigenvalue of P
        
        Since P = LLᵀ + εI and LLᵀ ⪰ 0, we have α ≥ ε.
        
        Returns:
            Alpha value (strong concavity constant)
        """
        with torch.no_grad():
            L_tril = torch.tril(self.L)
            P = L_tril @ L_tril.T + self._epsilon * torch.eye(
                self.action_dim, device=self.L.device
            )
            eigenvalues = torch.linalg.eigvalsh(P)
            return eigenvalues.min().item()


class DualQNetwork(nn.Module):
    """
    Twin Q-networks for double Q-learning (reduces overestimation).
    
    Uses two ConcaveQNetwork with shared encoder option.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension
        hidden_dim: Hidden layer dimension
        epsilon: Concavity strength
        use_spectral_norm: Whether to use spectral normalization
    """
    
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        epsilon: float = 0.1,
        use_spectral_norm: bool = True
    ):
        super().__init__()
        
        self.q1 = ConcaveQNetwork(
            state_dim, action_dim, hidden_dim, epsilon, use_spectral_norm
        )
        self.q2 = ConcaveQNetwork(
            state_dim, action_dim, hidden_dim, epsilon, use_spectral_norm
        )
        
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Q-values from both networks.
        
        Returns:
            Tuple of (q1_value, q2_value)
        """
        return self.q1(state, action), self.q2(state, action)
    
    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute min(Q1, Q2) for conservative value estimation.
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
    
    def get_alpha(self) -> float:
        """Get minimum alpha across both networks."""
        return min(self.q1.get_alpha(), self.q2.get_alpha())


def create_qnetwork(
    state_dim: int,
    action_dim: int,
    hidden_dim: int = 256,
    epsilon: float = 0.1,
    use_spectral_norm: bool = True,
    dual: bool = True
) -> nn.Module:
    """
    Factory function to create Q-network.
    
    Args:
        state_dim: State dimension
        action_dim: Action dimension  
        hidden_dim: Hidden layer dimension
        epsilon: Concavity strength (α ≥ ε)
        use_spectral_norm: Whether to bound Lipschitz constant
        dual: Whether to use twin Q-networks
        
    Returns:
        Q-network module
    """
    if dual:
        return DualQNetwork(
            state_dim, action_dim, hidden_dim, epsilon, use_spectral_norm
        )
    else:
        return ConcaveQNetwork(
            state_dim, action_dim, hidden_dim, epsilon, use_spectral_norm
        )
