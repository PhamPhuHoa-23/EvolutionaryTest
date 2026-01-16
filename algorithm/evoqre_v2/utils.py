"""
Utility functions for EvoQRE v2.

This module provides:
- soft_update: Polyak averaging for target networks
- hard_update: Direct copy for target networks
- Various helper functions
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Iterator, Tuple


def soft_update(
    target: nn.Module,
    source: nn.Module,
    tau: float = 0.005
) -> None:
    """
    Soft update target network parameters.
    
    θ_target = τ * θ_source + (1 - τ) * θ_target
    
    Args:
        target: Target network to update
        source: Source network
        tau: Interpolation parameter (default: 0.005)
    """
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(
            tau * source_param.data + (1.0 - tau) * target_param.data
        )


def hard_update(target: nn.Module, source: nn.Module) -> None:
    """
    Hard update (copy) target network parameters.
    
    Args:
        target: Target network to update
        source: Source network
    """
    target.load_state_dict(source.state_dict())


def get_optimizer(
    parameters: Iterator,
    optimizer_type: str = 'adam',
    lr: float = 1e-4,
    weight_decay: float = 0.0
) -> torch.optim.Optimizer:
    """
    Create optimizer for given parameters.
    
    Args:
        parameters: Model parameters
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd')
        lr: Learning rate
        weight_decay: Weight decay
        
    Returns:
        Optimizer instance
    """
    if optimizer_type.lower() == 'adam':
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'adamw':
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_type.lower() == 'sgd':
        return torch.optim.SGD(parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def subsample_opponents(
    opponent_actions: torch.Tensor,
    M_prime: int
) -> torch.Tensor:
    """
    Subsample opponent action particles for efficiency.
    
    Instead of averaging over all M opponent particles,
    randomly select M' << M for gradient estimation.
    
    Args:
        opponent_actions: Opponent actions (num_opponents, M, action_dim)
        M_prime: Number of samples to select
        
    Returns:
        Subsampled actions (num_opponents, M_prime, action_dim)
    """
    num_opponents, M, action_dim = opponent_actions.shape
    
    if M_prime >= M:
        return opponent_actions
    
    # Random indices for each opponent
    indices = torch.randint(0, M, (num_opponents, M_prime), device=opponent_actions.device)
    
    # Gather selected particles
    # Expand indices for gathering: (num_opponents, M_prime, action_dim)
    indices_exp = indices.unsqueeze(-1).expand(-1, -1, action_dim)
    subsampled = torch.gather(opponent_actions, dim=1, index=indices_exp)
    
    return subsampled


def compute_collision_rate(
    positions: torch.Tensor,
    threshold: float = 2.0
) -> float:
    """
    Compute pairwise collision rate.
    
    Args:
        positions: Agent positions (batch, num_agents, 2)
        threshold: Collision distance threshold
        
    Returns:
        Collision rate (fraction of scenarios with collision)
    """
    batch, num_agents, _ = positions.shape
    
    # Pairwise distances
    # (batch, num_agents, 1, 2) - (batch, 1, num_agents, 2)
    diff = positions.unsqueeze(2) - positions.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)  # (batch, num_agents, num_agents)
    
    # Mask diagonal (self-distance = 0)
    mask = torch.eye(num_agents, device=positions.device).unsqueeze(0)
    distances = distances + mask * 1e6  # Large value for self
    
    # Check collisions
    min_distances = distances.min(dim=-1)[0].min(dim=-1)[0]  # (batch,)
    collisions = (min_distances < threshold).float()
    
    return collisions.mean().item()


def compute_diversity(
    trajectories: torch.Tensor
) -> float:
    """
    Compute trajectory diversity as mean pairwise distance.
    
    Args:
        trajectories: Generated trajectories (batch, num_samples, time, 2)
        
    Returns:
        Mean pairwise trajectory distance
    """
    batch, num_samples, time_steps, dim = trajectories.shape
    
    # Flatten time dimension
    traj_flat = trajectories.reshape(batch, num_samples, -1)
    
    # Pairwise distances
    diff = traj_flat.unsqueeze(2) - traj_flat.unsqueeze(1)
    distances = torch.norm(diff, dim=-1)  # (batch, num_samples, num_samples)
    
    # Mean of upper triangle (exclude diagonal and duplicates)
    mask = torch.triu(torch.ones_like(distances[0]), diagonal=1)
    
    total_dist = (distances * mask).sum(dim=(1, 2))
    num_pairs = mask.sum()
    
    diversity = total_dist / num_pairs
    
    return diversity.mean().item()


def kde_log_likelihood(
    samples: torch.Tensor,
    targets: torch.Tensor,
    bandwidth: float = 1.0
) -> torch.Tensor:
    """
    Compute log-likelihood via Kernel Density Estimation.
    
    Uses Gaussian kernel with Silverman's rule for bandwidth.
    
    Args:
        samples: Generated samples (batch, num_samples, dim)
        targets: Target points (batch, dim)
        bandwidth: KDE bandwidth (or use Silverman's rule if None)
        
    Returns:
        Log-likelihood for each target
    """
    batch, num_samples, dim = samples.shape
    
    # Silverman's rule: h = 1.06 * σ * n^(-1/5)
    if bandwidth is None:
        std = samples.std(dim=1).mean()
        bandwidth = 1.06 * std * (num_samples ** (-0.2))
    
    # Expand targets: (batch, 1, dim)
    targets_exp = targets.unsqueeze(1)
    
    # Distances: (batch, num_samples)
    diff = samples - targets_exp
    sq_distances = (diff ** 2).sum(dim=-1)
    
    # Gaussian kernel
    log_kernel = -0.5 * sq_distances / (bandwidth ** 2)
    log_kernel = log_kernel - 0.5 * dim * np.log(2 * np.pi * bandwidth ** 2)
    
    # Log-sum-exp for log-likelihood
    log_density = torch.logsumexp(log_kernel, dim=1) - np.log(num_samples)
    
    return log_density


class ReplayBuffer:
    """
    Simple replay buffer for experience storage.
    """
    
    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        
    def push(self, *args):
        """Save a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = args
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        """Sample a batch of transitions."""
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        batch = [self.buffer[i] for i in indices]
        return tuple(zip(*batch))
    
    def __len__(self) -> int:
        return len(self.buffer)
