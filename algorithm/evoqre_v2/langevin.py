"""
Langevin dynamics utilities for EvoQRE v2.

This module provides:
- langevin_sample: Sample actions via Langevin dynamics
- projected_langevin_step: Single step with projection to bounds

References:
- Algorithm 1: Particle-EvoQRE
- Corollary 4.4: Discrete-time convergence with O(√η) error
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Callable, Tuple


def projected_langevin_step(
    action: torch.Tensor,
    gradient: torch.Tensor,
    step_size: float,
    temperature: float,
    action_bound: float = 1.0,
    noise: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Single step of projected Langevin dynamics.
    
    Update rule (Euler-Maruyama discretization):
        a' = a + η∇Q(s,a) + √(2ητ)ξ
        a' = clip(a', -bound, bound)
    
    Args:
        action: Current action tensor (batch, action_dim)
        gradient: Gradient ∇_a Q(s, a)
        step_size: η (learning rate / step size)
        temperature: τ (noise scale)
        action_bound: Maximum action magnitude
        noise: Optional pre-generated noise (for reproducibility)
        
    Returns:
        Updated action tensor
    """
    if noise is None:
        noise = torch.randn_like(action)
    
    # Langevin update
    noise_scale = np.sqrt(2 * step_size * temperature)
    new_action = action + step_size * gradient + noise_scale * noise
    
    # Projection to action bounds (reflected boundary)
    new_action = torch.clamp(new_action, -action_bound, action_bound)
    
    return new_action


def langevin_sample(
    state: torch.Tensor,
    q_network: nn.Module,
    num_steps: int = 20,
    step_size: float = 0.1,
    temperature: float = 1.0,
    action_dim: int = 2,
    action_bound: float = 1.0,
    warm_start: Optional[torch.Tensor] = None,
    use_analytic_grad: bool = True
) -> torch.Tensor:
    """
    Sample actions via Langevin dynamics on Q-function landscape.
    
    Samples from the Gibbs distribution: π(a|s) ∝ exp(Q(s,a)/τ)
    
    Args:
        state: State tensor (batch, state_dim) or (state_dim,)
        q_network: Q-network with forward(state, action) method
        num_steps: Number of Langevin steps K
        step_size: Step size η
        temperature: Temperature τ
        action_dim: Action dimension
        action_bound: Action bound for projection
        warm_start: Optional initial action (for warm-starting)
        use_analytic_grad: Use analytic gradient if available
        
    Returns:
        Sampled action tensor
    """
    # Handle single state
    squeeze_output = False
    if state.dim() == 1:
        state = state.unsqueeze(0)
        squeeze_output = True
    
    batch_size = state.size(0)
    device = state.device
    
    # Initialize action particles
    if warm_start is not None:
        action = warm_start.clone()
    else:
        # Gaussian initialization centered at 0
        action = torch.randn(batch_size, action_dim, device=device) * 0.5
        action = torch.clamp(action, -action_bound, action_bound)
    
    # Langevin dynamics loop
    for step in range(num_steps):
        action.requires_grad_(True)
        
        # Compute gradient
        if use_analytic_grad and hasattr(q_network, 'get_action_gradient'):
            # Use analytic gradient (more efficient)
            with torch.no_grad():
                gradient = q_network.get_action_gradient(state, action.detach())
        else:
            # Use autograd
            if hasattr(q_network, 'min_q'):
                q_value = q_network.min_q(state, action)
            else:
                q_value = q_network(state, action)
            
            # Handle tuple output (dual Q)
            if isinstance(q_value, tuple):
                q_value = torch.min(q_value[0], q_value[1])
                
            gradient = torch.autograd.grad(
                outputs=q_value.sum(),
                inputs=action,
                create_graph=False,
                retain_graph=False
            )[0]
        
        # Langevin step
        action = projected_langevin_step(
            action=action.detach(),
            gradient=gradient,
            step_size=step_size,
            temperature=temperature,
            action_bound=action_bound
        )
    
    if squeeze_output:
        action = action.squeeze(0)
        
    return action.detach()


def langevin_sample_batch(
    states: torch.Tensor,
    q_network: nn.Module,
    num_particles: int = 50,
    num_steps: int = 20,
    step_size: float = 0.1,
    temperature: float = 1.0,
    action_dim: int = 2,
    action_bound: float = 1.0
) -> torch.Tensor:
    """
    Sample multiple particles per state.
    
    Args:
        states: State tensor (batch, state_dim)
        q_network: Q-network
        num_particles: Number of particles M per state
        num_steps: Langevin steps
        step_size: η
        temperature: τ
        action_dim: Action dimension
        action_bound: Action bound
        
    Returns:
        Actions tensor (batch, num_particles, action_dim)
    """
    batch_size = states.size(0)
    device = states.device
    
    # Expand states for particles: (batch, 1, state_dim) -> (batch, M, state_dim)
    states_exp = states.unsqueeze(1).expand(-1, num_particles, -1)
    states_flat = states_exp.reshape(-1, states.size(-1))
    
    # Sample actions
    actions_flat = langevin_sample(
        state=states_flat,
        q_network=q_network,
        num_steps=num_steps,
        step_size=step_size,
        temperature=temperature,
        action_dim=action_dim,
        action_bound=action_bound
    )
    
    # Reshape back
    actions = actions_flat.reshape(batch_size, num_particles, action_dim)
    
    return actions


def compute_soft_value(
    state: torch.Tensor,
    q_network: nn.Module,
    num_samples: int = 50,
    num_steps: int = 10,
    step_size: float = 0.1,
    temperature: float = 1.0,
    action_dim: int = 2,
    action_bound: float = 1.0
) -> torch.Tensor:
    """
    Estimate soft value V(s) = τ log ∫ exp(Q(s,a)/τ) da
    
    Uses Monte Carlo approximation with Langevin samples:
        V(s) ≈ τ log (1/M ∑_m exp(Q(s, a_m)/τ))
    
    Args:
        state: State tensor
        q_network: Q-network
        num_samples: Number of samples M for estimation
        num_steps: Langevin steps per sample
        step_size: η
        temperature: τ
        action_dim: Action dimension
        action_bound: Action bound
        
    Returns:
        Soft value tensor
    """
    squeeze_output = False
    if state.dim() == 1:
        state = state.unsqueeze(0)
        squeeze_output = True
        
    # Sample actions
    actions = langevin_sample_batch(
        states=state,
        q_network=q_network,
        num_particles=num_samples,
        num_steps=num_steps,
        step_size=step_size,
        temperature=temperature,
        action_dim=action_dim,
        action_bound=action_bound
    )  # (batch, M, action_dim)
    
    # Compute Q-values
    batch_size = state.size(0)
    state_exp = state.unsqueeze(1).expand(-1, num_samples, -1)
    
    # Flatten for Q evaluation
    state_flat = state_exp.reshape(-1, state.size(-1))
    action_flat = actions.reshape(-1, action_dim)
    
    with torch.no_grad():
        if hasattr(q_network, 'min_q'):
            q_values = q_network.min_q(state_flat, action_flat)
        else:
            q_values = q_network(state_flat, action_flat)
            if isinstance(q_values, tuple):
                q_values = torch.min(q_values[0], q_values[1])
    
    # Reshape: (batch, M, 1)
    q_values = q_values.reshape(batch_size, num_samples, -1)
    
    # Soft value: τ log (1/M ∑ exp(Q/τ))
    # Use log-sum-exp trick for numerical stability
    soft_value = temperature * torch.logsumexp(
        q_values / temperature, dim=1
    ) - temperature * np.log(num_samples)
    
    if squeeze_output:
        soft_value = soft_value.squeeze(0)
        
    return soft_value
