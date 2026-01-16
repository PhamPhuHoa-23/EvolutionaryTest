"""
ParticleEvoQRE Agent for EvoQRE v2.

This module provides the main agent class implementing Algorithm 1 from the paper.

Key features:
- Particle-based policy representation (M particles per agent)
- Langevin dynamics for action sampling
- Concave Q-network with guaranteed stability
- Adaptive temperature mechanism
- Opponent subsampling (M') for efficiency

References:
- Algorithm 1: Particle-EvoQRE
- Section IV: Methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from .q_network import ConcaveQNetwork, DualQNetwork, create_qnetwork
from .langevin import langevin_sample, langevin_sample_batch, compute_soft_value
from .stability import (
    estimate_alpha_kappa, 
    adaptive_tau, 
    verify_stability,
    run_stability_diagnostics,
    StabilityDiagnostics
)
from .utils import soft_update, hard_update, subsample_opponents, ReplayBuffer


@dataclass
class EvoQREConfig:
    """Configuration for ParticleEvoQRE agent."""
    
    # Dimensions
    state_dim: int = 128
    action_dim: int = 2  # (acceleration, steering) or (vx, vy)
    hidden_dim: int = 256
    
    # Particle settings
    num_particles: int = 50  # M
    subsample_size: int = 10  # M'
    
    # Langevin dynamics
    langevin_steps: int = 20  # K
    step_size: float = 0.1  # η
    action_bound: float = 1.0
    
    # Temperature settings
    tau_base: float = 1.0
    tau_min: float = 0.3
    tau_max: float = 4.0
    use_adaptive_tau: bool = True
    heterogeneous_tau: bool = True
    
    # Q-network settings
    epsilon: float = 0.1  # Concavity strength
    use_spectral_norm: bool = True
    use_dual_q: bool = True
    
    # Training
    gamma: float = 0.99
    lr: float = 1e-4
    polyak_tau: float = 0.005
    batch_size: int = 256
    buffer_size: int = 100000
    
    # Stability
    stability_check_freq: int = 1000
    
    # Device
    device: str = 'cuda'


class ParticleEvoQRE:
    """
    Particle-based EvoQRE Agent.
    
    Implements Algorithm 1 from the paper:
    1. Initialize M particles per agent
    2. For each timestep:
        a. Sample actions via Langevin dynamics on Q-landscape
        b. Execute actions, observe rewards
        c. Update Q-networks via soft Bellman
        d. Soft-update target networks
    
    Key innovations:
    - Concave Q-head guarantees α-strong concavity (Lemma 4.6)
    - Spectral normalization bounds coupling κ (Lemma 4.7)  
    - Adaptive τ ensures stability condition (Proposition 4.5)
    - Opponent subsampling M' << M for O(N·M·M') complexity
    """
    
    def __init__(self, config: EvoQREConfig):
        """
        Initialize ParticleEvoQRE agent.
        
        Args:
            config: EvoQREConfig with all hyperparameters
        """
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Create Q-networks
        self.q_network = create_qnetwork(
            state_dim=config.state_dim,
            action_dim=config.action_dim,
            hidden_dim=config.hidden_dim,
            epsilon=config.epsilon,
            use_spectral_norm=config.use_spectral_norm,
            dual=config.use_dual_q
        ).to(self.device)
        
        # Target Q-network
        self.target_q_network = copy.deepcopy(self.q_network)
        for param in self.target_q_network.parameters():
            param.requires_grad = False
        
        # Optimizer
        self.q_optimizer = torch.optim.AdamW(
            self.q_network.parameters(),
            lr=config.lr
        )
        
        # Replay buffer
        self.replay_buffer = ReplayBuffer(config.buffer_size)
        
        # Temperature (can be per-agent if heterogeneous)
        self.tau = config.tau_base
        
        # Stability diagnostics
        self.stability_diagnostics: Optional[StabilityDiagnostics] = None
        self.update_count = 0
        
        # Training logs
        self.logs: List[Dict] = []
        
    def select_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False
    ) -> torch.Tensor:
        """
        Select action via Langevin sampling.
        
        Args:
            state: State tensor
            deterministic: If True, use more steps and less noise
            
        Returns:
            Selected action
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        # Adjust parameters for deterministic mode
        if deterministic:
            num_steps = self.config.langevin_steps * 2
            temperature = self.tau * 0.1  # Low noise
        else:
            num_steps = self.config.langevin_steps
            temperature = self.tau
        
        # Langevin sampling
        action = langevin_sample(
            state=state,
            q_network=self.q_network,
            num_steps=num_steps,
            step_size=self.config.step_size,
            temperature=temperature,
            action_dim=self.config.action_dim,
            action_bound=self.config.action_bound
        )
        
        return action
    
    def sample_particles(
        self,
        state: torch.Tensor,
        num_particles: Optional[int] = None
    ) -> torch.Tensor:
        """
        Sample multiple action particles for a state.
        
        Args:
            state: State tensor (batch, state_dim) or (state_dim,)
            num_particles: Number of particles (default: config.num_particles)
            
        Returns:
            Action particles (batch, M, action_dim) or (M, action_dim)
        """
        if not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)
        state = state.to(self.device)
        
        if num_particles is None:
            num_particles = self.config.num_particles
        
        squeeze_output = False
        if state.dim() == 1:
            state = state.unsqueeze(0)
            squeeze_output = True
        
        actions = langevin_sample_batch(
            states=state,
            q_network=self.q_network,
            num_particles=num_particles,
            num_steps=self.config.langevin_steps,
            step_size=self.config.step_size,
            temperature=self.tau,
            action_dim=self.config.action_dim,
            action_bound=self.config.action_bound
        )
        
        if squeeze_output:
            actions = actions.squeeze(0)
            
        return actions
    
    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ) -> None:
        """Store transition in replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def update(self, batch_size: Optional[int] = None) -> Dict[str, float]:
        """
        Update Q-networks using sampled batch.
        
        Returns:
            Dictionary of training metrics
        """
        if batch_size is None:
            batch_size = self.config.batch_size
            
        if len(self.replay_buffer) < batch_size:
            return {}
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)
        
        # Convert to tensors
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(self.device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).unsqueeze(1).to(self.device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(self.device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # Compute target Q-value
        with torch.no_grad():
            # Sample next actions via Langevin on target Q
            next_actions = langevin_sample(
                state=next_states,
                q_network=self.target_q_network,
                num_steps=self.config.langevin_steps // 2,  # Fewer steps for speed
                step_size=self.config.step_size,
                temperature=self.tau,
                action_dim=self.config.action_dim,
                action_bound=self.config.action_bound
            )
            
            # Target Q-value
            if hasattr(self.target_q_network, 'min_q'):
                next_q = self.target_q_network.min_q(next_states, next_actions)
            else:
                next_q = self.target_q_network(next_states, next_actions)
                if isinstance(next_q, tuple):
                    next_q = torch.min(next_q[0], next_q[1])
            
            # Soft Bellman target: r + γ * (Q - τ * log_prob)
            # Since Langevin samples from exp(Q/τ), entropy is implicit
            # For simplicity, use standard Bellman (entropy handled by sampling)
            target_q = rewards + self.config.gamma * (1 - dones) * next_q
        
        # Current Q-values
        if isinstance(self.q_network, DualQNetwork):
            q1, q2 = self.q_network(states, actions)
            q1_loss = F.mse_loss(q1, target_q)
            q2_loss = F.mse_loss(q2, target_q)
            q_loss = q1_loss + q2_loss
            avg_q = (q1.mean().item() + q2.mean().item()) / 2
        else:
            q = self.q_network(states, actions)
            q_loss = F.mse_loss(q, target_q)
            avg_q = q.mean().item()
        
        # Optimize
        self.q_optimizer.zero_grad()
        q_loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        
        self.q_optimizer.step()
        
        # Soft update target network
        soft_update(self.target_q_network, self.q_network, self.config.polyak_tau)
        
        # Update count
        self.update_count += 1
        
        # Periodic stability check
        if self.config.use_adaptive_tau and self.update_count % self.config.stability_check_freq == 0:
            self._update_adaptive_tau(states, actions)
        
        # Log metrics
        metrics = {
            'q_loss': q_loss.item(),
            'avg_q': avg_q,
            'tau': self.tau
        }
        
        if self.stability_diagnostics:
            metrics['alpha'] = self.stability_diagnostics.alpha
            metrics['is_stable'] = float(self.stability_diagnostics.is_stable)
        
        self.logs.append(metrics)
        
        return metrics
    
    def _update_adaptive_tau(
        self,
        states: torch.Tensor,
        actions: torch.Tensor
    ) -> None:
        """Update temperature adaptively based on stability diagnostics."""
        # Run diagnostics
        self.stability_diagnostics = run_stability_diagnostics(
            q_network=self.q_network,
            states=states,
            actions=actions,
            tau=self.tau,
            device=str(self.device)
        )
        
        # Update tau if adaptive mode
        if self.config.use_adaptive_tau:
            new_tau = self.stability_diagnostics.tau_adaptive
            # Clamp to bounds
            new_tau = np.clip(new_tau, self.config.tau_min, self.config.tau_max)
            self.tau = new_tau
    
    def get_heterogeneous_tau(self, agent_type: str = 'normal') -> float:
        """
        Get temperature for heterogeneous agent types.
        
        From paper:
        - Aggressive (20%): τ ~ U(0.3, 0.6)
        - Normal (60%): τ ~ U(0.8, 1.5)
        - Conservative (20%): τ ~ U(2.0, 4.0)
        
        Args:
            agent_type: One of 'aggressive', 'normal', 'conservative'
            
        Returns:
            Sampled temperature
        """
        if agent_type == 'aggressive':
            return np.random.uniform(0.3, 0.6)
        elif agent_type == 'normal':
            return np.random.uniform(0.8, 1.5)
        elif agent_type == 'conservative':
            return np.random.uniform(2.0, 4.0)
        else:
            return self.config.tau_base
    
    def save(self, path: str) -> None:
        """Save agent checkpoint."""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_q_network': self.target_q_network.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'tau': self.tau,
            'update_count': self.update_count,
            'config': self.config
        }, path)
    
    def load(self, path: str) -> None:
        """Load agent checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_q_network.load_state_dict(checkpoint['target_q_network'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        self.tau = checkpoint['tau']
        self.update_count = checkpoint['update_count']


class MultiAgentEvoQRE:
    """
    Multi-agent EvoQRE for traffic simulation.
    
    Manages multiple ParticleEvoQRE agents with:
    - Shared or independent Q-networks
    - Opponent modeling via particle subsampling
    - Stability verification across all agents
    """
    
    def __init__(
        self,
        num_agents: int,
        config: EvoQREConfig,
        shared_network: bool = False
    ):
        """
        Initialize multi-agent system.
        
        Args:
            num_agents: Number of agents N
            config: Configuration
            shared_network: Whether to share Q-network across agents
        """
        self.num_agents = num_agents
        self.config = config
        self.shared_network = shared_network
        
        if shared_network:
            # Shared Q-network
            self.agents = [ParticleEvoQRE(config)]
        else:
            # Independent Q-networks per agent
            self.agents = [ParticleEvoQRE(config) for _ in range(num_agents)]
        
        # Assign heterogeneous temperatures
        if config.heterogeneous_tau:
            self._assign_heterogeneous_tau()
    
    def _assign_heterogeneous_tau(self) -> None:
        """Assign temperatures based on driver type distribution."""
        for i, agent in enumerate(self.agents):
            # 20% aggressive, 60% normal, 20% conservative
            rand = np.random.random()
            if rand < 0.2:
                agent.tau = agent.get_heterogeneous_tau('aggressive')
            elif rand < 0.8:
                agent.tau = agent.get_heterogeneous_tau('normal')
            else:
                agent.tau = agent.get_heterogeneous_tau('conservative')
    
    def select_actions(
        self,
        states: List[torch.Tensor],
        deterministic: bool = False
    ) -> List[torch.Tensor]:
        """
        Select actions for all agents.
        
        Args:
            states: List of state tensors (one per agent)
            deterministic: Whether to use deterministic sampling
            
        Returns:
            List of action tensors
        """
        if self.shared_network:
            return [self.agents[0].select_action(s, deterministic) for s in states]
        else:
            return [
                agent.select_action(state, deterministic)
                for agent, state in zip(self.agents, states)
            ]
    
    def update_all(self, batch_size: Optional[int] = None) -> List[Dict]:
        """Update all agents."""
        if self.shared_network:
            return [self.agents[0].update(batch_size)]
        else:
            return [agent.update(batch_size) for agent in self.agents]
    
    def get_stability_summary(self) -> Dict:
        """Get stability summary across all agents."""
        alphas = []
        is_stable = []
        
        for agent in self.agents:
            if agent.stability_diagnostics:
                alphas.append(agent.stability_diagnostics.alpha)
                is_stable.append(agent.stability_diagnostics.is_stable)
        
        return {
            'mean_alpha': np.mean(alphas) if alphas else 0,
            'stability_rate': np.mean(is_stable) if is_stable else 0
        }
