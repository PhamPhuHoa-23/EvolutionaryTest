# ---
# EvoQRE v2: Particle-based Quantal Response Equilibrium
# 
# This module implements the methodology from:
# "EvoQRE: Particle-Based Langevin Sampling for Multi-Agent
#  Quantal Response Equilibrium in Traffic Simulation"
#
# Key features:
# - Concave Q-head for α-strong concavity (Lemma 4.6)
# - Spectral normalization for Lipschitz bound (Lemma 4.7)
# - Adaptive temperature for universal stability (Proposition 4.5)
# - Particle-based Langevin sampling (Algorithm 1)
# ---

from .q_network import ConcaveQNetwork, SpectralNormEncoder, DualQNetwork, create_qnetwork
from .agent import ParticleEvoQRE, MultiAgentEvoQRE, EvoQREConfig
from .stability import estimate_alpha_kappa, adaptive_tau, verify_stability, run_stability_diagnostics
from .langevin import langevin_sample, langevin_sample_batch, projected_langevin_step, compute_soft_value
from .utils import soft_update, hard_update, subsample_opponents, ReplayBuffer

__all__ = [
    # Q-networks
    'ConcaveQNetwork',
    'SpectralNormEncoder', 
    'DualQNetwork',
    'create_qnetwork',
    # Agents
    'ParticleEvoQRE',
    'MultiAgentEvoQRE',
    'EvoQREConfig',
    # Stability
    'estimate_alpha_kappa',
    'adaptive_tau',
    'verify_stability',
    'run_stability_diagnostics',
    # Langevin
    'langevin_sample',
    'langevin_sample_batch',
    'projected_langevin_step',
    'compute_soft_value',
    # Utils
    'soft_update',
    'hard_update',
    'subsample_opponents',
    'ReplayBuffer',
]

__version__ = '2.0.0'
