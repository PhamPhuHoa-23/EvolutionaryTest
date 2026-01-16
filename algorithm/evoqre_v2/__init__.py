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

from .q_network import ConcaveQNetwork, SpectralNormEncoder, create_qnetwork
from .agent import ParticleEvoQRE
from .stability import estimate_alpha_kappa, adaptive_tau, verify_stability
from .langevin import langevin_sample, projected_langevin_step
from .utils import soft_update, hard_update

__all__ = [
    'ConcaveQNetwork',
    'SpectralNormEncoder', 
    'create_qnetwork',
    'ParticleEvoQRE',
    'estimate_alpha_kappa',
    'adaptive_tau',
    'verify_stability',
    'langevin_sample',
    'projected_langevin_step',
    'soft_update',
    'hard_update',
]

__version__ = '2.0.0'
