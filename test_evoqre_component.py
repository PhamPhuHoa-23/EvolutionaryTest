import torch
import sys
import os

# Add root directory to path
sys.path.append(os.getcwd())

from algorithm.EvoQRE import EvoQRE
import yaml

def test_evoqre_initialization():
    print("Testing EvoQRE Initialization...")
    config = {
        'hidden_dim': 64,
        'actor_learning_rate': 1e-3,
        'critic_learning_rate': 1e-3,
        'constrainted_critic_learning_rate': 1e-3,
        'density_learning_rate': 1e-3,
        'N_quantile': 32,
        'cost_quantile': 28,
        'tau_update': 0.005,
        'LR_QN': 1e-3,
        'type': 'CVaR',
        'method': 'SplineDQN',
        'is_magnet': False,
        'eta_coef1': 0.0,
        'beta_coef': 0.0, 
        'penalty_initial_value': 1.0,
        'lamda': 0.95,
        'eps': 0.2,
        'gamma': 0.99,
        'offset': 5,
        'entropy_coef': 0.01,
        'epochs': 1,
        'algorithm': 'EvoQRE',
        'gae': True,
        'target_kl': 0.01,
        'batch_size': 10
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    agent = EvoQRE(state_dim=128, agent_number=2, config=config, device=device)
    print("EvoQRE initialized successfully.")
    
    # Test forward pass of policy
    dummy_state = torch.randn(1, 128).to(device)
    weights, means, scales = agent.get_action_dist(dummy_state)
    print(f"Policy Output Shapes: weights={weights.shape}, means={means.shape}, scales={scales.shape}")
    
    assert weights.shape == (1, 10) # default components
    assert means.shape == (1, 10, 2)
    assert scales.shape == (1, 10, 2)
    
    # Test Action choice
    action = agent.choose_action(dummy_state.squeeze(0))
    print(f"Sampled Action: {action.shape}")
    assert action.shape == (2,)

    print("Test passed!")

if __name__ == '__main__':
    test_evoqre_initialization()
