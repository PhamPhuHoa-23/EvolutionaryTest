# %%
import torch
import torch.nn as nn
import numpy as np
import yaml
import os
import sys
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append('..')

from algorithm.TrafficGamer import TrafficGamer
from algorithm.EvoQRE_Langevin import EvoQRE_Langevin
from predictors.autoval import AutoQCNet
from datasets import ArgoverseV2Dataset
from torch_geometric.loader import DataLoader
from transforms import TargetBuilder
from utils.utils import seed_everything
from utils.data_utils import expand_data

# %% [markdown]
# # EvoQRE vs TrafficGamer Comparison
# This notebook compares the proposed EvoQRE (Langevin) agent against the baseline TrafficGamer on the Argoverse 2 Validation Set.

# %%
# Configuration
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
CONFIG_PATH = '../configs/EvoQRE.yaml' # Assuming EvoQRE config works for both or we load base config
CHECKPOINT_PATH = '~/Multi-agent-competitive-environment/checkpoints/epoch=19-step=499780.ckpt' # Placeholder
DATA_ROOT = '../datasets' # Placeholder, user needs to adjust

# Metrics Storage
metrics = {
    'TrafficGamer': {'nll': [], 'coll': [], 'ade': [], 'fde': [], 'off_road': []},
    'EvoQRE': {'nll': [], 'coll': [], 'ade': [], 'fde': [], 'off_road': []}
}

# %%
# Load Config
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Load World Model (QCNet)
model = AutoQCNet.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
model.eval()

# Load Dataset
dataset = ArgoverseV2Dataset(
    root=DATA_ROOT,
    split='val',
    transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
)

# Select a subset for evaluation (e.g., first 10 scenarios)
subset_indices = range(10)
subset = dataset[subset_indices]
dataloader = DataLoader(subset, batch_size=1, shuffle=False)

# %%
# Initialize Agents
state_dim = model.num_modes * config["hidden_dim"]
agent_num = 2 # Placeholder, dynamically adjusted in loop usually

traffic_gamer = TrafficGamer(state_dim, agent_num, config, DEVICE)
evo_qre = EvoQRE_Langevin(state_dim, agent_num, config, DEVICE)

# Load Agent Checkpoints (Placeholder)
# traffic_gamer.load_state_dict(torch.load('path/to/tg.pth'))
# evo_qre.load_state_dict(torch.load('path/to/evo.pth'))

# %%
# Evaluation Loop
def evaluate_agent(agent, loader, name):
    results = {'nll': [], 'coll': [], 'ade': [], 'fde': [], 'off_road': [], 'jerk': []}
    
    for batch in tqdm(loader, desc=f"Evaluating {name}"):
        batch = batch.to(DEVICE)
        
        # Prepare Data (expand logic from train_evoqre.py)
        # Note: This is simplified. Actual expansion needs the helper function.
        # Here we assume batch is ready or we skip complex expansion for this snippet.
        # In real run, use expand_data(batch, ...)
        
        # Mock Rollout
        # In a real game evaluation, we would loop over time steps T
        # Here we just check one-step prediction or full rollout if feasible
        
        # 1. Get Action from Agent
        # state = batch.x ...
        # action = agent.choose_action(state)
        
        # 2. Update Dynamics (Mock)
        # next_state = ...
        
        # 3. Calculate Metrics
        # NLL
        # Collisions
        # ADE/FDE vs Ground Truth
        
        # Placeholder Results
        results['nll'].append(np.random.normal(0.5, 0.1))
        results['coll'].append(np.random.choice([0, 1], p=[0.95, 0.05]))
        results['ade'].append(np.random.normal(1.0, 0.2))
        results['fde'].append(np.random.normal(2.0, 0.5))
        results['off_road'].append(0.0)
    
    return results

# %%
# Run Evaluation
# metrics['TrafficGamer'] = evaluate_agent(traffic_gamer, dataloader, 'TrafficGamer')
# metrics['EvoQRE'] = evaluate_agent(evo_qre, dataloader, 'EvoQRE')

# %%
# Note: Since we don't have trained weights ready for execution in this environment,
# the above loop is commented out. 
# Usage: Uncomment and set paths to actual checkpoints to run.

# Print Summary
# print("Results:")
# for agent, res in metrics.items():
#     print(f"--- {agent} ---")
#     print(f"NLL: {np.mean(res['nll']):.4f}")
#     print(f"Collision Rate: {np.mean(res['coll']) * 100:.2f}%")
#     print(f"minADE: {np.mean(res['ade']):.4f}")
#     print(f"minFDE: {np.mean(res['fde']):.4f}")

# %%
# Visualization Placeholder
def plot_results(metrics):
    labels = list(metrics.keys())
    nll = [np.mean(metrics[l]['nll']) for l in labels]
    coll = [np.mean(metrics[l]['coll']) for l in labels]
    
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    
    ax[0].bar(labels, nll, color=['blue', 'green'])
    ax[0].set_title('NLL (Lower is Better)')
    
    ax[1].bar(labels, coll, color=['blue', 'green'])
    ax[1].set_title('Collision Rate')
    
    plt.tight_layout()
    plt.show()

# plot_results(metrics)
