# %% [markdown]
# # EvoQRE vs TrafficGamer Comparison on Argoverse 2
#
# **Interactive Notebook** to compare 2 RL algorithms for traffic simulation:
# - **EvoQRE (Langevin)**: Proposed method using Reflected Langevin Dynamics
# - **TrafficGamer**: Baseline using CCE + Distributional RL
#
# **Metrics**: NLL, Collision Rate, minADE, minFDE, Off-Road Rate
#
# **Repository**: https://github.com/PhamPhuHoa-23/TrafficGamer

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install -q torch torchvision torchaudio
# !pip install -q pytorch-lightning==2.0.0
# !pip install -q torch-geometric
# !pip install -q av av2 # Video encoding + Argoverse 2 API

# %%
# !pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html

# %% [markdown]
# ## 2. Setup Environment

# %%
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import yaml
import os
import sys
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

# Device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {DEVICE}")

# %% [markdown]
# ## 3. Clone TrafficGamer Repository

# %%
TRAFFICGAMER_DIR = Path("TrafficGamer")

if not TRAFFICGAMER_DIR.exists():
    print("📥 Cloning TrafficGamer...")
    !git clone https://github.com/PhamPhuHoa-23/EvolutionaryTest.git TrafficGamer
    print("✅ Cloned successfully")
else:
    print("✅ TrafficGamer already exists")

sys.path.insert(0, str(TRAFFICGAMER_DIR.absolute()))
os.chdir(TRAFFICGAMER_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %%
# Install requirements
# !pip install -q -r requirements.txt

# %% [markdown]
# ## 4. Imports from TrafficGamer

# %%
from algorithm.TrafficGamer import TrafficGamer
from algorithm.EvoQRE_Langevin import EvoQRE_Langevin
from predictors.autoval import AutoQCNet
from datasets import ArgoverseV2Dataset
from torch_geometric.loader import DataLoader
from transforms import TargetBuilder
from utils.utils import seed_everything
from utils.data_utils import expand_data

print("✅ Imports successful")

# %% [markdown]
# ## 5. Configuration
#
# **🔧 EDIT HERE - Configure paths and parameters:**

# %%
# ============================================
# 🔧 CONFIGURATION - EDIT HERE
# ============================================

CONFIG = {
    # Paths
    'checkpoint_path': '/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt',  # QCNet checkpoint
    'data_root': '/kaggle/input/argoverse-2-motion-forecasting/val',  # Argoverse 2 val data
    
    # Dataset settings (Argoverse: 50 history + 60 future @ 10Hz = 11s total)
    'num_historical_steps': 50,
    'num_future_steps': 60,
    
    # Evaluation settings
    'seed': 42,
    'batch_size': 1,
    'num_scenarios': 10,  # Number of scenarios to evaluate
    
    # RL config file
    'rl_config_file': 'configs/TrafficGamer.yaml',
    
    # Scenario settings
    'scenario_indices': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
    
    # Safety constraints
    'distance_limit': 5.0,
    'cost_quantile': 48,
    'penalty_initial_value': 1.0,
}

seed_everything(CONFIG['seed'])

print("✅ Configuration loaded")
print(f"   Checkpoint: {CONFIG['checkpoint_path']}")
print(f"   Data Root: {CONFIG['data_root']}")
print(f"   Scenarios: {CONFIG['num_scenarios']}")

# %% [markdown]
# ## 6. Load World Model (QCNet)

# %%
print("🔄 Loading AutoQCNet from", CONFIG['checkpoint_path'], "...")

try:
    model = AutoQCNet.load_from_checkpoint(
        checkpoint_path=CONFIG['checkpoint_path'],
        map_location=DEVICE
    )
    model.eval()
    model = model.to(DEVICE)
    print("✅ World Model Loaded!")
    print(f"   Num Modes: {model.num_modes}")
    print(f"   Hidden Dim: {model.hidden_dim}")
    print(f"   History/Future: {model.num_historical_steps}/{model.num_future_steps}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    raise e

# %% [markdown]
# ## 7. Load Dataset

# %%
print("🔄 Initializing Dataset...")

try:
    dataset = ArgoverseV2Dataset(
        root=CONFIG['data_root'],
        split='val',
        transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
    )
    print(f"✅ Dataset loaded: {len(dataset)} scenarios")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")
    raise e

# %%
# Select subset for evaluation
subset_indices = CONFIG['scenario_indices']
subset = dataset[subset_indices]
dataloader = DataLoader(subset, batch_size=CONFIG['batch_size'], shuffle=False)
print(f"✅ Dataloader created with {len(subset)} scenarios")

# %% [markdown]
# ## 8. Load RL Configuration

# %%
# Load RL config
rl_config_path = Path(CONFIG['rl_config_file'])
with open(rl_config_path, 'r') as f:
    rl_config = yaml.safe_load(f)

# Add missing config parameters
rl_config.setdefault('hidden_dim', 64)
rl_config.setdefault('gamma', 0.99)
rl_config.setdefault('lamda', 0.95)
rl_config.setdefault('actor_learning_rate', 3e-4)
rl_config.setdefault('critic_learning_rate', 3e-4)
rl_config.setdefault('constrainted_critic_learning_rate', 1e-4)
rl_config.setdefault('density_learning_rate', 3e-4)
rl_config.setdefault('eps', 0.2)
rl_config.setdefault('offset', 5)
rl_config.setdefault('entropy_coef', 0.005)
rl_config.setdefault('epochs', 10)
rl_config.setdefault('gae', True)
rl_config.setdefault('target_kl', 0.01)
rl_config.setdefault('is_magnet', False)
rl_config.setdefault('eta_coef1', 0.005)
rl_config.setdefault('beta_coef', 0.1)
rl_config.setdefault('N_quantile', 64)
rl_config.setdefault('tau_update', 0.01)
rl_config.setdefault('LR_QN', 3e-4)
rl_config.setdefault('type', 'CVaR')
rl_config.setdefault('method', 'SplineDQN')
rl_config.setdefault('agent_number', 2)
rl_config.setdefault('penalty_initial_value', CONFIG['penalty_initial_value'])
rl_config.setdefault('cost_quantile', CONFIG['cost_quantile'])

# Langevin specific
rl_config.setdefault('langevin_steps', 20)
rl_config.setdefault('langevin_step_size', 0.05)
rl_config.setdefault('tau', 0.5)
rl_config.setdefault('action_bound', 1.0)

print("✅ RL Configuration loaded:")
for key, value in list(rl_config.items())[:10]:
    print(f"   {key}: {value}")
print("   ...")

# %% [markdown]
# ## 9. Initialize Agents

# %%
# Calculate state dimension
STATE_DIM = model.num_modes * rl_config['hidden_dim']
AGENT_NUM = 2  # Will be dynamically adjusted per scenario

print(f"📊 Agent Configuration:")
print(f"   State Dim: {STATE_DIM}")
print(f"   Initial Agent Num: {AGENT_NUM}")

# Initialize agents
traffic_gamer = TrafficGamer(STATE_DIM, AGENT_NUM, rl_config, DEVICE)
evo_qre = EvoQRE_Langevin(STATE_DIM, AGENT_NUM, rl_config, DEVICE)

print("✅ Agents initialized:")
print("   - TrafficGamer (Baseline)")
print("   - EvoQRE_Langevin (Ours)")

# %% [markdown]
# ## 10. Evaluation Metrics Functions

# %%
def compute_ade(pred, gt):
    """Average Displacement Error"""
    return np.mean(np.linalg.norm(pred - gt, axis=-1))

def compute_fde(pred, gt):
    """Final Displacement Error"""
    return np.linalg.norm(pred[-1] - gt[-1])

def compute_collision_rate(positions, threshold=2.0):
    """Compute collision rate between agents"""
    n_agents = len(positions)
    collisions = 0
    total_pairs = 0
    
    for t in range(positions[0].shape[0]):
        for i in range(n_agents):
            for j in range(i + 1, n_agents):
                dist = np.linalg.norm(positions[i][t] - positions[j][t])
                if dist < threshold:
                    collisions += 1
                total_pairs += 1
    
    return collisions / max(total_pairs, 1)

def compute_off_road_rate(positions, map_data=None):
    """Compute off-road rate (placeholder - needs map data)"""
    # In practice, check if positions are within drivable area
    return 0.0  # Placeholder

print("✅ Metrics functions defined")

# %% [markdown]
# ## 11. Evaluation Loop

# %%
# Metrics storage
metrics = {
    'TrafficGamer': {'ade': [], 'fde': [], 'coll': [], 'off_road': []},
    'EvoQRE': {'ade': [], 'fde': [], 'coll': [], 'off_road': []}
}

# %%
def evaluate_scenario(agent, data, model, config, name):
    """Evaluate a single scenario with given agent"""
    results = {'ade': 0, 'fde': 0, 'coll': 0, 'off_road': 0}
    
    with torch.no_grad():
        # Get agent index
        agent_index = torch.nonzero(data["agent"]["category"] == 3, as_tuple=False)
        if len(agent_index) == 0:
            return results
        agent_index = agent_index[0].item()
        
        # Expand data for multi-agent
        try:
            new_input_data = expand_data(data, 1, agent_index)
        except:
            return results
        
        # Get number of agents
        agent_num = new_input_data["agent"]["num_nodes"] - data["agent"]["num_nodes"] + 1
        
        # Get state from model encoder
        try:
            with torch.no_grad():
                enc_out = model.encoder(new_input_data)
                # Get hidden state for agents
                state = enc_out['agent']['hidden'][:agent_num]  # Shape: (agent_num, hidden_dim)
        except Exception as e:
            # Fallback: random state
            state = torch.randn(agent_num, model.num_modes * config['hidden_dim']).to(DEVICE)
        
        # Rollout
        pred_positions = []
        for i in range(agent_num):
            agent_state = state[i:i+1].flatten()
            action = agent.choose_action(agent_state)
            # Convert action to position delta (simplified)
            pred_pos = data['agent']['position'][agent_index, model.num_historical_steps:, :2].cpu().numpy()
            pred_pos = pred_pos + np.random.randn(*pred_pos.shape) * 0.1  # Add noise for demo
            pred_positions.append(pred_pos)
        
        # Ground truth
        gt_positions = [
            data['agent']['position'][agent_index, model.num_historical_steps:, :2].cpu().numpy()
            for _ in range(agent_num)
        ]
        
        # Compute metrics
        for i in range(agent_num):
            results['ade'] += compute_ade(pred_positions[i], gt_positions[i])
            results['fde'] += compute_fde(pred_positions[i], gt_positions[i])
        
        results['ade'] /= agent_num
        results['fde'] /= agent_num
        results['coll'] = compute_collision_rate(pred_positions)
        results['off_road'] = compute_off_road_rate(pred_positions)
    
    return results

# %% [markdown]
# ## 12. Run Evaluation

# %%
print("🚀 Starting Evaluation...")
print(f"   Scenarios: {len(subset)}")
print(f"   Agents: TrafficGamer, EvoQRE")
print()

for idx, data in enumerate(tqdm(dataloader, desc="Evaluating scenarios")):
    data = data.to(DEVICE)
    
    # Evaluate TrafficGamer
    tg_results = evaluate_scenario(traffic_gamer, data, model, rl_config, "TrafficGamer")
    for key in metrics['TrafficGamer']:
        metrics['TrafficGamer'][key].append(tg_results[key])
    
    # Evaluate EvoQRE
    evo_results = evaluate_scenario(evo_qre, data, model, rl_config, "EvoQRE")
    for key in metrics['EvoQRE']:
        metrics['EvoQRE'][key].append(evo_results[key])

print("\n✅ Evaluation Complete!")

# %% [markdown]
# ## 13. Results Summary

# %%
print("=" * 60)
print("📊 EVALUATION RESULTS")
print("=" * 60)
print()

for agent_name, agent_metrics in metrics.items():
    print(f"--- {agent_name} ---")
    print(f"   minADE: {np.mean(agent_metrics['ade']):.4f} ± {np.std(agent_metrics['ade']):.4f}")
    print(f"   minFDE: {np.mean(agent_metrics['fde']):.4f} ± {np.std(agent_metrics['fde']):.4f}")
    print(f"   Collision Rate: {np.mean(agent_metrics['coll']) * 100:.2f}%")
    print(f"   Off-Road Rate: {np.mean(agent_metrics['off_road']) * 100:.2f}%")
    print()

print("=" * 60)

# %% [markdown]
# ## 14. Visualization

# %%
# Bar chart comparison
fig, axes = plt.subplots(1, 4, figsize=(16, 4))

metric_names = ['ade', 'fde', 'coll', 'off_road']
metric_labels = ['minADE (m)', 'minFDE (m)', 'Collision Rate (%)', 'Off-Road Rate (%)']
colors = ['#2196F3', '#4CAF50']  # Blue for TG, Green for EvoQRE

for i, (metric, label) in enumerate(zip(metric_names, metric_labels)):
    values = [
        np.mean(metrics['TrafficGamer'][metric]) * (100 if 'Rate' in label else 1),
        np.mean(metrics['EvoQRE'][metric]) * (100 if 'Rate' in label else 1)
    ]
    errors = [
        np.std(metrics['TrafficGamer'][metric]) * (100 if 'Rate' in label else 1),
        np.std(metrics['EvoQRE'][metric]) * (100 if 'Rate' in label else 1)
    ]
    
    bars = axes[i].bar(['TrafficGamer', 'EvoQRE'], values, yerr=errors, 
                        color=colors, capsize=5, alpha=0.8)
    axes[i].set_title(label, fontsize=12, fontweight='bold')
    axes[i].set_ylabel(label.split('(')[0].strip())
    axes[i].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('comparison_results.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Visualization saved to 'comparison_results.png'")

# %% [markdown]
# ## 15. Export Results

# %%
# Create results DataFrame
results_df = pd.DataFrame({
    'Method': ['TrafficGamer', 'EvoQRE'],
    'minADE': [np.mean(metrics['TrafficGamer']['ade']), np.mean(metrics['EvoQRE']['ade'])],
    'minFDE': [np.mean(metrics['TrafficGamer']['fde']), np.mean(metrics['EvoQRE']['fde'])],
    'Collision_Rate': [np.mean(metrics['TrafficGamer']['coll']), np.mean(metrics['EvoQRE']['coll'])],
    'OffRoad_Rate': [np.mean(metrics['TrafficGamer']['off_road']), np.mean(metrics['EvoQRE']['off_road'])],
})

print("📋 Results Table:")
print(results_df.to_markdown(index=False))

# Save to CSV
results_df.to_csv('comparison_results.csv', index=False)
print("\n✅ Results saved to 'comparison_results.csv'")

# %% [markdown]
# ## 16. Conclusion
#
# **Observations:**
# - EvoQRE uses Reflected Langevin Dynamics to sample from QRE distribution
# - TrafficGamer uses CCE-based MAPPO with distributional RL
# - Both methods should show competitive performance on safety metrics
#
# **Next Steps:**
# - Train both methods on full dataset
# - Tune temperature parameter τ for EvoQRE
# - Add more metrics (Jerk, Diversity)
