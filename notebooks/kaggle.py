# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # EvoQRE vs TrafficGamer Multi-Agent Training on Argoverse 2
# 
# **Comparison of 2 RL algorithms for traffic simulation:**
# - **EvoQRE (Langevin)**: Proposed method using Reflected Langevin Dynamics
# - **TrafficGamer**: Baseline using CCE + Distributional RL
# 
# **Metrics**: Reward, Cost (Collision), minADE, minFDE
# 
# **Repository**: https://github.com/PhamPhuHoa-23/EvolutionaryTest

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Install Dependencies

# %% [code] {"jupyter":{"outputs_hidden":false}}
!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning==2.0.0
!pip install -q torch-geometric
!pip install -q av av2 neptune

# %% [code] {"jupyter":{"outputs_hidden":false}}
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    cuda_version = torch.version.cuda.replace('.', '')[:3]
    !pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_version}.html
else:
    !pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cpu.html

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Setup Environment

# %% [code] {"jupyter":{"outputs_hidden":false}}
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
from argparse import Namespace

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {DEVICE}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Clone Repository

# %% [code] {"jupyter":{"outputs_hidden":false}}
REPO_DIR = Path("EvolutionaryTest")

if not REPO_DIR.exists():
    print("📥 Cloning EvolutionaryTest...")
    !git clone https://github.com/PhamPhuHoa-23/EvolutionaryTest.git
    print("✅ Cloned successfully")
else:
    print("✅ EvolutionaryTest already exists")
    # Pull latest changes
    !cd EvolutionaryTest && git pull

sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
!pip install -q -r requirements.txt
!pip install -q neptune

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 4. Imports

# %% [code] {"jupyter":{"outputs_hidden":false}}
from algorithm.TrafficGamer import TrafficGamer
from algorithm.EvoQRE_Langevin import EvoQRE_Langevin
from predictors.autoval import AutoQCNet
from datasets import ArgoverseV2Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from transforms import TargetBuilder
from utils.utils import seed_everything
from utils.rollout import PPO_process_batch

print("✅ Imports successful")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Configuration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-14T17:49:02.749517Z","iopub.execute_input":"2026-01-14T17:49:02.750101Z","iopub.status.idle":"2026-01-14T17:49:02.754791Z","shell.execute_reply.started":"2026-01-14T17:49:02.750081Z","shell.execute_reply":"2026-01-14T17:49:02.754419Z"}}
CONFIG = {
    # Paths
    'checkpoint_path': '/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt',
    'data_root': '/kaggle/input/argoverse/argoverse',
    
    # Training settings
    'seed': 42,
    'num_episodes': 25000,       # Training episodes per scenario
    'batch_size': 4,          # Rollouts per episode
    'num_scenarios': 10,      # Number of scenarios to train on
    'max_agents': 5,          # Max agents per scenario (for speed)
    
    # RL config
    'rl_config_file': 'configs/TrafficGamer.yaml',
    
    # Safety constraints
    'distance_limit': 5.0,
    'penalty_initial_value': 1.0,
    'cost_quantile': 48,
    
    # Magnet (imitation guidance)
    'use_magnet': False,
    'eta_coef1': 0.0,
    'eta_coef2': 0.1,
}

seed_everything(CONFIG['seed'])

# Create args namespace for compatibility
args = Namespace(
    scenario=1,
    distance_limit=CONFIG['distance_limit'],
    eta_coef1=CONFIG['eta_coef1'],
    eta_coef2=CONFIG['eta_coef2'],
    track=False,
    magnet=CONFIG['use_magnet'],
)

print("✅ Configuration loaded")
print(f"   Episodes: {CONFIG['num_episodes']}")
print(f"   Scenarios: {CONFIG['num_scenarios']}")
print(f"   Max Agents: {CONFIG['max_agents']}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Load World Model (QCNet)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-14T17:47:08.914459Z","iopub.execute_input":"2026-01-14T17:47:08.914795Z","iopub.status.idle":"2026-01-14T17:47:09.442388Z","shell.execute_reply.started":"2026-01-14T17:47:08.914779Z","shell.execute_reply":"2026-01-14T17:47:09.441916Z"}}
print("🔄 Loading AutoQCNet...")

model = AutoQCNet.load_from_checkpoint(
    checkpoint_path=CONFIG['checkpoint_path'],
    map_location=DEVICE
)
model.eval()
model = model.to(DEVICE)

# Freeze encoder/decoder
for param in model.encoder.parameters():
    param.requires_grad = False
for param in model.decoder.parameters():
    param.requires_grad = False

print("✅ World Model Loaded!")
print(f"   Num Modes: {model.num_modes}")
print(f"   Hidden Dim: {model.hidden_dim}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Load Dataset

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("🔄 Loading Dataset...")

dataset = ArgoverseV2Dataset(
    root=CONFIG['data_root'],
    split='val',
    transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
)

print(f"✅ Dataset loaded: {len(dataset)} scenarios")

NUM_SCENARIOS = min(CONFIG['num_scenarios'], len(dataset))
print(f"📊 Will train on {NUM_SCENARIOS} scenarios")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 8. Load RL Configuration

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-14T17:49:06.309797Z","iopub.execute_input":"2026-01-14T17:49:06.310277Z","iopub.status.idle":"2026-01-14T17:49:06.317606Z","shell.execute_reply.started":"2026-01-14T17:49:06.310257Z","shell.execute_reply":"2026-01-14T17:49:06.317216Z"}}
with open(CONFIG['rl_config_file'], 'r') as f:
    rl_config = yaml.safe_load(f)

# Add missing defaults
rl_config.setdefault('hidden_dim', 128)
rl_config.setdefault('gamma', 0.99)
rl_config.setdefault('lamda', 0.95)
rl_config.setdefault('actor_learning_rate', 5e-5)
rl_config.setdefault('critic_learning_rate', 1e-4)
rl_config.setdefault('constrainted_critic_learning_rate', 1e-4)
rl_config.setdefault('density_learning_rate', 3e-4)
rl_config.setdefault('eps', 0.2)
rl_config.setdefault('offset', 5)
rl_config.setdefault('entropy_coef', 0.005)
rl_config.setdefault('epochs', 10)
rl_config.setdefault('gae', True)
rl_config.setdefault('target_kl', 0.01)
rl_config.setdefault('is_magnet', CONFIG['use_magnet'])
rl_config.setdefault('eta_coef1', CONFIG['eta_coef1'])
rl_config.setdefault('eta_coef2', CONFIG['eta_coef2'])
rl_config.setdefault('beta_coef', 0.1)
rl_config.setdefault('N_quantile', 64)
rl_config.setdefault('tau_update', 0.01)
rl_config.setdefault('LR_QN', 3e-4)
rl_config.setdefault('type', 'CVaR')
rl_config.setdefault('method', 'SplineDQN')
rl_config['batch_size'] = CONFIG['batch_size']
rl_config['episodes'] = CONFIG['num_episodes']
rl_config['penalty_initial_value'] = CONFIG['penalty_initial_value']
rl_config['cost_quantile'] = CONFIG['cost_quantile']

# Langevin specific
rl_config.setdefault('langevin_steps', 20)
rl_config.setdefault('langevin_step_size', 0.05)
rl_config.setdefault('tau', 0.5)
rl_config.setdefault('action_bound', 1.0)

offset = rl_config['offset']
STATE_DIM = model.num_modes * rl_config['hidden_dim']

print("✅ RL Configuration loaded")
print(f"   Epochs: {rl_config['epochs']}")
print(f"   State Dim: {STATE_DIM}")

# Import for static map loading
from av2.map.map_api import ArgoverseStaticMap
from pathlib import Path as Pth

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 9. Training Function

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-14T17:49:08.293483Z","iopub.execute_input":"2026-01-14T17:49:08.29405Z","iopub.status.idle":"2026-01-14T17:49:08.299485Z","shell.execute_reply.started":"2026-01-14T17:49:08.29403Z","shell.execute_reply":"2026-01-14T17:49:08.299081Z"}}
def train_agents(agents, agent_name, num_episodes, rl_config, new_input_data, model, choose_agent, offset, args, agent_num, scenario_static_map):
    """Train agents and return metrics."""
    metrics = {'episode_rewards': [], 'episode_costs': []}
    
    for episode in tqdm(range(num_episodes), desc=f"{agent_name}", leave=False):
        # Initialize transition list
        transition_list = [
            {
                "observations": [[] for _ in range(agent_num)],
                "actions": [[] for _ in range(agent_num)],
                "next_observations": [[] for _ in range(agent_num)],
                "rewards": [[] for _ in range(agent_num)],
                "magnet": [[] for _ in range(agent_num)],
                "costs": [[] for _ in range(agent_num)],
                "dones": [],
            }
            for _ in range(rl_config['batch_size'])
        ]
        
        # Rollout batches
        with torch.no_grad():
            for batch in range(rl_config['batch_size']):
                PPO_process_batch(
                    args, batch, new_input_data, model, agents, choose_agent,
                    offset, scenario_static_map, 1, transition_list,
                    render=False, agent_num=agent_num, dataset_type='av2'
                )
        
        # Update agents
        for i in range(agent_num):
            logs = agents[i].update(transition_list, i)
        
        # Calculate episode metrics
        episode_reward = 0
        episode_cost = 0
        for i in range(agent_num):
            for t in range(int(model.num_future_steps / offset)):
                for b in range(rl_config['batch_size']):
                    episode_reward += float(transition_list[b]["rewards"][i][t])
                    episode_cost += float(transition_list[b]["costs"][i][t])
        
        metrics['episode_rewards'].append(episode_reward / (agent_num * rl_config['batch_size']))
        metrics['episode_costs'].append(episode_cost / (agent_num * rl_config['batch_size']))
    
    return metrics

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 10. Multi-Agent Scenario Training Loop

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-14T17:49:14.607659Z","iopub.execute_input":"2026-01-14T17:49:14.608275Z","iopub.status.idle":"2026-01-14T17:49:14.611458Z","shell.execute_reply.started":"2026-01-14T17:49:14.608255Z","shell.execute_reply":"2026-01-14T17:49:14.611068Z"}}
# Metrics aggregation
all_tg_rewards = []
all_tg_costs = []
all_evo_rewards = []
all_evo_costs = []
scenario_results = []

print(f"\n{'='*60}")
print(f"🚀 Starting Multi-Agent Training")
print(f"   Scenarios: {NUM_SCENARIOS}")
print(f"   Episodes/Scenario: {CONFIG['num_episodes']}")
print(f"   Max Agents: {CONFIG['max_agents']}")
print(f"{'='*60}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-14T17:49:18.095954Z","iopub.execute_input":"2026-01-14T17:49:18.09646Z"}}
from torch_geometric.loader import DataLoader as PyGLoader

for scenario_idx in tqdm(range(NUM_SCENARIOS), desc="Scenarios"):
    print(f"\n📍 Scenario {scenario_idx + 1}/{NUM_SCENARIOS}")
    
    # Load scenario data using DataLoader
    single_loader = PyGLoader([dataset[scenario_idx]], batch_size=1, shuffle=False)
    data = next(iter(single_loader)).to(DEVICE)
    
    # Handle batch indexing
    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]
    
    # ===== MULTI-AGENT SELECTION =====
    # Select all valid vehicles as agents
    all_vehicle_mask = data["agent"]["type"] == 0  # Type 0 = vehicle
    valid_mask = data["agent"]["valid_mask"][:, model.num_historical_steps - 1]  # Valid at last history step
    combined_mask = all_vehicle_mask & valid_mask
    
    agent_indices = torch.nonzero(combined_mask, as_tuple=False).squeeze(-1)
    
    if len(agent_indices) == 0:
        print(f"   ⚠️ No valid vehicles, skipping...")
        continue
    
    # Limit number of agents
    agent_num = min(len(agent_indices), CONFIG['max_agents'])
    agent_indices = agent_indices[:agent_num]
    choose_agent = agent_indices.tolist()
    
    # Use data directly (no expand_data needed since we're using existing vehicles)
    new_input_data = data
    rl_config['agent_number'] = agent_num
    
    # Get scenario_id for map loading
    if 'scenario_id' in data:
        scenario_id = data['scenario_id'] if isinstance(data['scenario_id'], str) else data['scenario_id'][0]
    else:
        scenario_id = dataset.processed_file_names[scenario_idx].replace('.pkl', '')
    
    # Load static map
    map_path = Pth(CONFIG['data_root']) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
    try:
        scenario_static_map = ArgoverseStaticMap.from_json(map_path) if map_path.exists() else None
    except:
        scenario_static_map = None
    
    if scenario_static_map is None:
        print(f"   ⚠️ No map found, skipping...")
        continue
    
    # print(f"   Agents: {agent_num}, Map: ✓")
    
    # Initialize agents for this scenario
    tg_agents = [TrafficGamer(STATE_DIM, agent_num, rl_config, DEVICE) for _ in range(agent_num)]
    evo_agents = [EvoQRE_Langevin(STATE_DIM, agent_num, rl_config, DEVICE) for _ in range(agent_num)]
    
    # Train TrafficGamer
    tg_metrics = train_agents(
        tg_agents, "TrafficGamer", CONFIG['num_episodes'], rl_config,
        new_input_data, model, choose_agent, offset, args, agent_num, scenario_static_map
    )
    all_tg_rewards.extend(tg_metrics['episode_rewards'])
    all_tg_costs.extend(tg_metrics['episode_costs'])
    
    # Train EvoQRE
    evo_metrics = train_agents(
        evo_agents, "EvoQRE", CONFIG['num_episodes'], rl_config,
        new_input_data, model, choose_agent, offset, args, agent_num, scenario_static_map
    )
    all_evo_rewards.extend(evo_metrics['episode_rewards'])
    all_evo_costs.extend(evo_metrics['episode_costs'])
    
    # Save scenario result
    scenario_results.append({
        'scenario_id': scenario_id,
        'agent_num': agent_num,
        'tg_reward': np.mean(tg_metrics['episode_rewards'][-5:]),
        'tg_cost': np.mean(tg_metrics['episode_costs'][-5:]),
        'evo_reward': np.mean(evo_metrics['episode_rewards'][-5:]),
        'evo_cost': np.mean(evo_metrics['episode_costs'][-5:]),
    })
    
    print(f"   TG: R={scenario_results[-1]['tg_reward']:.2f}, C={scenario_results[-1]['tg_cost']:.2f}")
    print(f"   EvoQRE: R={scenario_results[-1]['evo_reward']:.2f}, C={scenario_results[-1]['evo_cost']:.2f}")

# %% [code] {"jupyter":{"outputs_hidden":false}}
print(f"\n{'='*60}")
print("✅ Training Complete!")
print(f"   Scenarios processed: {len(scenario_results)}")
print(f"   Total episodes: {len(all_tg_rewards)}")
print(f"{'='*60}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 11. Results Summary

# %% [code] {"jupyter":{"outputs_hidden":false}}
print("=" * 60)
print("📊 TRAINING RESULTS")
print("=" * 60)
print()

print(f"--- TrafficGamer ---")
print(f"   Avg Reward: {np.mean(all_tg_rewards):.4f} ± {np.std(all_tg_rewards):.4f}")
print(f"   Avg Cost:   {np.mean(all_tg_costs):.4f} ± {np.std(all_tg_costs):.4f}")
print()

print(f"--- EvoQRE ---")
print(f"   Avg Reward: {np.mean(all_evo_rewards):.4f} ± {np.std(all_evo_rewards):.4f}")
print(f"   Avg Cost:   {np.mean(all_evo_costs):.4f} ± {np.std(all_evo_costs):.4f}")
print()

# Collision rate (cost > 0 means collision)
tg_collision_rate = np.mean([1 if c > 0 else 0 for c in all_tg_costs]) * 100
evo_collision_rate = np.mean([1 if c > 0 else 0 for c in all_evo_costs]) * 100
print(f"   TG Collision Rate:     {tg_collision_rate:.2f}%")
print(f"   EvoQRE Collision Rate: {evo_collision_rate:.2f}%")

print("=" * 60)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 12. Visualization

# %% [code] {"jupyter":{"outputs_hidden":false}}
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Rewards over episodes
axes[0, 0].plot(all_tg_rewards, label='TrafficGamer', alpha=0.7, color='#2196F3')
axes[0, 0].plot(all_evo_rewards, label='EvoQRE', alpha=0.7, color='#4CAF50')
axes[0, 0].set_title('Training Rewards', fontsize=14, fontweight='bold')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Reward')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# Costs over episodes
axes[0, 1].plot(all_tg_costs, label='TrafficGamer', alpha=0.7, color='#F44336')
axes[0, 1].plot(all_evo_costs, label='EvoQRE', alpha=0.7, color='#FF9800')
axes[0, 1].set_title('Training Costs (Lower = Safer)', fontsize=14, fontweight='bold')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Cost')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# Bar comparison - Reward
reward_means = [np.mean(all_tg_rewards), np.mean(all_evo_rewards)]
reward_stds = [np.std(all_tg_rewards), np.std(all_evo_rewards)]
bars1 = axes[1, 0].bar(['TrafficGamer', 'EvoQRE'], reward_means, yerr=reward_stds,
                        color=['#2196F3', '#4CAF50'], capsize=5, alpha=0.8)
axes[1, 0].set_title('Average Reward Comparison', fontsize=14, fontweight='bold')
axes[1, 0].set_ylabel('Reward')
axes[1, 0].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, reward_means):
    axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=11)

# Bar comparison - Cost
cost_means = [np.mean(all_tg_costs), np.mean(all_evo_costs)]
cost_stds = [np.std(all_tg_costs), np.std(all_evo_costs)]
bars2 = axes[1, 1].bar(['TrafficGamer', 'EvoQRE'], cost_means, yerr=cost_stds,
                        color=['#F44336', '#FF9800'], capsize=5, alpha=0.8)
axes[1, 1].set_title('Average Cost Comparison', fontsize=14, fontweight='bold')
axes[1, 1].set_ylabel('Cost')
axes[1, 1].grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars2, cost_means):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=11)

plt.tight_layout()
plt.savefig('training_comparison.png', dpi=150)
plt.show()

print("✅ Plot saved to 'training_comparison.png'")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 13. Export Results

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Summary table
results_df = pd.DataFrame({
    'Method': ['TrafficGamer', 'EvoQRE'],
    'Avg_Reward': [np.mean(all_tg_rewards), np.mean(all_evo_rewards)],
    'Std_Reward': [np.std(all_tg_rewards), np.std(all_evo_rewards)],
    'Avg_Cost': [np.mean(all_tg_costs), np.mean(all_evo_costs)],
    'Std_Cost': [np.std(all_tg_costs), np.std(all_evo_costs)],
    'Collision_Rate_%': [tg_collision_rate, evo_collision_rate],
})

print("📋 Results Table:")
print(results_df.to_markdown(index=False))

results_df.to_csv('comparison_results.csv', index=False)
print("\n✅ Results saved to 'comparison_results.csv'")

# %% [code] {"jupyter":{"outputs_hidden":false}}
# Per-scenario results
scenario_df = pd.DataFrame(scenario_results)
scenario_df.to_csv('scenario_results.csv', index=False)
print(f"✅ Scenario results saved ({len(scenario_results)} scenarios)")
print(scenario_df.head(10).to_markdown(index=False))

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 14. Conclusion
# 
# **Training Complete!**
# - TrafficGamer: CCE-based MAPPO with distributional RL
# - EvoQRE: Reflected Langevin Dynamics for QRE sampling
# - Compare rewards (higher is better) and costs (lower is safer)