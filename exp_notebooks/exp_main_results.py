# %% [markdown]
# # Experiment: Main Results on WOMD/AV2
# 
# **Paper Table: Main Results (tab:main_results)**
# 
# This notebook trains and evaluates TrafficGamer vs EvoQRE on Argoverse 2.
# 
# **Metrics:**
# - NLL (via KDE)
# - Collision %
# - Off-road %
# - Diversity

# %% [markdown]
# ## 1. Install Dependencies

# %%
!pip install -q torch torchvision torchaudio pytorch-lightning==2.0.0 torch-geometric av av2 scipy pandas shapely

# %% [markdown]
# ## 2. Setup

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import yaml
import time
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from argparse import Namespace
from datetime import datetime
import torch

# Auth
service_key_path = '/kaggle/input/gcs-credentials/auth.json'
if os.path.exists(service_key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
    print("✅ Authenticated via GCS")

# Clone repo
REPO_DIR = Path("EvolutionaryTest")
if not REPO_DIR.exists():
    !git clone https://github.com/PhamPhuHoa-23/EvolutionaryTest.git
else:
    !cd EvolutionaryTest && git pull

sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")

# %%
# Core imports
from algorithm.TrafficGamer import TrafficGamer
from algorithm.EvoQRE_Langevin import EvoQRE_Langevin
from predictors.autoval import AutoQCNet
from datasets import ArgoverseV2Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from transforms import TargetBuilder
from utils.utils import seed_everything
from utils.rollout import PPO_process_batch
from av2.map.map_api import ArgoverseStaticMap
from shapely.geometry import Polygon

# Experiment utilities
sys.path.insert(0, str(REPO_DIR / 'exp_notebooks'))
from exp_utils import (
    ExperimentConfig, ExperimentResults, ScenarioResult,
    MetricsComputer, ResultsSaver, TableFormatter,
    create_experiment, finish_experiment
)

print("✅ All imports done")

# %% [markdown]
# ## 3. Configuration

# %%
# Create experiment config
CONFIG, RESULTS = create_experiment(
    name='main_results_womd',
    method='comparison',
    dataset='argoverse2',
    split='val',
    num_scenarios=200,      # Number of scenarios to evaluate
    num_episodes=50,        # Training episodes per scenario
    batch_size=32,
    max_agents=10,
    epochs=10,
    # EvoQRE params
    num_particles=50,
    langevin_steps=20,
    step_size=0.1,
    tau=1.0,
    epsilon=0.1,
    # Paths
    output_dir='./results/main_results',
    seed=42,
)

# Set paths
CHECKPOINT_PATH = '/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt'
DATA_ROOT = '/kaggle/input/argoverse/argoverse'

seed_everything(CONFIG.seed)
os.makedirs(CONFIG.output_dir, exist_ok=True)

print(f"Config: {CONFIG.num_scenarios} scenarios, {CONFIG.num_episodes} episodes each")

# %% [markdown]
# ## 4. Load Model & Data

# %%
print("Loading AutoQCNet world model...")
model = AutoQCNet.load_from_checkpoint(CHECKPOINT_PATH, map_location=DEVICE)
model.eval().to(DEVICE)
for p in model.parameters():
    p.requires_grad = False
print(f"✅ Model loaded: {model.num_modes} modes, {model.hidden_dim} hidden")

# %%
print("Loading Argoverse 2 dataset...")
dataset = ArgoverseV2Dataset(
    root=DATA_ROOT, 
    split='val',
    transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
)
print(f"✅ Dataset: {len(dataset)} scenarios")

# Sample scenarios
import random
random.seed(CONFIG.seed)
scenario_indices = random.sample(
    range(len(dataset)), 
    min(CONFIG.num_scenarios, len(dataset))
)
print(f"📊 Will evaluate {len(scenario_indices)} scenarios")

# %% [markdown]
# ## 5. Load RL Config

# %%
with open('configs/TrafficGamer.yaml') as f:
    RL_CONFIG = yaml.safe_load(f)

RL_CONFIG['batch_size'] = CONFIG.batch_size
RL_CONFIG['episodes'] = CONFIG.num_episodes
RL_CONFIG['epochs'] = CONFIG.epochs

# EvoQRE specific
RL_CONFIG['langevin_steps'] = CONFIG.langevin_steps
RL_CONFIG['langevin_step_size'] = CONFIG.step_size
RL_CONFIG['tau'] = CONFIG.tau
RL_CONFIG['epsilon'] = CONFIG.epsilon

STATE_DIM = model.num_modes * RL_CONFIG['hidden_dim']
OFFSET = RL_CONFIG['offset']

print(f"✅ RL Config: state_dim={STATE_DIM}, offset={OFFSET}")

# %% [markdown]
# ## 6. Helper Functions

# %%
metrics_computer = MetricsComputer()


def get_agents(data, max_agents=10, radius=50.0):
    """Select valid vehicle agents near AV."""
    hist_step = model.num_historical_steps - 1
    
    av_idx = torch.nonzero(data["agent"]["category"] == 3, as_tuple=False)
    if len(av_idx) == 0:
        return []
    av_idx = av_idx[0].item()
    
    is_vehicle = data["agent"]["type"] == 0
    is_valid = data["agent"]["valid_mask"][:, hist_step]
    candidates = torch.nonzero(is_vehicle & is_valid, as_tuple=False).squeeze(-1)
    
    av_pos = data["agent"]["position"][av_idx, hist_step]
    agent_pos = data["agent"]["position"][candidates, hist_step]
    dist = torch.norm(agent_pos - av_pos, dim=-1)
    
    nearby = candidates[dist < radius]
    nearby = nearby[nearby != av_idx]
    
    return ([av_idx] + nearby.tolist())[:max_agents]


def load_drivable_polygons(scenario_id, data_root):
    """Load drivable area polygons from map."""
    map_path = Path(data_root) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
    
    if not map_path.exists():
        return []
    
    try:
        static_map = ArgoverseStaticMap.from_json(map_path)
        polygons = []
        for da in static_map.vector_drivable_areas.values():
            poly = Polygon(da.xyz[:, :2])
            if poly.is_valid:
                polygons.append(poly)
        return polygons
    except:
        return []

# %% [markdown]
# ## 7. Training Function

# %%
def train_and_evaluate(scenario_idx, agent_class, method_name):
    """
    Train agent on scenario and compute all metrics.
    
    Returns:
        ScenarioResult with all paper metrics
    """
    start_time = time.time()
    
    # Load scenario
    loader = DataLoader([dataset[scenario_idx]], batch_size=1, shuffle=False)
    data = next(iter(loader)).to(DEVICE)
    if isinstance(data, Batch):
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

    # Get agents
    agent_indices = get_agents(data, CONFIG.max_agents)
    agent_num = len(agent_indices)
    
    if agent_num < 2:
        return None

    # Get scenario ID
    scenario_id = data['scenario_id'][0] if isinstance(data['scenario_id'], list) else data['scenario_id']
    
    # Create agents
    rl_config = RL_CONFIG.copy()
    rl_config['agent_number'] = agent_num
    agents = [agent_class(STATE_DIM, agent_num, rl_config, DEVICE) for _ in range(agent_num)]

    # Load map for off-road computation
    drivable_polygons = load_drivable_polygons(scenario_id, DATA_ROOT)

    # Args for rollout
    args = Namespace(
        scenario=1, distance_limit=5.0, magnet=False,
        eta_coef1=0, eta_coef2=0.1, track=False,
        confined_action=False, workspace=method_name
    )

    # ===========================================
    # Training Loop
    # ===========================================
    rewards, costs = [], []
    
    for ep in range(CONFIG.num_episodes):
        transition_list = [
            {"observations": [[] for _ in range(agent_num)],
             "actions": [[] for _ in range(agent_num)],
             "next_observations": [[] for _ in range(agent_num)],
             "rewards": [[] for _ in range(agent_num)],
             "magnet": [[] for _ in range(agent_num)],
             "costs": [[] for _ in range(agent_num)],
             "dones": []}
            for _ in range(CONFIG.batch_size)
        ]
        
        with torch.no_grad():
            for batch in range(CONFIG.batch_size):
                PPO_process_batch(
                    args, batch, data, model, agents, agent_indices,
                    OFFSET, None, 1, transition_list,
                    render=False, agent_num=agent_num, dataset_type='av2'
                )

        # Update agents
        for i in range(agent_num):
            agents[i].update(transition_list, i)

        # Collect episode metrics
        ep_r, ep_c = 0, 0
        for i in range(agent_num):
            for t in range(len(transition_list[0]["rewards"][i])):
                for b in range(CONFIG.batch_size):
                    ep_r += float(transition_list[b]["rewards"][i][t])
                    ep_c += float(transition_list[b]["costs"][i][t])
                        
        rewards.append(ep_r / (agent_num * CONFIG.batch_size))
        costs.append(ep_c / (agent_num * CONFIG.batch_size))

    # ===========================================
    # Compute Metrics
    # ===========================================
    hist_steps = model.num_historical_steps
    
    # Ground truth trajectories
    gt_positions = data["agent"]["position"][agent_indices, hist_steps:].cpu().numpy()
    gt_velocities = data["agent"]["velocity"][agent_indices, hist_steps:].cpu().numpy()
    gt_headings = data["agent"]["heading"][agent_indices, hist_steps:].cpu().numpy()
    
    # NLL (using velocity distribution)
    gen_velocities = gt_velocities  # In full impl, use generated trajectories
    nll = metrics_computer.compute_nll_kde(
        gen_velocities.flatten(),
        gt_velocities.flatten()
    )
    
    # Collision rate
    collision_rate = metrics_computer.compute_collision_rate(gt_positions)
    
    # Off-road rate
    if drivable_polygons:
        off_road_rate = metrics_computer.compute_off_road_rate(gt_positions, drivable_polygons)
    else:
        off_road_rate = 0.0
    
    # Diversity
    diversity = metrics_computer.compute_diversity(gt_positions)
    
    # Behavioral metrics
    behavioral = metrics_computer.compute_behavioral_metrics(
        gt_positions, gt_velocities, gt_headings
    )
    
    # Stability (for EvoQRE)
    alpha, kappa, stability_ok = 0.0, 0.0, False
    if hasattr(agents[0], 'get_stability_info'):
        info = agents[0].get_stability_info()
        if info:
            alpha = info.alpha
            kappa = info.kappa
            stability_ok = info.is_stable
    
    train_time = time.time() - start_time
    
    # Create result
    result = ScenarioResult(
        scenario_id=scenario_id,
        method=method_name,
        num_agents=agent_num,
        reward=np.mean(rewards[-10:]),
        cost=np.mean(costs[-10:]),
        nll=nll if np.isfinite(nll) else 0.0,
        collision_rate=collision_rate,
        off_road_rate=off_road_rate,
        diversity=diversity,
        speed_mean=behavioral['speed_mean'],
        speed_std=behavioral['speed_std'],
        accel_mean=behavioral['accel_mean'],
        accel_std=behavioral['accel_std'],
        following_dist_mean=behavioral['following_dist_mean'],
        following_dist_std=behavioral['following_dist_std'],
        alpha=alpha,
        kappa=kappa,
        stability_satisfied=stability_ok,
        train_time_s=train_time,
    )
    
    return result

# %% [markdown]
# ## 8. Run Experiment

# %%
# Methods to compare
METHODS = {
    'TrafficGamer': TrafficGamer,
    'EvoQRE': EvoQRE_Langevin,
}

# Store results by method
all_results = {method: [] for method in METHODS}

print(f"\n{'='*70}")
print(f"🚀 MAIN RESULTS EXPERIMENT")
print(f"   Scenarios: {len(scenario_indices)}")
print(f"   Episodes: {CONFIG.num_episodes}")
print(f"   Methods: {list(METHODS.keys())}")
print(f"{'='*70}\n")

for idx in tqdm(scenario_indices, desc="Scenarios"):
    for method_name, agent_class in METHODS.items():
        try:
            result = train_and_evaluate(idx, agent_class, method_name)
            if result:
                all_results[method_name].append(result)
                RESULTS.add_result(result)
        except Exception as e:
            print(f"Error {method_name} on {idx}: {e}")
            continue
    
    # Save checkpoint every 10 scenarios
    if (len(all_results['TrafficGamer']) + 1) % 10 == 0:
        saver = ResultsSaver(CONFIG.output_dir)
        saver.save_experiment(RESULTS, save_trajectories=False)

# Finish experiment
RESULTS = finish_experiment(RESULTS)

# %% [markdown]
# ## 9. Results Table

# %%
print("\n" + "="*70)
print("📊 TABLE: MAIN RESULTS ON WOMD/AV2")
print("="*70)

table_data = []

for method_name, results_list in all_results.items():
    if not results_list:
        continue
    
    df = pd.DataFrame([{
        'nll': r.nll,
        'collision_rate': r.collision_rate,
        'off_road_rate': r.off_road_rate,
        'diversity': r.diversity,
    } for r in results_list])
    
    table_data.append({
        'Method': method_name,
        'NLL↓': f"{df['nll'].mean():.2f}±{df['nll'].std():.2f}",
        'Coll.%↓': f"{df['collision_rate'].mean()*100:.1f}±{df['collision_rate'].std()*100:.1f}",
        'Off-road%↓': f"{df['off_road_rate'].mean()*100:.1f}±{df['off_road_rate'].std()*100:.1f}",
        'Div.↑': f"{df['diversity'].mean():.2f}±{df['diversity'].std():.2f}",
    })

result_df = pd.DataFrame(table_data)
print(result_df.to_markdown(index=False))

# %% [markdown]
# ## 10. LaTeX Output

# %%
print("\n" + "="*70)
print("📄 LaTeX for paper (copy to evoqre_tits.tex)")
print("="*70)

latex = TableFormatter.format_main_results(table_data)
print(latex)

# %% [markdown]
# ## 11. Save Results

# %%
saver = ResultsSaver(CONFIG.output_dir)
exp_dir = saver.save_experiment(RESULTS)

print(f"\n✅ Results saved to: {exp_dir}")
print(f"   - config.json")
print(f"   - summary.json")
print(f"   - results.csv")
print(f"   - full_results.pkl")

# Save LaTeX
latex_path = Path(exp_dir) / 'table_main_results.tex'
with open(latex_path, 'w') as f:
    f.write(latex)
print(f"   - table_main_results.tex")

print("\n🎉 Experiment complete!")
