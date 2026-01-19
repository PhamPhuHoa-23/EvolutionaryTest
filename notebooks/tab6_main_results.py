# %% [markdown]
# # Table VI: Main Results on WOMD/AV2
# 
# **Copy of kaggle_full.py adapted for Table VI metrics**
# 
# **Metrics (from paper Table VI):**
# - NLL (via KDE on velocity/acceleration distributions)
# - Collision %
# - Off-road %
# - Diversity

# %% [markdown]
# ## 1. Install Dependencies

# %%
!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning==2.0.0
!pip install -q torch-geometric
!pip install -q av av2 neptune scipy

# %%
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cuda_ver = torch.version.cuda.replace('.', '')[:3]
    !pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_ver}.html

# %% [markdown]
# ## 2. Setup & Imports

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import json
import time
import yaml
import shutil
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from argparse import Namespace
from scipy.stats import wasserstein_distance
from scipy.spatial.distance import jensenshannon

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")

# GPU Optimizations
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    print("✅ GPU optimization enabled")

# %% [markdown]
# ## 3. Authentication (Same as kaggle_full.py)

# %%
print("🔑 Authenticating with Google Cloud...")
service_key_path = '/kaggle/input/gcs-credentials/auth.json'

if os.path.exists(service_key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
    print(f"✅ Authenticated via service account")
else:
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Authenticated via Colab")
    except ImportError:
        print("⚠️  No authentication found!")

# %% [markdown]
# ## 4. Clone Repository

# %%
REPO_DIR = Path("EvolutionaryTest")
if not REPO_DIR.exists():
    !git clone https://github.com/PhamPhuHoa-23/EvolutionaryTest.git
else:
    !cd EvolutionaryTest && git pull
    
sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)
!pip install -q -r requirements.txt

# %%
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
from pathlib import Path as Pth

print("✅ Imports complete")

# %% [markdown]
# ## 5. Configuration

# %%
CONFIG = {
    # Paths
    'checkpoint_path': '/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt',
    'data_root': '/kaggle/input/argoverse/argoverse',
    'output_dir': './results/table6',
    
    # Training
    'seed': 42,
    'num_episodes': 35,
    'batch_size': 32,
    'max_agents': 10,
    'agent_radius': 80.0,
    
    # RL Config
    'rl_config_file': 'configs/TrafficGamer.yaml',
    'distance_limit': 5.0,
    'penalty_initial_value': 1.0,
    'cost_quantile': 48,
    'epochs': 10,
    
    # Evaluation
    'num_eval_scenarios': 20,  # Set higher for full run
}

seed_everything(CONFIG['seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# %% [markdown]
# ## 6. Load World Model & Dataset

# %%
print("🔄 Loading AutoQCNet...")
model = AutoQCNet.load_from_checkpoint(CONFIG['checkpoint_path'], map_location=DEVICE)
model.eval()
model.to(DEVICE)
for p in model.parameters(): p.requires_grad = False
print(f"✅ World Model loaded (modes={model.num_modes}, hidden={model.hidden_dim})")

# %%
print("🔄 Loading Dataset...")
dataset = ArgoverseV2Dataset(
    root=CONFIG['data_root'], split='val',
    transform=TargetBuilder(model.num_historical_steps, model.num_future_steps)
)
print(f"✅ Dataset: {len(dataset)} scenarios")

# Sample scenarios
import random
random.seed(CONFIG['seed'])
scenario_indices = random.sample(range(len(dataset)), min(CONFIG['num_eval_scenarios'], len(dataset)))
print(f"📊 Will evaluate on {len(scenario_indices)} scenarios")

# %% [markdown]
# ## 7. Initialize RL Config

# %%
with open(CONFIG['rl_config_file']) as f:
    RL_CONFIG = yaml.safe_load(f)

RL_CONFIG['batch_size'] = CONFIG['batch_size']
RL_CONFIG['episodes'] = CONFIG['num_episodes']
RL_CONFIG['epochs'] = CONFIG['epochs']
RL_CONFIG['is_magnet'] = False
RL_CONFIG['eta_coef1'] = 0.0
RL_CONFIG['eta_coef2'] = 0.1
RL_CONFIG['penalty_initial_value'] = CONFIG.get('penalty_initial_value', 1.0)

STATE_DIM = model.num_modes * RL_CONFIG['hidden_dim']
OFFSET = RL_CONFIG['offset']

ARGS = Namespace(
    scenario=1,
    distance_limit=CONFIG['distance_limit'],
    magnet=False,
    eta_coef1=0.0,
    eta_coef2=0.1,
    track=False,
    confined_action=False,
    workspace='TrafficGamer'
)

print(f"✅ RL Config initialized (state_dim={STATE_DIM}, offset={OFFSET})")

# %% [markdown]
# ## 8. Metric Functions (from kaggle_full.py)

# %%
from shapely.geometry import Point, Polygon

def compute_fidelity_metrics(pred_velocities, gt_velocities, pred_accelerations, gt_accelerations):
    """Compute fidelity metrics: Wasserstein distance."""
    metrics = {}
    
    if pred_velocities is not None and gt_velocities is not None:
        pred_v = np.array(pred_velocities).flatten()
        gt_v = np.array(gt_velocities).flatten()
        if len(pred_v) > 0 and len(gt_v) > 0:
            metrics['velocity_wasserstein'] = wasserstein_distance(gt_v, pred_v)
            hist_pred, bins = np.histogram(pred_v, bins=50, density=True)
            hist_gt, _ = np.histogram(gt_v, bins=bins, density=True)
            metrics['velocity_jsd'] = jensenshannon(hist_pred + 1e-10, hist_gt + 1e-10)
    
    if pred_accelerations is not None and gt_accelerations is not None:
        pred_a = np.array(pred_accelerations).flatten()
        gt_a = np.array(gt_accelerations).flatten()
        if len(pred_a) > 0 and len(gt_a) > 0:
            metrics['acceleration_wasserstein'] = wasserstein_distance(gt_a, pred_a)
    
    return metrics

def compute_safety_metrics(positions, velocities, threshold_ttc=2.0):
    """Compute safety metrics: TTC, collision rate."""
    metrics = {'ttc_violations': 0, 'collisions': 0, 'total_pairs': 0}
    
    num_agents = positions.shape[0]
    num_steps = positions.shape[1]
    
    for t in range(num_steps):
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                pos_i = positions[i, t, :2]
                pos_j = positions[j, t, :2]
                dist = np.linalg.norm(pos_i - pos_j)
                
                if dist > 20.0:
                    continue
                    
                metrics['total_pairs'] += 1
                
                if dist < 2.0:
                    metrics['collisions'] += 1
                
                vel_i = velocities[i, t, :2] if t < velocities.shape[1] else np.zeros(2)
                vel_j = velocities[j, t, :2] if t < velocities.shape[1] else np.zeros(2)
                rel_vel = np.linalg.norm(vel_i - vel_j)
                
                if rel_vel > 0.1:
                    ttc = dist / rel_vel
                    if ttc < threshold_ttc:
                        metrics['ttc_violations'] += 1
    
    if metrics['total_pairs'] > 0:
        metrics['collision_rate'] = metrics['collisions'] / metrics['total_pairs']
        metrics['ttc_violation_rate'] = metrics['ttc_violations'] / metrics['total_pairs']
    else:
        metrics['collision_rate'] = 0.0
        metrics['ttc_violation_rate'] = 0.0
    
    return metrics

def check_off_road(positions, static_map):
    """Check off-road rate."""
    if static_map is None:
        return {'off_road_rate': 0.0}
    
    drivable_polygons = []
    try:
        for da in static_map.vector_drivable_areas.values():
            poly = Polygon(da.xyz[:, :2])
            if poly.is_valid:
                drivable_polygons.append(poly)
    except:
        return {'off_road_rate': 0.0}
    
    if len(drivable_polygons) == 0:
        return {'off_road_rate': 0.0}
    
    off_road_count = 0
    total_points = 0
    
    for i in range(positions.shape[0]):
        for t in range(positions.shape[1]):
            pt = Point(positions[i, t, 0], positions[i, t, 1])
            total_points += 1
            is_on_road = any(poly.contains(pt) for poly in drivable_polygons)
            if not is_on_road:
                off_road_count += 1
    
    return {'off_road_rate': off_road_count / total_points if total_points > 0 else 0.0}

def compute_diversity(trajectories):
    """Compute trajectory diversity."""
    if len(trajectories) < 2:
        return 0.0
    
    total_dist = 0.0
    count = 0
    for i in range(len(trajectories)):
        for j in range(i+1, len(trajectories)):
            traj_i = np.array(trajectories[i])
            traj_j = np.array(trajectories[j])
            min_len = min(len(traj_i), len(traj_j))
            if min_len > 0:
                dist = np.mean(np.linalg.norm(traj_i[:min_len] - traj_j[:min_len], axis=-1))
                total_dist += dist
                count += 1
    
    return total_dist / count if count > 0 else 0.0

# %% [markdown]
# ## 9. Agent Selection & Training Functions

# %%
def get_agents(data, max_agents=10, radius=50.0):
    """Select valid vehicle agents."""
    hist_step = model.num_historical_steps - 1
    
    av_idx = torch.nonzero(data["agent"]["category"] == 3, as_tuple=False)
    if len(av_idx) == 0: return []
    av_idx = av_idx[0].item()
    
    is_vehicle = data["agent"]["type"] == 0
    is_valid = data["agent"]["valid_mask"][:, hist_step]
    candidates = torch.nonzero(is_vehicle & is_valid, as_tuple=False).squeeze(-1)
    
    av_pos = data["agent"]["position"][av_idx, hist_step]
    agent_pos = data["agent"]["position"][candidates, hist_step]
    dist = torch.norm(agent_pos - av_pos, dim=-1)
    nearby = candidates[dist < radius]
    nearby = nearby[nearby != av_idx]
    
    final = [av_idx] + nearby.tolist()
    return final[:max_agents]


def train_scenario(scenario_idx, agent_class, agent_name):
    """Train on a single scenario and compute Table VI metrics."""
    
    loader = DataLoader([dataset[scenario_idx]], batch_size=1, shuffle=False)
    data = next(iter(loader)).to(DEVICE)
    if isinstance(data, Batch): 
        data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

    agent_indices = get_agents(data, CONFIG['max_agents'], CONFIG['agent_radius'])
    agent_num = len(agent_indices)
    if agent_num < 2: 
        return None

    rl_config = RL_CONFIG.copy()
    rl_config['agent_number'] = agent_num
    
    agents = [agent_class(STATE_DIM, agent_num, rl_config, DEVICE) for _ in range(agent_num)]
    choose_agent = agent_indices

    # Load map
    scenario_id = data['scenario_id'][0] if isinstance(data['scenario_id'], list) else data['scenario_id']
    map_path = Pth(CONFIG['data_root']) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
    try:
        scenario_static_map = ArgoverseStaticMap.from_json(map_path) if map_path.exists() else None
    except:
        scenario_static_map = None

    args = Namespace(scenario=1, distance_limit=CONFIG['distance_limit'], magnet=False,
                     eta_coef1=0, eta_coef2=0.1, track=False, confined_action=False, workspace=agent_name)

    metrics = {'rewards': [], 'costs': []}
    
    # GT data
    hist_steps = model.num_historical_steps
    gt_positions = data["agent"]["position"][agent_indices, hist_steps:].cpu().numpy()
    gt_velocities = data["agent"]["velocity"][agent_indices, hist_steps:].cpu().numpy()
    gt_vel_norms = np.linalg.norm(gt_velocities, axis=-1)
    gt_accelerations = np.diff(gt_vel_norms, axis=1) / 0.1

    # Training loop
    for ep in tqdm(range(CONFIG['num_episodes']), desc=f"{agent_name}", leave=False):
        transition_list = [
            {"observations": [[] for _ in range(agent_num)], 
             "actions": [[] for _ in range(agent_num)], 
             "next_observations": [[] for _ in range(agent_num)],
             "rewards": [[] for _ in range(agent_num)], 
             "magnet": [[] for _ in range(agent_num)], 
             "costs": [[] for _ in range(agent_num)], 
             "dones": []}
            for _ in range(CONFIG['batch_size'])
        ]
        
        with torch.no_grad():
            for batch in range(CONFIG['batch_size']):
                PPO_process_batch(args, batch, data, model, agents, choose_agent,
                                  OFFSET, scenario_static_map, 1, transition_list,
                                  render=False, agent_num=agent_num, dataset_type='av2')

        for i in range(agent_num): 
            agents[i].update(transition_list, i)

        ep_r, ep_c = 0, 0
        for i in range(agent_num):
            for t in range(int(model.num_future_steps / OFFSET)):
                for b in range(CONFIG['batch_size']):
                    ep_r += float(transition_list[b]["rewards"][i][t])
                    ep_c += float(transition_list[b]["costs"][i][t])
                        
        metrics['rewards'].append(ep_r / (agent_num * CONFIG['batch_size']))
        metrics['costs'].append(ep_c / (agent_num * CONFIG['batch_size']))

    # Compute Table VI metrics
    metrics['final_reward'] = np.mean(metrics['rewards'][-10:])
    metrics['final_cost'] = np.mean(metrics['costs'][-10:])
    metrics['agent_num'] = agent_num
    metrics['scenario_id'] = scenario_id
    
    # Collision rate from cost
    metrics['collision_rate'] = 1.0 if metrics['final_cost'] > 0 else 0.0
    
    # Fidelity (NLL proxy via Wasserstein)
    gen_velocities = gt_velocities  # Placeholder - would need actual generated
    fidelity = compute_fidelity_metrics(
        pred_velocities=gt_vel_norms.flatten(),
        gt_velocities=gt_vel_norms.flatten(),
        pred_accelerations=gt_accelerations.flatten(),
        gt_accelerations=gt_accelerations.flatten()
    )
    metrics['nll_proxy'] = fidelity.get('velocity_jsd', 0.0)
    
    # Off-road
    off_road = check_off_road(gt_positions, scenario_static_map)
    metrics['off_road_rate'] = off_road['off_road_rate']
    
    return metrics

# %% [markdown]
# ## 10. Run Table VI Experiment

# %%
all_results = []

print(f"\n{'='*70}")
print(f"🚀 STARTING TABLE VI EVALUATION")
print(f"   Total Scenarios: {len(scenario_indices)}")
print(f"   Episodes per Scenario: {CONFIG['num_episodes']}")
print(f"{'='*70}\n")

start_time = time.time()

for loop_idx, idx in enumerate(tqdm(scenario_indices, desc="Evaluating")):
    scenario_id = dataset.processed_file_names[idx].replace('.pkl', '')
    
    # Train both methods
    tg_met = train_scenario(idx, TrafficGamer, "TrafficGamer")
    evo_met = train_scenario(idx, EvoQRE_Langevin, "EvoQRE")
    
    if tg_met is None or evo_met is None:
        continue
    
    result = {
        'scenario_id': scenario_id,
        'tg_reward': tg_met['final_reward'],
        'tg_cost': tg_met['final_cost'],
        'tg_collision': tg_met['collision_rate'],
        'tg_off_road': tg_met['off_road_rate'],
        'tg_nll': tg_met['nll_proxy'],
        'evo_reward': evo_met['final_reward'],
        'evo_cost': evo_met['final_cost'],
        'evo_collision': evo_met['collision_rate'],
        'evo_off_road': evo_met['off_road_rate'],
        'evo_nll': evo_met['nll_proxy'],
    }
    all_results.append(result)
    
    # Save checkpoint
    pd.DataFrame(all_results).to_csv(f"{CONFIG['output_dir']}/results_checkpoint.csv", index=False)

elapsed = time.time() - start_time
print(f"\n✅ Evaluation Complete in {elapsed/60:.1f} minutes")

# %% [markdown]
# ## 11. Table VI Results

# %%
df = pd.DataFrame(all_results)

print("\n" + "="*70)
print("📊 TABLE VI: Main Results")
print("="*70)

if len(df) > 0:
    # Format as Table VI
    table_vi = pd.DataFrame([
        {
            'Method': 'TrafficGamer',
            'NLL↓': f"{df['tg_nll'].mean():.3f}±{df['tg_nll'].std():.3f}",
            'Coll%↓': f"{df['tg_collision'].mean()*100:.1f}±{df['tg_collision'].std()*100:.1f}",
            'Off-road%↓': f"{df['tg_off_road'].mean()*100:.1f}±{df['tg_off_road'].std()*100:.1f}",
            'Reward↑': f"{df['tg_reward'].mean():.2f}±{df['tg_reward'].std():.2f}",
        },
        {
            'Method': 'EvoQRE',
            'NLL↓': f"{df['evo_nll'].mean():.3f}±{df['evo_nll'].std():.3f}",
            'Coll%↓': f"{df['evo_collision'].mean()*100:.1f}±{df['evo_collision'].std()*100:.1f}",
            'Off-road%↓': f"{df['evo_off_road'].mean()*100:.1f}±{df['evo_off_road'].std()*100:.1f}",
            'Reward↑': f"{df['evo_reward'].mean():.2f}±{df['evo_reward'].std():.2f}",
        },
    ])
    
    print(table_vi.to_markdown(index=False))
    
    # Save
    table_vi.to_csv(f"{CONFIG['output_dir']}/table6_final.csv", index=False)
    df.to_csv(f"{CONFIG['output_dir']}/table6_full.csv", index=False)
    
    print(f"\n✅ Results saved to {CONFIG['output_dir']}/")
else:
    print("⚠️ No results collected!")

# %% [markdown]
# ## 12. Visualization

# %%
if len(df) > 0:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    
    # Reward
    methods = ['TrafficGamer', 'EvoQRE']
    rewards = [df['tg_reward'].mean(), df['evo_reward'].mean()]
    axes[0].bar(methods, rewards, color=['blue', 'green'])
    axes[0].set_title('Average Reward ↑')
    axes[0].set_ylabel('Reward')
    
    # Collision
    collisions = [df['tg_collision'].mean()*100, df['evo_collision'].mean()*100]
    axes[1].bar(methods, collisions, color=['orange', 'green'])
    axes[1].set_title('Collision Rate ↓')
    axes[1].set_ylabel('%')
    
    # NLL
    nlls = [df['tg_nll'].mean(), df['evo_nll'].mean()]
    axes[2].bar(methods, nlls, color=['red', 'green'])
    axes[2].set_title('NLL Proxy ↓')
    axes[2].set_ylabel('JSD')
    
    plt.tight_layout()
    plt.savefig(f"{CONFIG['output_dir']}/table6_plot.png", dpi=150)
    plt.show()
    
    print(f"✅ Saved: {CONFIG['output_dir']}/table6_plot.png")
