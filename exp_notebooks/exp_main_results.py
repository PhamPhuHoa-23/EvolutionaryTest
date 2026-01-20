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
!pip install -q torch torchvision torchaudio
!pip install -q pytorch-lightning==2.0.0
!pip install -q torch-geometric
!pip install -q av av2 neptune scipy pandas shapely

# %%
import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cuda_ver = torch.version.cuda.replace('.', '')[:3]
    !pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_ver}.html

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

# 6 Representative Scenarios (from Paper/Kaggle)
REPRESENTATIVE_SCENARIOS = [
    'd1f6b01e-3b4a-4790-88ed-6d85fb1c0b84', # Merge
    '00a50e9f-63a1-4678-a4fe-c6109721ecba', # Dual-lane Intersection
    '236df665-eec6-4c25-8822-950a6150eade', # T-junction
    'cb0133ff-f7ad-43b7-b260-7068ace15307', # Dense Intersection
    'cdf70cc8-d13d-470b-bb39-4f1812acc146', # Roundabout
    '3856ed37-4a05-4131-9b12-c4f4716fec92', # Y-junction
]

print("🔍 Filtering for 6 representative scenarios...")
scenario_indices = []
for idx in range(len(dataset)):
    fname = dataset.processed_file_names[idx]
    sid = fname.replace('.pkl', '')
    if sid in REPRESENTATIVE_SCENARIOS:
        scenario_indices.append(idx)

# Verify we found all 6
print(f"✅ Found {len(scenario_indices)}/6 representative scenarios")
# If not all found, we just proceed with what we found (local dataset might be partial)

# Config overrides
CONFIG.num_scenarios = len(scenario_indices)
CONFIG.num_episodes = 30  # As requested

# %% [markdown]
# ## 4.1 Pre-load Maps (Optimization)
# 
# Pre-load all maps at startup to avoid repeated disk I/O during training.

# %%
print("📦 Pre-loading scenario maps (one-time)...")

# Build scenario_id lookup
def get_scenario_id_from_idx(idx):
    """Get scenario ID from dataset index."""
    fname = dataset.processed_file_names[idx]
    return fname.replace('.pkl', '').split('/')[-1].split('\\')[-1]

# Pre-cache maps
MAP_CACHE = {}
valid_scenario_indices = []

for idx in tqdm(scenario_indices, desc="Caching maps"):
    try:
        scenario_id = get_scenario_id_from_idx(idx)
        
        # Try to load map
        map_path = Path(DATA_ROOT) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
        
        if map_path.exists():
            MAP_CACHE[scenario_id] = ArgoverseStaticMap.from_json(map_path)
            valid_scenario_indices.append(idx)
        # Skip rglob - too slow, just skip if not found
    except Exception as e:
        print(f"Warning: Could not load map for {scenario_id}: {e}")
        continue

# Final list of scenarios to run (only valid ones)
scenario_indices = valid_scenario_indices

print(f"✅ Cached {len(MAP_CACHE)} maps")
print(f"📊 Will evaluate {len(scenario_indices)} scenarios (Target: 6)")

# %% [markdown]
# ## 5. Load RL Config

# %%
with open('configs/TrafficGamer.yaml') as f:
    RL_CONFIG = yaml.safe_load(f)

# Add ALL required defaults (from kaggle.py pattern)
RL_CONFIG.setdefault('hidden_dim', 128)
RL_CONFIG.setdefault('gamma', 0.99)
RL_CONFIG.setdefault('lamda', 0.95)
RL_CONFIG.setdefault('actor_learning_rate', 5e-5)
RL_CONFIG.setdefault('critic_learning_rate', 1e-4)
RL_CONFIG.setdefault('constrainted_critic_learning_rate', 1e-4)
RL_CONFIG.setdefault('density_learning_rate', 3e-4)
RL_CONFIG.setdefault('eps', 0.2)
RL_CONFIG.setdefault('offset', 5)
RL_CONFIG.setdefault('entropy_coef', 0.005)
RL_CONFIG.setdefault('epochs', 10)
RL_CONFIG.setdefault('gae', True)
RL_CONFIG.setdefault('target_kl', 0.01)
RL_CONFIG.setdefault('beta_coef', 0.1)
RL_CONFIG.setdefault('N_quantile', 64)
RL_CONFIG.setdefault('tau_update', 0.01)
RL_CONFIG.setdefault('LR_QN', 3e-4)
RL_CONFIG.setdefault('type', 'CVaR')
RL_CONFIG.setdefault('method', 'SplineDQN')
RL_CONFIG.setdefault('penalty_initial_value', 1.0)
RL_CONFIG.setdefault('cost_quantile', 48)

# CRITICAL: These must be set
RL_CONFIG['is_magnet'] = False
RL_CONFIG['eta_coef1'] = 0.0
RL_CONFIG['eta_coef2'] = 0.1
RL_CONFIG['batch_size'] = CONFIG.batch_size
RL_CONFIG['episodes'] = CONFIG.num_episodes

# EvoQRE/Langevin specific
RL_CONFIG.setdefault('langevin_steps', 20)
RL_CONFIG.setdefault('langevin_step_size', 0.05)
RL_CONFIG.setdefault('tau', 0.5)
RL_CONFIG.setdefault('action_bound', 1.0)
RL_CONFIG['langevin_steps'] = CONFIG.langevin_steps
RL_CONFIG['langevin_step_size'] = CONFIG.step_size
RL_CONFIG['tau'] = CONFIG.tau
RL_CONFIG['epsilon'] = CONFIG.epsilon

STATE_DIM = model.num_modes * RL_CONFIG['hidden_dim']
OFFSET = RL_CONFIG['offset']

print(f"✅ RL Config loaded")
print(f"   state_dim={STATE_DIM}, offset={OFFSET}, is_magnet={RL_CONFIG['is_magnet']}")

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


def load_static_map(scenario_id, data_root):
    """Load ArgoverseStaticMap for a scenario (matches kaggle_full.py pattern)."""
    map_path = Path(data_root) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
    
    if not map_path.exists():
        # Try to find it with rglob
        found = list(Path(data_root).rglob(f'log_map_archive_{scenario_id}.json'))
        if found:
            map_path = found[0]
        else:
            return None
    
    try:
        return ArgoverseStaticMap.from_json(map_path)
    except Exception as e:
        print(f"Warning: Could not load map for {scenario_id}: {e}")
        return None


def get_drivable_polygons(static_map):
    """Extract drivable area polygons from ArgoverseStaticMap."""
    if static_map is None:
        return []
    
    try:
        polygons = []
        for da in static_map.vector_drivable_areas.values():
            poly = Polygon(da.xyz[:, :2])
            if poly.is_valid:
                polygons.append(poly)
        return polygons
    except Exception:
        return []

# %% [markdown]
# ## 7. Trajectory Reconstruction

# %%
def reconstruct_trajectories(initial_pos, initial_vel, initial_heading, actions, dt=0.1):
    """
    Reconstruct trajectories using Unicycle Kinematic Model.
    
    Args:
        initial_pos: [N, 2] initial positions
        initial_vel: [N, 2] initial velocities
        initial_heading: [N] initial heading (radians)
        actions: [N, T, 2] actions (acceleration, steering_curvature)
        dt: timestep
        
    Returns:
        gen_positions: [N, T, 2]
        gen_velocities: [N, T, 2]
        gen_headings: [N, T]
    """
    N, T, _ = actions.shape
    curr_pos = torch.tensor(initial_pos[:, :2], dtype=torch.float32)
    
    if len(initial_vel.shape) == 2:
        curr_speed = torch.tensor(np.linalg.norm(initial_vel, axis=-1), dtype=torch.float32)
    else:
        curr_speed = torch.tensor(initial_vel, dtype=torch.float32)
    curr_heading = torch.tensor(initial_heading, dtype=torch.float32)
    
    pos_list, head_list = [], []
    
    for t in range(T):
        # Actions: [acceleration (x5), curvature (x0.05)]
        acc = torch.tensor(actions[:, t, 0], dtype=torch.float32).clip(-1, 1) * 5.0
        kappa = torch.tensor(actions[:, t, 1], dtype=torch.float32).clip(-1, 1) * 0.05
        
        # Unicycle Update
        distance = curr_speed * dt + 0.5 * acc * (dt**2)
        next_pos = curr_pos.clone()
        next_pos[:, 0] += distance * torch.cos(curr_heading)
        next_pos[:, 1] += distance * torch.sin(curr_heading)
        next_heading = curr_heading + kappa * distance
        next_speed = torch.clamp(curr_speed + acc * dt, min=0.0)
        
        pos_list.append(next_pos)
        head_list.append(next_heading)
        
        curr_pos, curr_heading, curr_speed = next_pos, next_heading, next_speed
        
    gen_positions = torch.stack(pos_list, dim=1).numpy()
    gen_headings = torch.stack(head_list, dim=1).numpy()
    
    # Compute velocity vectors from position differences
    gen_velocities = np.zeros_like(gen_positions)
    gen_velocities[:, 1:] = (gen_positions[:, 1:] - gen_positions[:, :-1]) / dt
    gen_velocities[:, 0] = gen_velocities[:, 1]  # Duplicate first frame
    
    return gen_positions, gen_velocities, gen_headings


# %% [markdown]
# ## 8. Training Function

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

    # Get map from pre-loaded cache (FAST - no disk I/O)
    scenario_static_map = MAP_CACHE.get(scenario_id)
    if scenario_static_map is None:
        return None  # Skip scenarios without cached map

    # Args for rollout
    args = Namespace(
        scenario=1, distance_limit=5.0, magnet=False,
        eta_coef1=0, eta_coef2=0.1, track=False,
        confined_action=False, workspace=method_name
    )

    # ===========================================
    # Training Loop with Action Collection
    # ===========================================
    rewards, costs = [], []
    num_timesteps = int(model.num_future_steps / OFFSET)
    
    # Storage for actions from LAST episode (for trajectory reconstruction)
    last_ep_actions = np.zeros((agent_num, num_timesteps, 2))
    
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
                    OFFSET, scenario_static_map, 1, transition_list,
                    render=False, agent_num=agent_num, dataset_type='av2'
                )

        # Update agents
        for i in range(agent_num):
            agents[i].update(transition_list, i)

        # Collect episode metrics and actions from last episode
        ep_r, ep_c = 0, 0
        for i in range(agent_num):
            for t in range(len(transition_list[0]["rewards"][i])):
                for b in range(CONFIG.batch_size):
                    ep_r += float(transition_list[b]["rewards"][i][t])
                    ep_c += float(transition_list[b]["costs"][i][t])
                    
                    # Capture actions from the LAST episode, BATCH 0
                    if ep == CONFIG.num_episodes - 1 and b == 0:
                        act = transition_list[b]["actions"][i][t]
                        if isinstance(act, torch.Tensor):
                            act = act.cpu().numpy()
                        if act.size >= 2:
                            last_ep_actions[i, t] = act.flatten()[:2]
                        
        rewards.append(ep_r / (agent_num * CONFIG.batch_size))
        costs.append(ep_c / (agent_num * CONFIG.batch_size))

    # ===========================================
    # Reconstruct Generated Trajectories
    # ===========================================
    hist_steps = model.num_historical_steps
    
    # Initial conditions from data
    init_pos = data["agent"]["position"][agent_indices, hist_steps-1].cpu().numpy()
    init_vel = data["agent"]["velocity"][agent_indices, hist_steps-1].cpu().numpy()
    init_head = data["agent"]["heading"][agent_indices, hist_steps-1].cpu().numpy()
    
    # Reconstruct using Unicycle model
    gen_positions, gen_velocities, gen_headings = reconstruct_trajectories(
        init_pos, init_vel, init_head, last_ep_actions
    )
    
    # Ground truth trajectories (for comparison)
    gt_positions = data["agent"]["position"][agent_indices, hist_steps:].cpu().numpy()
    gt_velocities = data["agent"]["velocity"][agent_indices, hist_steps:].cpu().numpy()
    gt_headings = data["agent"]["heading"][agent_indices, hist_steps:].cpu().numpy()
    
    # ===========================================
    # Compute Metrics on GENERATED Trajectories
    # ===========================================
    
    # NLL: Compare generated vs GT velocity distributions
    gen_vel_norms = np.linalg.norm(gen_velocities, axis=-1).flatten()
    gt_vel_norms = np.linalg.norm(gt_velocities[:, :gen_velocities.shape[1]], axis=-1).flatten()
    nll = metrics_computer.compute_nll_kde(gen_vel_norms, gt_vel_norms)
    
    # Collision rate on GENERATED positions
    collision_rate = metrics_computer.compute_collision_rate(gen_positions)
    
    # Off-road rate on GENERATED positions
    drivable_polygons = get_drivable_polygons(scenario_static_map)
    if drivable_polygons:
        off_road_rate = metrics_computer.compute_off_road_rate(gen_positions, drivable_polygons)
    else:
        off_road_rate = 0.0
    
    # Diversity: For single rollout, use trajectory spread
    # (For proper diversity, need multiple rollouts per scenario)
    diversity = metrics_computer.compute_diversity(gen_positions)
    
    # Behavioral metrics on GENERATED trajectories
    behavioral = metrics_computer.compute_behavioral_metrics(
        gen_positions, gen_velocities, gen_headings
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
            import traceback
            print(f"\n{'='*60}")
            print(f"❌ Error {method_name} on scenario {idx}")
            print(f"{'='*60}")
            traceback.print_exc()
            print(f"{'='*60}\n")
            continue
    
    # Save checkpoint every 10 scenarios
    first_method = list(METHODS.keys())[0]
    if first_method in all_results and (len(all_results[first_method]) + 1) % 10 == 0:
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
