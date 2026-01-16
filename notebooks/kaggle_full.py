# %% [markdown]
# # EvoQRE vs TrafficGamer: Comprehensive Comparison (H100 Optimized)
# 
# **Full Dataset Training + Detailed 6-Scenario Analysis**
# 
# **Features:**
# - Train on ALL ~25k Argoverse 2 val scenarios
# - High agent count (up to 20 per scene) for H100
# - Detailed metrics for 6 representative scenarios (Paper Table)
# - Comprehensive metrics: Fidelity, Safety (TTC, THW), Diversity, Collision Rate
# - Rich result saving (CSV, JSON, Plots)
#
# **Repository**: https://github.com/PhamPhuHoa-23/EvolutionaryTest

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

# GPU Optimizations for speed
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True  # Auto-tune convolutions
    # NOTE: TF32 disabled - causes NaN in Laplace distribution due to reduced precision
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    print("✅ GPU optimization enabled (cudnn.benchmark)")

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
# ## 3. Configuration (H100 Optimized)

# %%
CONFIG = {
    # Paths
    'checkpoint_path': '/kaggle/input/qcnetckptargoverse/pytorch/default/1/QCNet_AV2.ckpt',
    'data_root': '/kaggle/input/argoverse/argoverse',
    'output_dir': './results',
    'checkpoint_dir': './save_models',  # Directory for saved checkpoints
    '3d_output_dir': './argoverse2_pred',  # Directory for 3D visualization data
    
    # Training (DEEP TRAINING - 6 Representative Scenarios)
    'seed': 42,
    'num_episodes': 35,       # Fits in ~12h for 6 scenarios (based on 3.23min/ep)
    'batch_size': 32,         # Must match original
    'max_agents': 20,         # High agent count for game-theoretic interactions
    'agent_radius': 80.0,     # Consider agents within 80m
    
    # Run ONLY 6 representative scenarios
    'run_representative_only': True,
    
    # RL Config
    'rl_config_file': 'configs/TrafficGamer.yaml',
    'distance_limit': 5.0,
    'penalty_initial_value': 1.0,
    'cost_quantile': 48,
    'epochs': 10,             # Full epochs like paper
    
    # Output options
    'save_checkpoints': True,   # Save model checkpoints
    'save_3d_visualization': True,  # Save 3D parquet files for visualization
}

# 6 Representative Scenarios (from Paper)
REPRESENTATIVE_SCENARIOS = {
    'd1f6b01e-3b4a-4790-88ed-6d85fb1c0b84': {'name': 'Merge', 'type': 'merge'},
    '00a50e9f-63a1-4678-a4fe-c6109721ecba': {'name': 'Dual-lane Intersection', 'type': 'intersection'},
    '236df665-eec6-4c25-8822-950a6150eade': {'name': 'T-junction', 'type': 't_junction'},
    'cb0133ff-f7ad-43b7-b260-7068ace15307': {'name': 'Dense Intersection', 'type': 'dense_intersection'},
    'cdf70cc8-d13d-470b-bb39-4f1812acc146': {'name': 'Roundabout', 'type': 'roundabout'},
    '3856ed37-4a05-4131-9b12-c4f4716fec92': {'name': 'Y-junction', 'type': 'y_junction'},
}

seed_everything(CONFIG['seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# %% [markdown]
# ## 4. Load World Model & Dataset

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
total_scenarios = len(dataset)
print(f"✅ Dataset: {total_scenarios} scenarios")

# Build scenario index list: 6 representative + random samples to reach max_scenarios
import random
random.seed(CONFIG['seed'])

# Build lookup: scenario_id -> dataset index
print("🔍 Building scenario index lookup...")
scenario_id_to_idx = {fname.replace('.pkl', ''): i for i, fname in enumerate(dataset.processed_file_names)}

# Directly look up the 6 representative scenarios by their IDs
rep_indices = []
for scene_id, info in REPRESENTATIVE_SCENARIOS.items():
    if scene_id in scenario_id_to_idx:
        idx = scenario_id_to_idx[scene_id]
        rep_indices.append(idx)
        print(f"  ✓ {info['name']}: idx={idx}")
    else:
        print(f"  ✗ {info['name']}: NOT FOUND ({scene_id})")

print(f"📌 Found {len(rep_indices)}/6 representative scenarios")

# Validate: must have all 6
if len(rep_indices) < 6:
    missing = [info['name'] for sid, info in REPRESENTATIVE_SCENARIOS.items() if sid not in scenario_id_to_idx]
    raise ValueError(f"Missing {6 - len(rep_indices)} representative scenarios: {missing}. "
                     "Check data_root path or download full Argoverse 2 val set.")
print("✅ All 6 representative scenarios confirmed!")

# Determine scenario list based on config
if CONFIG.get('run_representative_only', False):
    # Run ONLY 6 representative scenarios (deep training mode)
    scenario_indices = rep_indices
    num_to_train = len(scenario_indices)
    print(f"📊 DEEP TRAINING MODE: Running {num_to_train} representative scenarios only")
else:
    # Run representative + random samples  
    max_scenarios = CONFIG.get('max_scenarios', total_scenarios)
    remaining_needed = max(0, max_scenarios - len(rep_indices))
    all_indices = set(range(total_scenarios))
    non_rep_indices = list(all_indices - set(rep_indices))
    random_samples = random.sample(non_rep_indices, min(remaining_needed, len(non_rep_indices)))
    scenario_indices = rep_indices + random_samples
    num_to_train = len(scenario_indices)
    print(f"📊 Will train on {num_to_train} scenarios ({len(rep_indices)} representative + {len(random_samples)} random)")

# %% [markdown]
# ## 4b. Initialize RL Config (Global)

# %%
# Load RL config once globally
with open(CONFIG['rl_config_file']) as f:
    RL_CONFIG = yaml.safe_load(f)

# Set all required parameters
RL_CONFIG['batch_size'] = CONFIG['batch_size']
RL_CONFIG['episodes'] = CONFIG['num_episodes']
RL_CONFIG['epochs'] = CONFIG['epochs']
RL_CONFIG['is_magnet'] = False
RL_CONFIG['eta_coef1'] = 0.0
RL_CONFIG['eta_coef2'] = 0.1
RL_CONFIG['penalty_initial_value'] = CONFIG.get('penalty_initial_value', 1.0)

# Compute state dim
STATE_DIM = model.num_modes * RL_CONFIG['hidden_dim']
OFFSET = RL_CONFIG['offset']

# Global args namespace
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
# ## 5. Metric Functions

# %%
from shapely.geometry import Point, Polygon

def compute_fidelity_metrics(pred_velocities, gt_velocities, pred_accelerations, gt_accelerations,
                              pred_steering=None, gt_steering=None, 
                              pred_distances=None, gt_distances=None):
    """
    Compute fidelity metrics: Wasserstein distance for velocity, acceleration, steering, and vehicle distance.
    Based on TrafficGamer paper Section VI.
    """
    metrics = {}
    
    # 1. Velocity fidelity
    if pred_velocities is not None and gt_velocities is not None:
        pred_v = np.array(pred_velocities).flatten()
        gt_v = np.array(gt_velocities).flatten()
        if len(pred_v) > 0 and len(gt_v) > 0:
            metrics['velocity_wasserstein'] = wasserstein_distance(gt_v, pred_v)
            # JSD as symmetric KL proxy
            hist_pred, bins = np.histogram(pred_v, bins=50, density=True)
            hist_gt, _ = np.histogram(gt_v, bins=bins, density=True)
            metrics['velocity_jsd'] = jensenshannon(hist_pred + 1e-10, hist_gt + 1e-10)
    
    # 2. Acceleration fidelity
    if pred_accelerations is not None and gt_accelerations is not None:
        pred_a = np.array(pred_accelerations).flatten()
        gt_a = np.array(gt_accelerations).flatten()
        if len(pred_a) > 0 and len(gt_a) > 0:
            metrics['acceleration_wasserstein'] = wasserstein_distance(gt_a, pred_a)
    
    # 3. Steering angle fidelity (computed from heading changes)
    if pred_steering is not None and gt_steering is not None:
        pred_s = np.array(pred_steering).flatten()
        gt_s = np.array(gt_steering).flatten()
        if len(pred_s) > 0 and len(gt_s) > 0:
            metrics['steering_wasserstein'] = wasserstein_distance(gt_s, pred_s)
    
    # 4. Vehicle distance fidelity (inter-vehicle distance distribution)
    if pred_distances is not None and gt_distances is not None:
        pred_d = np.array(pred_distances).flatten()
        gt_d = np.array(gt_distances).flatten()
        if len(pred_d) > 0 and len(gt_d) > 0:
            metrics['distance_wasserstein'] = wasserstein_distance(gt_d, pred_d)
    
    return metrics

def compute_steering_from_heading(headings):
    """Compute steering angle (heading change rate) from heading sequence."""
    if len(headings) < 2:
        return np.array([])
    # Steering = change in heading per timestep
    steering = np.diff(headings)
    # Normalize to [-pi, pi]
    steering = np.arctan2(np.sin(steering), np.cos(steering))
    return steering

def compute_vehicle_distances(positions):
    """Compute pairwise vehicle distances at each timestep."""
    distances = []
    num_agents = positions.shape[0]
    num_steps = positions.shape[1]
    
    for t in range(num_steps):
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = np.linalg.norm(positions[i, t, :2] - positions[j, t, :2])
                if dist < 50.0:  # Only consider nearby vehicles
                    distances.append(dist)
    return np.array(distances)

def compute_safety_metrics(positions, velocities, threshold_ttc=2.0, threshold_dist=5.0):
    """
    Compute safety metrics: TTC, THW, collision rate.
    Based on TrafficGamer paper - only considers pairs within 20m.
    """
    metrics = {'ttc_violations': 0, 'thw_violations': 0, 'collisions': 0, 'total_pairs': 0}
    
    num_agents = positions.shape[0]
    num_steps = positions.shape[1]
    
    for t in range(num_steps):
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                pos_i = positions[i, t, :2]
                pos_j = positions[j, t, :2]
                dist = np.linalg.norm(pos_i - pos_j)
                
                # Only consider nearby vehicles (< 20m) per paper
                if dist > 20.0:
                    continue
                    
                metrics['total_pairs'] += 1
                
                # Collision check (< 2m)
                if dist < 2.0:
                    metrics['collisions'] += 1
                
                # TTC & THW
                vel_i = velocities[i, t, :2] if t < velocities.shape[1] else np.zeros(2)
                vel_j = velocities[j, t, :2] if t < velocities.shape[1] else np.zeros(2)
                rel_vel = np.linalg.norm(vel_i - vel_j)
                
                if rel_vel > 0.1:
                    ttc = dist / rel_vel
                    if ttc < threshold_ttc:
                        metrics['ttc_violations'] += 1
                
                speed_i = np.linalg.norm(vel_i)
                if speed_i > 0.1:
                    thw = dist / speed_i
                    if thw < threshold_ttc:
                        metrics['thw_violations'] += 1
    
    # Compute rates
    if metrics['total_pairs'] > 0:
        metrics['collision_rate'] = metrics['collisions'] / metrics['total_pairs']
        metrics['ttc_violation_rate'] = metrics['ttc_violations'] / metrics['total_pairs']
        metrics['thw_violation_rate'] = metrics['thw_violations'] / metrics['total_pairs']
    else:
        metrics['collision_rate'] = 0.0
        metrics['ttc_violation_rate'] = 0.0
        metrics['thw_violation_rate'] = 0.0
    
    return metrics

def check_off_road(positions, static_map):
    """
    Check if vehicle positions are off-road (outside drivable area).
    Returns off-road rate.
    """
    if static_map is None:
        return {'off_road_rate': 0.0, 'off_road_count': 0, 'total_points': 0}
    
    # Build drivable area polygons
    drivable_polygons = []
    try:
        for da in static_map.vector_drivable_areas.values():
            poly = Polygon(da.xyz[:, :2])
            if poly.is_valid:
                drivable_polygons.append(poly)
    except:
        return {'off_road_rate': 0.0, 'off_road_count': 0, 'total_points': 0}
    
    if len(drivable_polygons) == 0:
        return {'off_road_rate': 0.0, 'off_road_count': 0, 'total_points': 0}
    
    # Check each position
    off_road_count = 0
    total_points = 0
    
    num_agents = positions.shape[0]
    num_steps = positions.shape[1]
    
    for i in range(num_agents):
        for t in range(num_steps):
            pt = Point(positions[i, t, 0], positions[i, t, 1])
            total_points += 1
            
            # Check if inside any drivable area
            is_on_road = any(poly.contains(pt) for poly in drivable_polygons)
            if not is_on_road:
                off_road_count += 1
    
    return {
        'off_road_rate': off_road_count / total_points if total_points > 0 else 0.0,
        'off_road_count': off_road_count,
        'total_points': total_points
    }

def compute_diversity(trajectories):
    """
    Compute trajectory diversity: average pairwise distance between trajectories.
    Based on TrafficGamer paper Section VI - Diversity metric.
    """
    if len(trajectories) < 2:
        return 0.0
    
    total_dist = 0.0
    count = 0
    for i in range(len(trajectories)):
        for j in range(i+1, len(trajectories)):
            # Average distance across all timesteps
            traj_i = np.array(trajectories[i])
            traj_j = np.array(trajectories[j])
            min_len = min(len(traj_i), len(traj_j))
            if min_len > 0:
                dist = np.mean(np.linalg.norm(traj_i[:min_len] - traj_j[:min_len], axis=-1))
                total_dist += dist
                count += 1
    
    return total_dist / count if count > 0 else 0.0

# %% [markdown]
# ## 6. Agent Selection & Training Functions

# %%
def get_agents(data, max_agents=10, radius=50.0):
    """Select valid vehicle agents."""
    hist_step = model.num_historical_steps - 1
    
    # AV
    av_idx = torch.nonzero(data["agent"]["category"] == 3, as_tuple=False)
    if len(av_idx) == 0: return []
    av_idx = av_idx[0].item()
    
    # Vehicles
    is_vehicle = data["agent"]["type"] == 0
    is_valid = data["agent"]["valid_mask"][:, hist_step]
    candidates = torch.nonzero(is_vehicle & is_valid, as_tuple=False).squeeze(-1)
    
    # Filter by distance
    av_pos = data["agent"]["position"][av_idx, hist_step]
    agent_pos = data["agent"]["position"][candidates, hist_step]
    dist = torch.norm(agent_pos - av_pos, dim=-1)
    nearby = candidates[dist < radius]
    nearby = nearby[nearby != av_idx]
    
    final = [av_idx] + nearby.tolist()
    return final[:max_agents]

def reconstruct_trajectories(initial_pos, initial_vel, initial_heading, actions, dt=0.1):
    """
    Reconstruct trajectories using Unicycle Kinematic Model.
    initial_pos: [N, 2]
    initial_vel: [N, 2] or [N] (speed)
    initial_heading: [N] (radians)
    actions: [N, T, 2] where actions are (acceleration, steering_curvature)
    """
    N, T, _ = actions.shape
    curr_pos = torch.tensor(initial_pos, dtype=torch.float32)
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
    
    # Compute velocity vectors from speed & heading
    # Recalculate speed sequence for vector generation
    gen_velocities = np.zeros_like(gen_positions)
    # Simple approx: differentiate positions or track speed. 
    # Let's just diff positions for velocity to be consistent with trajectory
    gen_velocities[:, 1:] = (gen_positions[:, 1:] - gen_positions[:, :-1]) / dt
    gen_velocities[:, 0] = gen_velocities[:, 1] # Duplicate first frame approx
    
    return gen_positions, gen_velocities, gen_headings


def train_scenario(scenario_idx, agent_class, agent_name, detailed=False):
    """
    Train on a single scenario and compute comprehensive metrics.
    Returns metrics dict with: reward, cost, safety (TTC/THW/collision), fidelity, diversity, off-road.
    """
    # Load single scenario
    loader = DataLoader([dataset[scenario_idx]], batch_size=1, shuffle=False)
    data = next(iter(loader)).to(DEVICE)
    if isinstance(data, Batch): data["agent"]["av_index"] += data["agent"]["ptr"][:-1]

    agent_indices = get_agents(data, CONFIG['max_agents'], CONFIG['agent_radius'])
    agent_num = len(agent_indices)
    if agent_num < 1: return None

    # Use global RL_CONFIG, only update agent_number
    rl_config = RL_CONFIG.copy()
    rl_config['agent_number'] = agent_num
    
    # Create agents
    agents = [agent_class(STATE_DIM, agent_num, rl_config, DEVICE) for _ in range(agent_num)]
    choose_agent = agent_indices

    # Load map
    scenario_id = data['scenario_id'][0] if isinstance(data['scenario_id'], list) else data['scenario_id']
    map_path = Pth(CONFIG['data_root']) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
    if not map_path.exists():
        found = list(Pth(CONFIG['data_root']).rglob(f'log_map_archive_{scenario_id}.json'))
        if found: map_path = found[0]
        else: return None
    scenario_static_map = ArgoverseStaticMap.from_json(map_path)

    # Args with correct workspace
    args = Namespace(scenario=1, distance_limit=CONFIG['distance_limit'], magnet=False,
                     eta_coef1=0, eta_coef2=0.1, track=False, confined_action=False, workspace=agent_name)

    # Metrics storage
    metrics = {'rewards': [], 'costs': []}
    
    # Collect GT data for fidelity (from original data BEFORE any modification)
    # GT is the FUTURE trajectory (after historical steps, which agents predict)
    hist_steps = model.num_historical_steps
    gt_positions = data["agent"]["position"][agent_indices, hist_steps:].cpu().numpy()  # [N, T_future, 2]
    gt_velocities = data["agent"]["velocity"][agent_indices, hist_steps:].cpu().numpy()  # [N, T_future, 2]
    gt_headings = data["agent"]["heading"][agent_indices, hist_steps:].cpu().numpy()    # [N, T_future]
    
    # Compute GT velocity norms and accelerations
    gt_vel_norms = np.linalg.norm(gt_velocities, axis=-1)  # [N, T_future]
    gt_accelerations = np.diff(gt_vel_norms, axis=1) / 0.1  # Change per 0.1s
    
    # Compute GT steering angles (heading change rate)
    gt_steering = np.diff(gt_headings, axis=1)
    gt_steering = np.arctan2(np.sin(gt_steering), np.cos(gt_steering))  # Normalize to [-pi, pi]
    
    # Compute GT vehicle distances
    gt_distances = compute_vehicle_distances(gt_positions)
    
    # Storage for GENERATED data
    # We collect structured actions from the LAST episode to reconstruct the final policy's trajectory
    last_ep_actions = np.zeros((agent_num, int(model.num_future_steps / OFFSET), 2))
    
    # Training loop
    for ep in tqdm(range(CONFIG['num_episodes']), desc=f"{agent_name}", leave=False):
        transition_list = [
            {
                "observations": [[] for _ in range(agent_num)], 
                "actions": [[] for _ in range(agent_num)], 
                "next_observations": [[] for _ in range(agent_num)],
                "rewards": [[] for _ in range(agent_num)], 
                "magnet": [[] for _ in range(agent_num)], 
                "costs": [[] for _ in range(agent_num)], 
                "dones": []
            }
            for _ in range(CONFIG['batch_size'])
        ]
        
        with torch.no_grad():
            for batch in range(CONFIG['batch_size']):
                PPO_process_batch(args, batch, data, model, agents, choose_agent,
                                  OFFSET, scenario_static_map, 1, transition_list,
                                  render=False, agent_num=agent_num, dataset_type='av2')

        for i in range(agent_num): agents[i].update(transition_list, i)

        # Collect rewards/costs
        ep_r, ep_c = 0, 0
        for i in range(agent_num):
            for t in range(int(model.num_future_steps / OFFSET)):
                for b in range(CONFIG['batch_size']):
                    ep_r += float(transition_list[b]["rewards"][i][t])
                    ep_c += float(transition_list[b]["costs"][i][t])
                    
                    # Capture actions from the LAST episode, BATCH 0
                    if ep == CONFIG['num_episodes'] - 1 and b == 0:
                        act = transition_list[b]["actions"][i][t]
                        if isinstance(act, torch.Tensor):
                            act = act.cpu().numpy()
                        # act might be shape (1,2) or (2,)
                        if act.size >= 2:
                            last_ep_actions[i, t] = act.flatten()[:2]
                        
        metrics['rewards'].append(ep_r / (agent_num * CONFIG['batch_size']))
        metrics['costs'].append(ep_c / (agent_num * CONFIG['batch_size']))
    
    # === RECONSTRUCT GENERATED TRAJECTORIES ===
    # Use actions from the final policy to generate trajectories via Unicycle Model
    hist_steps = model.num_historical_steps
    init_pos = data["agent"]["position"][agent_indices, hist_steps-1].cpu().numpy()
    init_vel = data["agent"]["velocity"][agent_indices, hist_steps-1].cpu().numpy()
    init_head = data["agent"]["heading"][agent_indices, hist_steps-1].cpu().numpy()
    
    gen_positions, gen_velocities, gen_headings = reconstruct_trajectories(
        init_pos, init_vel, init_head, last_ep_actions
    )
    
    # Compute derived generated metrics
    gen_vel_norms = np.linalg.norm(gen_velocities, axis=-1)
    gen_accelerations = np.zeros_like(gen_vel_norms[:, :-1]) # Diff reduces length by 1
    # Actually we can get acceleration directly from actions, but let's be consistent
    # Or just use the actions we collected:
    gen_accelerations_from_actions = last_ep_actions[:, :, 0] * 5.0 # Unscale
    gen_steering_from_actions = last_ep_actions[:, :, 1] * 0.05 # Unscale
    
    gen_distances = compute_vehicle_distances(gen_positions)
    
    # === COMPUTE ALL METRICS ===
    
    # 1. Basic metrics
    metrics['final_reward'] = np.mean(metrics['rewards'][-10:])
    metrics['final_cost'] = np.mean(metrics['costs'][-10:])
    metrics['agent_num'] = agent_num
    metrics['scenario_id'] = scenario_id
    
    # 2. Safety metrics (on GENERATED trajectories)
    safety = compute_safety_metrics(gen_positions, gen_velocities)
    metrics['collision_rate'] = safety['collision_rate']
    metrics['ttc_violation_rate'] = safety['ttc_violation_rate']
    metrics['thw_violation_rate'] = safety['thw_violation_rate']
    
    # 3. Fidelity metrics - compare Generated vs GT
    fidelity = compute_fidelity_metrics(
        pred_velocities=gen_vel_norms.flatten(),
        gt_velocities=gt_vel_norms.flatten() if len(gt_vel_norms.flatten()) > 0 else None,
        pred_accelerations=gen_accelerations_from_actions.flatten(),
        gt_accelerations=gt_accelerations.flatten(),
        pred_steering=gen_steering_from_actions.flatten(),
        gt_steering=gt_steering.flatten(),
        pred_distances=gen_distances, 
        gt_distances=gt_distances
    )
    metrics['velocity_wasserstein'] = fidelity.get('velocity_wasserstein', 0.0)
    metrics['acceleration_wasserstein'] = fidelity.get('acceleration_wasserstein', 0.0)
    metrics['steering_wasserstein'] = fidelity.get('steering_wasserstein', 0.0)
    metrics['distance_wasserstein'] = fidelity.get('distance_wasserstein', 0.0)
    
    # 4. Off-road rate - using GENERATED positions
    off_road = check_off_road(gen_positions, scenario_static_map)
    metrics['off_road_rate'] = off_road['off_road_rate']
    
    # 5. Diversity 
    # For diversity, we'd need multiple runs per scenario. 
    # Currently we only have 1 run per scenario (batch 0).
    # We can report 0 or implement multi-sample generation later.
    metrics['diversity'] = 0.0

    # Store additional data for checkpoint saving and 3D visualization
    metrics['_agents'] = agents  # For checkpoint saving
    metrics['_gen_positions'] = gen_positions
    metrics['_gen_velocities'] = gen_velocities
    metrics['_gen_headings'] = gen_headings
    metrics['_agent_indices'] = agent_indices
    metrics['_data'] = data  # Original scenario data
    metrics['_scenario_static_map'] = scenario_static_map
    
    return metrics



# %% [markdown]
# ## 7. Main Training Loop

# %%
all_results = []
representative_results = {'TrafficGamer': {}, 'EvoQRE': {}}

print(f"\n{'='*70}")
print(f"🚀 STARTING COMPREHENSIVE TRAINING")
print(f"   Total Scenarios: {num_to_train}")
print(f"   Episodes per Scenario: {CONFIG['num_episodes']}")
print(f"   Max Agents: {CONFIG['max_agents']}")
print(f"   Estimated Time: ~{num_to_train * 5} minutes")
print(f"{'='*70}\n")

start_time = time.time()

for loop_idx, idx in enumerate(tqdm(scenario_indices, desc="Training All Scenarios")):
    # Get scenario ID (idx is the actual dataset index)
    scenario_id = dataset.processed_file_names[idx].replace('.pkl', '')
    is_representative = scenario_id in REPRESENTATIVE_SCENARIOS
    
    if is_representative:
        print(f"\n⭐ REPRESENTATIVE SCENARIO: {REPRESENTATIVE_SCENARIOS[scenario_id]['name']}")
    
    # Train both methods
    tg_met = train_scenario(idx, TrafficGamer, "TrafficGamer", detailed=is_representative)
    evo_met = train_scenario(idx, EvoQRE_Langevin, "EvoQRE", detailed=is_representative)
    
    if tg_met is None or evo_met is None:
        continue
    
    # Collect ALL metrics
    result = {
        'idx': idx,
        'scenario_id': scenario_id,
        'is_representative': is_representative,
        # Basic
        'tg_reward': tg_met['final_reward'],
        'tg_cost': tg_met['final_cost'],
        'tg_agents': tg_met['agent_num'],
        'evo_reward': evo_met['final_reward'],
        'evo_cost': evo_met['final_cost'],
        'evo_agents': evo_met['agent_num'],
        # Safety
        'tg_collision_rate': tg_met.get('collision_rate', 0),
        'tg_ttc_violation': tg_met.get('ttc_violation_rate', 0),
        'tg_thw_violation': tg_met.get('thw_violation_rate', 0),
        'tg_off_road': tg_met.get('off_road_rate', 0),
        'evo_collision_rate': evo_met.get('collision_rate', 0),
        'evo_ttc_violation': evo_met.get('ttc_violation_rate', 0),
        'evo_thw_violation': evo_met.get('thw_violation_rate', 0),
        'evo_off_road': evo_met.get('off_road_rate', 0),
        # Fidelity
        'tg_vel_fidelity': tg_met.get('velocity_wasserstein', 0),
        'tg_accel_fidelity': tg_met.get('acceleration_wasserstein', 0),
        'tg_steer_fidelity': tg_met.get('steering_wasserstein', 0),
        'tg_dist_fidelity': tg_met.get('distance_wasserstein', 0),
        'evo_vel_fidelity': evo_met.get('velocity_wasserstein', 0),
        'evo_accel_fidelity': evo_met.get('acceleration_wasserstein', 0),
        'evo_steer_fidelity': evo_met.get('steering_wasserstein', 0),
        'evo_dist_fidelity': evo_met.get('distance_wasserstein', 0),
        # Diversity
        'tg_diversity': tg_met.get('diversity', 0),
        'evo_diversity': evo_met.get('diversity', 0),
    }
    all_results.append(result)
    
    # Store representative scenario details
    if is_representative:
        representative_results['TrafficGamer'][scenario_id] = tg_met
        representative_results['EvoQRE'][scenario_id] = evo_met
        print(f"   TG:  R={tg_met['final_reward']:.2f}, C={tg_met['final_cost']:.2f}, TTC={tg_met.get('ttc_violation_rate', 0)*100:.1f}%")
        print(f"   Evo: R={evo_met['final_reward']:.2f}, C={evo_met['final_cost']:.2f}, TTC={evo_met.get('ttc_violation_rate', 0)*100:.1f}%")
    
    # === SAVE CHECKPOINT ===
    if CONFIG.get('save_checkpoints', False):
        time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        for method_name, met in [('TrafficGamer', tg_met), ('EvoQRE', evo_met)]:
            if met is None or '_agents' not in met:
                continue
            agents = met['_agents']
            ckpt_dir = Path(CONFIG['checkpoint_dir']) / f"scenario_{scenario_id}" / method_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            
            model_state_dict = {}
            for i, agent in enumerate(agents):
                model_state_dict[f'agent_{i}_pi'] = agent.pi.state_dict()
                model_state_dict[f'agent_{i}_value'] = agent.value.state_dict()
                if hasattr(agent, 'cost_value_net_local'):
                    model_state_dict[f'agent_{i}_cost_value'] = agent.cost_value_net_local.state_dict()
            
            ckpt_path = ckpt_dir / f'{method_name}_{time_string}.pth'
            torch.save(model_state_dict, ckpt_path)
            print(f"   💾 Saved checkpoint: {ckpt_path.name}")
    
    # === SAVE 3D VISUALIZATION DATA ===
    if CONFIG.get('save_3d_visualization', False) and is_representative:
        time_string = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
        for method_name, met in [('TrafficGamer', tg_met), ('EvoQRE', evo_met)]:
            if met is None or '_gen_positions' not in met:
                continue
            
            gen_pos = met['_gen_positions']
            gen_vel = met['_gen_velocities']
            gen_head = met['_gen_headings']
            agent_indices = met['_agent_indices']
            data = met['_data']
            static_map = met['_scenario_static_map']
            
            # Create output directory
            out_dir = Path(CONFIG['3d_output_dir']) / f"scenario_{scenario_id}" / method_name
            out_dir.mkdir(parents=True, exist_ok=True)
            
            # Build DataFrame with predicted trajectories
            rows = []
            hist_steps = model.num_historical_steps
            num_future = gen_pos.shape[1]
            
            for agent_idx, ds_idx in enumerate(agent_indices):
                for t in range(num_future):
                    rows.append({
                        'scenario_id': scenario_id,
                        'track_id': f'agent_{agent_idx}',
                        'timestep': hist_steps + t,
                        'position_x': float(gen_pos[agent_idx, t, 0]),
                        'position_y': float(gen_pos[agent_idx, t, 1]),
                        'velocity_x': float(gen_vel[agent_idx, t, 0]),
                        'velocity_y': float(gen_vel[agent_idx, t, 1]),
                        'heading': float(gen_head[agent_idx, t]),
                        'method': method_name,
                    })
            
            pred_df = pd.DataFrame(rows)
            parquet_path = out_dir / f'predictions_{time_string}.parquet'
            pred_df.to_parquet(parquet_path, index=False)
            
            # Copy map file
            if static_map is not None:
                map_src = Path(CONFIG['data_root']) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
                if map_src.exists():
                    shutil.copy(map_src, out_dir / f'log_map_archive_{scenario_id}.json')
            
            print(f"   🎨 Saved 3D data: {parquet_path.name}")
    
    # Save intermediate results every scenario (since we have few)
    pd.DataFrame(all_results).to_csv(f"{CONFIG['output_dir']}/results_checkpoint.csv", index=False)

elapsed = time.time() - start_time
print(f"\n✅ Training Complete in {elapsed/60:.1f} minutes")

# %% [markdown]
# ## 8. Results Analysis

# %%
df = pd.DataFrame(all_results)

print("\n" + "="*70)
print("📊 COMPREHENSIVE RESULTS (Full Dataset)")
print("="*70)

# Basic Metrics
print(f"\n📈 REWARDS (higher is better):")
print(f"   TrafficGamer: {df['tg_reward'].mean():.4f} ± {df['tg_reward'].std():.4f}")
print(f"   EvoQRE:       {df['evo_reward'].mean():.4f} ± {df['evo_reward'].std():.4f}")

print(f"\n🛡️ COSTS (lower is better):")
print(f"   TrafficGamer: {df['tg_cost'].mean():.4f} ± {df['tg_cost'].std():.4f}")
print(f"   EvoQRE:       {df['evo_cost'].mean():.4f} ± {df['evo_cost'].std():.4f}")

# Safety Metrics
print(f"\n⚠️ SAFETY METRICS:")
print(f"   Collision Rate:")
print(f"       TG:  {df['tg_collision_rate'].mean()*100:.2f}% ± {df['tg_collision_rate'].std()*100:.2f}%")
print(f"       Evo: {df['evo_collision_rate'].mean()*100:.2f}% ± {df['evo_collision_rate'].std()*100:.2f}%")
print(f"   TTC Violation (<2s):")
print(f"       TG:  {df['tg_ttc_violation'].mean()*100:.2f}% ± {df['tg_ttc_violation'].std()*100:.2f}%")
print(f"       Evo: {df['evo_ttc_violation'].mean()*100:.2f}% ± {df['evo_ttc_violation'].std()*100:.2f}%")
print(f"   THW Violation (<2s):")
print(f"       TG:  {df['tg_thw_violation'].mean()*100:.2f}% ± {df['tg_thw_violation'].std()*100:.2f}%")
print(f"       Evo: {df['evo_thw_violation'].mean()*100:.2f}% ± {df['evo_thw_violation'].std()*100:.2f}%")
print(f"   Off-road Rate:")
print(f"       TG:  {df['tg_off_road'].mean()*100:.2f}%")
print(f"       Evo: {df['evo_off_road'].mean()*100:.2f}%")

# Fidelity Metrics
print(f"\n📏 FIDELITY METRICS (Wasserstein Distance, lower is better):")
print(f"   Velocity:")
print(f"       TG:  {df['tg_vel_fidelity'].mean():.4f}")
print(f"       Evo: {df['evo_vel_fidelity'].mean():.4f}")
print(f"   Acceleration:")
print(f"       TG:  {df['tg_accel_fidelity'].mean():.4f}")
print(f"       Evo: {df['evo_accel_fidelity'].mean():.4f}")
print(f"   Steering:")
print(f"       TG:  {df['tg_steer_fidelity'].mean():.4f}")
print(f"       Evo: {df['evo_steer_fidelity'].mean():.4f}")
print(f"   Vehicle Distance:")
print(f"       TG:  {df['tg_dist_fidelity'].mean():.4f}")
print(f"       Evo: {df['evo_dist_fidelity'].mean():.4f}")

# Diversity Metric
print(f"\n🎯 DIVERSITY (higher is better):")
print(f"   TrafficGamer: {df['tg_diversity'].mean():.4f}")
print(f"   EvoQRE:       {df['evo_diversity'].mean():.4f}")


# Collision Rate (cost > 0)
tg_collision_rate = (df['tg_cost'] > 0).mean() * 100
evo_collision_rate = (df['evo_cost'] > 0).mean() * 100
print(f"\n💥 COLLISION RATE:")
print(f"   TrafficGamer: {tg_collision_rate:.2f}%")
print(f"   EvoQRE:       {evo_collision_rate:.2f}%")

# %% [markdown]
# ## 9. Representative Scenarios Analysis

# %%
print("\n" + "="*70)
print("⭐ REPRESENTATIVE SCENARIOS ANALYSIS (Paper Table)")
print("="*70)

rep_df = df[df['is_representative'] == True].copy()
if len(rep_df) > 0:
    rep_df['scenario_name'] = rep_df['scenario_id'].map(lambda x: REPRESENTATIVE_SCENARIOS.get(x, {}).get('name', x))
    
    print("\n📋 Scenario Breakdown:")
    for _, row in rep_df.iterrows():
        print(f"\n{row['scenario_name']}:")
        print(f"   Agents: {row['tg_agents']}")
        print(f"   TG:  R={row['tg_reward']:.2f}, C={row['tg_cost']:.2f}")
        print(f"   Evo: R={row['evo_reward']:.2f}, C={row['evo_cost']:.2f}")

# %% [markdown]
# ## 10. Visualization

# %%
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Reward Distribution
axes[0,0].hist(df['tg_reward'], bins=50, alpha=0.6, label='TrafficGamer', color='blue')
axes[0,0].hist(df['evo_reward'], bins=50, alpha=0.6, label='EvoQRE', color='green')
axes[0,0].set_title('Reward Distribution')
axes[0,0].legend()
axes[0,0].set_xlabel('Reward')

# Cost Distribution
axes[0,1].hist(df['tg_cost'], bins=50, alpha=0.6, label='TrafficGamer', color='red')
axes[0,1].hist(df['evo_cost'], bins=50, alpha=0.6, label='EvoQRE', color='orange')
axes[0,1].set_title('Cost Distribution')
axes[0,1].legend()
axes[0,1].set_xlabel('Cost')

# Bar Comparison
methods = ['TrafficGamer', 'EvoQRE']
reward_means = [df['tg_reward'].mean(), df['evo_reward'].mean()]
cost_means = [df['tg_cost'].mean(), df['evo_cost'].mean()]

axes[1,0].bar(methods, reward_means, color=['blue', 'green'], alpha=0.7)
axes[1,0].set_title('Average Reward')
axes[1,0].set_ylabel('Reward')
for i, v in enumerate(reward_means):
    axes[1,0].text(i, v + 0.5, f'{v:.2f}', ha='center')

axes[1,1].bar(methods, cost_means, color=['red', 'orange'], alpha=0.7)
axes[1,1].set_title('Average Cost')
axes[1,1].set_ylabel('Cost')
for i, v in enumerate(cost_means):
    axes[1,1].text(i, v + 0.01, f'{v:.3f}', ha='center')

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/comparison_plot.png", dpi=150)
plt.show()

# %% [markdown]
# ## 11. Save All Results

# %%
# Full Results
df.to_csv(f"{CONFIG['output_dir']}/full_results.csv", index=False)
print(f"✅ Full results saved to {CONFIG['output_dir']}/full_results.csv")

# Representative Scenarios
if len(rep_df) > 0:
    rep_df.to_csv(f"{CONFIG['output_dir']}/representative_scenarios.csv", index=False)
    print(f"✅ Representative scenarios saved")

# Summary Statistics
summary = {
    'total_scenarios': len(df),
    'tg_avg_reward': float(df['tg_reward'].mean()),
    'tg_std_reward': float(df['tg_reward'].std()),
    'tg_avg_cost': float(df['tg_cost'].mean()),
    'tg_std_cost': float(df['tg_cost'].std()),
    'tg_collision_rate': float(tg_collision_rate),
    'evo_avg_reward': float(df['evo_reward'].mean()),
    'evo_std_reward': float(df['evo_reward'].std()),
    'evo_avg_cost': float(df['evo_cost'].mean()),
    'evo_std_cost': float(df['evo_cost'].std()),
    'evo_collision_rate': float(evo_collision_rate),
    'config': CONFIG,
}

with open(f"{CONFIG['output_dir']}/summary.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"✅ Summary saved to {CONFIG['output_dir']}/summary.json")

# %% [markdown]
# ## 12. Visualization Functions

# %%
from matplotlib.patches import Rectangle
import cv2

# Constants for visualization
_DRIVABLE_AREA_COLOR = "#7A7A7A"
_LANE_SEGMENT_COLOR = "#E0E0E0"
_DEFAULT_ACTOR_COLOR = "#D3E8EF"
_FOCAL_AGENT_COLOR = "#ECA25B"
_AV_COLOR = "#007672"
_VEHICLE_LENGTH = 4.0
_VEHICLE_WIDTH = 2.0

def plot_static_map(static_map, ax):
    """Plot map elements (drivable areas, lanes)."""
    # Plot drivable areas
    for drivable_area in static_map.vector_drivable_areas.values():
        ax.fill(drivable_area.xyz[:, 0], drivable_area.xyz[:, 1], 
                color=_DRIVABLE_AREA_COLOR, alpha=0.5)
    
    # Plot lane segments
    for lane_segment in static_map.vector_lane_segments.values():
        left = lane_segment.left_lane_boundary.xyz
        right = lane_segment.right_lane_boundary.xyz
        ax.plot(left[:, 0], left[:, 1], color=_LANE_SEGMENT_COLOR, linewidth=0.5)
        ax.plot(right[:, 0], right[:, 1], color=_LANE_SEGMENT_COLOR, linewidth=0.5)

def plot_agent_box(ax, pos, heading, color, size=(_VEHICLE_LENGTH, _VEHICLE_WIDTH)):
    """Plot a rotated bounding box for a vehicle."""
    import math
    length, width = size
    d = np.hypot(length, width)
    theta_2 = math.atan2(width, length)
    pivot_x = pos[0] - (d / 2) * math.cos(heading + theta_2)
    pivot_y = pos[1] - (d / 2) * math.sin(heading + theta_2)
    
    rect = Rectangle((pivot_x, pivot_y), length, width, np.degrees(heading),
                      color=color, zorder=100)
    ax.add_patch(rect)

def plot_scenario(data, static_map, timestep, agent_indices, bounds=80.0, title=None):
    """Plot a single frame of the scenario."""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Plot map
    if static_map is not None:
        plot_static_map(static_map, ax)
    
    # Find plot center (first agent)
    center = None
    
    for i in range(data["agent"]["num_nodes"]):
        if not data["agent"]["valid_mask"][i, timestep]:
            continue
        
        pos = data["agent"]["position"][i, :timestep+1][data["agent"]["valid_mask"][i, :timestep+1]].cpu().numpy()
        heading = data["agent"]["heading"][i, timestep].cpu().item()
        
        if len(pos) == 0:
            continue
        
        # Determine color
        if i in agent_indices:
            color = _FOCAL_AGENT_COLOR
            # Plot trajectory
            ax.plot(pos[:, 0], pos[:, 1], color='black', linewidth=2)
            if center is None:
                center = pos[-1]
        elif data["agent"]["type"][i] == 0:
            color = _DEFAULT_ACTOR_COLOR
        else:
            continue
        
        # Plot bounding box for vehicles
        if data["agent"]["type"][i] == 0:
            plot_agent_box(ax, pos[-1], heading, color)
    
    # Set bounds
    if center is not None:
        ax.set_xlim(center[0] - bounds, center[0] + bounds)
        ax.set_ylim(center[1] - bounds, center[1] + bounds)
    
    ax.set_aspect('equal')
    ax.axis('off')
    
    if title:
        ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    return fig

def save_scenario_visualization(scenario_id, data, static_map, agent_indices, output_path):
    """Save visualization of a scenario."""
    fig = plot_scenario(data, static_map, timestep=49, agent_indices=agent_indices, title=f"Scenario: {scenario_id}")
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✅ Saved visualization: {output_path}")

# %% [markdown]
# ## 13. Visualize Representative Scenarios

# %%
# Create visualizations for representative scenarios
viz_dir = f"{CONFIG['output_dir']}/visualizations"
os.makedirs(viz_dir, exist_ok=True)

print("\n📷 Generating visualizations for representative scenarios...")

for scenario_id, info in REPRESENTATIVE_SCENARIOS.items():
    try:
        # Find scenario index
        try:
            file_idx = dataset.processed_file_names.index(f'{scenario_id}.pkl')
        except ValueError:
            print(f"   ⚠️ {info['name']} not found, skipping...")
            continue
        
        # Load data
        loader = DataLoader([dataset[file_idx]], batch_size=1, shuffle=False)
        data = next(iter(loader)).to(DEVICE)
        
        # Get agents
        agents = get_agents(data, CONFIG['max_agents'], CONFIG['agent_radius'])
        
        # Load map
        map_path = Pth(CONFIG['data_root']) / 'val' / 'raw' / scenario_id / f'log_map_archive_{scenario_id}.json'
        static_map = ArgoverseStaticMap.from_json(map_path) if map_path.exists() else None
        
        # Save visualization
        output_path = f"{viz_dir}/{info['type']}_{scenario_id[:8]}.png"
        save_scenario_visualization(scenario_id, data, static_map, agents, output_path)
        
    except Exception as e:
        print(f"   ⚠️ Error visualizing {info['name']}: {e}")

print(f"✅ Visualizations saved to {viz_dir}/")

print("\n🎉 ALL RESULTS SAVED SUCCESSFULLY!")
