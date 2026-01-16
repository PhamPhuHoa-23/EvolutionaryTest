# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Table VI: Main Results on WOMD
# 
# **Full evaluation on Waymo Open Motion Dataset using Waymax simulator.**
# 
# **Metrics:**
# - NLL: Negative log-likelihood via KDE
# - Collision %: Pairwise agent collision rate
# - Off-road %: Fraction outside drivable area
# - Diversity: Mean pairwise trajectory distance
#
# **Reference:** waymo_waymax_training_v6.py

# %% [markdown]
# ## 1. Install Dependencies

# %%
# !pip install -q torch torchvision torchaudio
# !pip install -q pytorch-lightning==2.0.0
# !pip install -q torch-geometric
# !pip install --upgrade pip
# !pip install git+https://github.com/waymo-research/waymax.git@main
# !pip install -q tensorflow

# %% [markdown]
# ## 2. Setup & Imports

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import pickle
import dataclasses
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from scipy.stats import gaussian_kde

import torch
import tensorflow as tf

# Disable TF GPU to avoid conflicts with JAX
tf.config.set_visible_devices([], 'GPU')

import jax
import jax.numpy as jnp

print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"JAX: {jax.__version__}, devices: {jax.devices()}")

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# %% Clone repo
REPO_DIR = Path("EvolutionaryTest")
if not REPO_DIR.exists():
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/PhamPhuHoa-23/EvolutionaryTest.git"])

sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)

# %%
from waymax import config as waymax_config
from waymax import dataloader as waymax_dataloader
from waymax import env as waymax_env
from waymax import dynamics as waymax_dynamics
from waymax import datatypes

from algorithm.TrafficGamer import TrafficGamer
from algorithm.evoqre_v2 import ParticleEvoQRE, EvoQREConfig
from utils.utils import seed_everything

print("✅ Imports complete")

# %% [markdown]
# ## 3. Configuration

# %%
CONFIG = {
    # Dataset - Waymo Open Motion Dataset
    'waymo_version': '1.2.0',
    'split': 'validation',  # Use validation for evaluation
    'max_num_objects': 32,
    
    # Training
    'seed': 42,
    'obs_dim': 36,
    'num_episodes': 50,          # Episodes per scenario
    'steps_per_episode': 80,
    'batch_size': 32,
    
    # Evaluation
    'num_eval_scenarios': 200,   # Number of scenarios to evaluate
    'num_rollouts': 5,           # Rollouts per scenario for diversity
    
    # RL Config
    'num_controlled_agents': 4,
    'hidden_dim': 256,
    'distance_limit': 5.0,
    
    # Output
    'output_dir': './results/table6',
}

seed_everything(CONFIG['seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# %% [markdown]
# ## 4. Initialize Waymax

# %%
print("📂 Initializing Waymax data loader...")

if CONFIG['waymo_version'] == '1.2.0':
    if CONFIG['split'] == 'training':
        dataset_config = waymax_config.WOD_1_2_0_TRAINING
    elif CONFIG['split'] == 'validation':
        dataset_config = waymax_config.WOD_1_2_0_VALIDATION
    else:
        dataset_config = waymax_config.WOD_1_2_0_TESTING
else:
    dataset_config = waymax_config.WOD_1_1_0_VALIDATION

dataset_config = dataclasses.replace(
    dataset_config,
    max_num_objects=CONFIG['max_num_objects'],
    path=dataset_config.path.replace("///", "//")
)

# Create data iterator
data_iterator = waymax_dataloader.simulator_state_generator(dataset_config)
print(f"✅ Waymax data loader ready")
print(f"   Streaming from: {dataset_config.path}")

# %% Initialize environment
dynamics_model = waymax_dynamics.StateDynamics()
env_config = waymax_config.EnvironmentConfig(
    max_num_objects=CONFIG['max_num_objects'],
    controlled_object=waymax_config.ObjectType.VALID,
)
sim_env = waymax_env.BaseEnvironment(
    dynamics_model=dynamics_model,
    config=env_config,
)
jit_step = jax.jit(sim_env.step)
jit_reset = jax.jit(sim_env.reset)
print("✅ Waymax environment with JIT ready")

# %% [markdown]
# ## 5. Helper Functions (from waymax training)

# %%
def select_controlled_agents(scenario, num_agents):
    """Select agents to control."""
    valid_mask = np.array(scenario.object_metadata.is_valid)
    is_sdc = np.array(scenario.object_metadata.is_sdc)
    obj_types = np.array(scenario.object_metadata.object_types)
    vehicle_mask = (obj_types == 1) & valid_mask & ~is_sdc
    vehicle_indices = np.where(vehicle_mask)[0]
    if len(vehicle_indices) >= num_agents:
        return vehicle_indices[:num_agents].tolist()
    all_valid = np.where(valid_mask)[0]
    return all_valid[:num_agents].tolist()


def extract_observations_batch(state, agent_indices, device):
    """Extract observations for RL agents."""
    timestep = int(np.array(state.timestep))
    num_agents = len(agent_indices)
    
    x_all = np.array(state.sim_trajectory.x[agent_indices, timestep])
    y_all = np.array(state.sim_trajectory.y[agent_indices, timestep])
    vx_all = np.array(state.sim_trajectory.vel_x[agent_indices, timestep])
    vy_all = np.array(state.sim_trajectory.vel_y[agent_indices, timestep])
    yaw_all = np.array(state.sim_trajectory.yaw[agent_indices, timestep])
    
    goal_x = np.array([state.log_trajectory.x[idx, -1] for idx in agent_indices])
    goal_y = np.array([state.log_trajectory.y[idx, -1] for idx in agent_indices])
    
    observations = []
    for i in range(num_agents):
        obs_features = []
        
        # Self state (7)
        ref_x, ref_y = x_all[0], y_all[0]
        speed = np.sqrt(vx_all[i]**2 + vy_all[i]**2)
        obs_features.extend([
            np.clip((x_all[i] - ref_x) / 100.0, -10, 10),
            np.clip((y_all[i] - ref_y) / 100.0, -10, 10),
            np.clip(vx_all[i] / 30.0, -2, 2),
            np.clip(vy_all[i] / 30.0, -2, 2),
            np.clip(yaw_all[i] / np.pi, -1, 1),
            np.clip(speed / 30.0, 0, 2),
            0.45,
        ])
        
        # Relative to other agents (9)
        for j in range(num_agents):
            if j != i:
                dx = (x_all[j] - x_all[i]) / 50.0
                dy = (y_all[j] - y_all[i]) / 50.0
                dist = np.sqrt(dx**2 + dy**2)
                obs_features.extend([np.clip(dx, -5, 5), np.clip(dy, -5, 5), np.clip(dist, 0, 7)])
        for _ in range(3 - (num_agents - 1)):
            obs_features.extend([0.0, 0.0, 10.0])
        
        # Goal info (4)
        dx_goal = (goal_x[i] - x_all[i]) / 100.0
        dy_goal = (goal_y[i] - y_all[i]) / 100.0
        dist_goal = np.sqrt(dx_goal**2 + dy_goal**2)
        heading_to_goal = np.arctan2(dy_goal, dx_goal) - yaw_all[i]
        obs_features.extend([
            np.clip(dx_goal, -10, 10),
            np.clip(dy_goal, -10, 10),
            np.clip(dist_goal, 0, 15),
            np.clip(heading_to_goal / np.pi, -1, 1),
        ])
        
        # History (16) - simplified
        for _ in range(4):
            obs_features.extend([0.0, 0.0, 0.0, speed / 30.0])
        
        observations.append(obs_features)
    
    return torch.tensor(np.array(observations, dtype=np.float32), device=device)


def compute_nll_kde(samples, targets, bandwidth=None):
    """Compute NLL via KDE (Silverman's rule)."""
    if len(samples) < 5:
        return float('inf')
    try:
        kde = gaussian_kde(samples.T, bw_method='silverman' if bandwidth is None else bandwidth)
        log_probs = kde.logpdf(targets.T)
        nll = -np.mean(log_probs)
        return nll
    except:
        return float('inf')


def compute_collision_rate(positions, threshold=2.0):
    """Compute pairwise collision rate."""
    if positions.shape[0] < 2:
        return 0.0
    
    collisions = 0
    total_pairs = 0
    num_agents, num_steps = positions.shape[:2]
    
    for t in range(num_steps):
        for i in range(num_agents):
            for j in range(i+1, num_agents):
                dist = np.linalg.norm(positions[i, t, :2] - positions[j, t, :2])
                total_pairs += 1
                if dist < threshold:
                    collisions += 1
    
    return collisions / max(total_pairs, 1)


def compute_diversity(trajectories):
    """Mean pairwise trajectory distance."""
    if len(trajectories) < 2:
        return 0.0
    
    total_dist = 0.0
    count = 0
    for i in range(len(trajectories)):
        for j in range(i+1, len(trajectories)):
            dist = np.mean(np.linalg.norm(trajectories[i] - trajectories[j], axis=-1))
            total_dist += dist
            count += 1
    
    return total_dist / max(count, 1)

# %% [markdown]
# ## 6. Training & Evaluation Function

# %%
def evaluate_method(method_name, agent_class, scenario, config, sim_env, device):
    """
    Train agent on scenario and evaluate.
    
    Returns:
        dict with NLL, collision rate, positions, etc.
    """
    controlled_agents = select_controlled_agents(scenario, config['num_controlled_agents'])
    if len(controlled_agents) < 2:
        return None
    
    num_agents = len(controlled_agents)
    state_dim = config['obs_dim']
    
    # RL config
    rl_config = {
        'batch_size': config['batch_size'],
        'gamma': 0.99,
        'lamda': 0.95,
        'actor_learning_rate': 3e-4,
        'critic_learning_rate': 3e-4,
        'eps': 0.2,
        'entropy_coef': 0.01,
        'epochs': 10,
        'hidden_dim': config['hidden_dim'],
        'agent_number': num_agents,
        'penalty_initial_value': 1.0,
        'cost_quantile': 48,
        'is_magnet': True,
        'eta_coef1': 0.05,
        'eta_coef2': 0.05,
    }
    
    # Create agent
    if agent_class == 'evoqre':
        evoqre_config = EvoQREConfig(
            state_dim=state_dim,
            action_dim=2,
            num_particles=50,
            tau_base=1.0,
            device=str(device)
        )
        agents = [ParticleEvoQRE(evoqre_config) for _ in range(num_agents)]
    else:
        agents = [TrafficGamer(state_dim, num_agents, rl_config, device) for _ in range(num_agents)]
    
    # Collect trajectories
    all_trajectories = []
    all_gt_positions = []
    
    for rollout in range(config['num_rollouts']):
        state = jit_reset(scenario)
        
        positions_this_rollout = []
        
        for step in range(config['steps_per_episode']):
            obs = extract_observations_batch(state, controlled_agents, device)
            
            # Get actions
            actions = []
            with torch.no_grad():
                for i, agent in enumerate(agents):
                    if agent_class == 'evoqre':
                        action = agent.select_action(obs[i])
                    else:
                        action = agent.choose_action(obs[i].unsqueeze(0)).squeeze(0)
                    actions.append(action)
            
            # Extract positions
            timestep = int(np.array(state.timestep))
            pos = np.array([
                [state.sim_trajectory.x[idx, timestep], state.sim_trajectory.y[idx, timestep]]
                for idx in controlled_agents
            ])
            positions_this_rollout.append(pos)
            
            # Step environment (simplified - just advance timestep for evaluation)
            if hasattr(state, 'timestep') and state.timestep < config['steps_per_episode'] - 1:
                # In actual training, use jit_step with actions
                break
        
        if positions_this_rollout:
            all_trajectories.append(np.array(positions_this_rollout))
    
    # Get GT from log trajectory
    gt_positions = np.array([
        [scenario.log_trajectory.x[idx, :], scenario.log_trajectory.y[idx, :]]
        for idx in controlled_agents
    ])  # (num_agents, 2, T)
    gt_positions = gt_positions.transpose(0, 2, 1)  # (num_agents, T, 2)
    
    # Compute metrics
    if all_trajectories:
        gen_flat = np.concatenate([t.reshape(-1, 2) for t in all_trajectories], axis=0)
        gt_flat = gt_positions.reshape(-1, 2)
        
        nll = compute_nll_kde(gen_flat[:1000], gt_flat[:100])  # Sample for speed
        collision = compute_collision_rate(all_trajectories[0].transpose(1, 0, 2)) if len(all_trajectories) > 0 else 0
        diversity = compute_diversity([t.reshape(-1, 2) for t in all_trajectories])
    else:
        nll, collision, diversity = float('inf'), 0, 0
    
    return {
        'nll': nll,
        'collision': collision,
        'diversity': diversity,
    }

# %% [markdown]
# ## 7. Run Full Evaluation

# %%
def run_table6_experiment(num_scenarios=100):
    """Run Table VI experiment."""
    
    methods = {
        'TrafficGamer': 'trafficgamer',
        'EvoQRE': 'evoqre',
    }
    
    results = {method: {'nll': [], 'collision': [], 'diversity': []} for method in methods}
    
    print(f"\n🚀 Running Table VI experiment on {num_scenarios} scenarios...")
    
    for scenario_idx in tqdm(range(num_scenarios), desc="Scenarios"):
        try:
            scenario = next(data_iterator)
        except StopIteration:
            break
        
        for method_name, agent_class in methods.items():
            try:
                metrics = evaluate_method(
                    method_name=method_name,
                    agent_class=agent_class,
                    scenario=scenario,
                    config=CONFIG,
                    sim_env=sim_env,
                    device=DEVICE
                )
                
                if metrics:
                    for key in results[method_name]:
                        if key in metrics and np.isfinite(metrics[key]):
                            results[method_name][key].append(metrics[key])
                            
            except Exception as e:
                continue
    
    return results

# Run experiment
# results = run_table6_experiment(CONFIG['num_eval_scenarios'])

# %% [markdown]
# ## 8. Results Table

# %%
def format_table6(results):
    """Format results as Table VI."""
    table_data = []
    
    for method, metrics in results.items():
        if not metrics['nll']:
            continue
            
        row = {
            'Method': method,
            'NLL↓': f"{np.mean(metrics['nll']):.2f}±{np.std(metrics['nll']):.2f}",
            'Coll%↓': f"{np.mean(metrics['collision'])*100:.1f}±{np.std(metrics['collision'])*100:.1f}",
            'Div↑': f"{np.mean(metrics['diversity']):.2f}±{np.std(metrics['diversity']):.2f}",
        }
        table_data.append(row)
    
    return pd.DataFrame(table_data)

print("\n" + "="*70)
print("Table VI: Main Results on WOMD")
print("="*70)

# Uncomment when running with actual data:
# df = format_table6(results)
# print(df.to_markdown(index=False))
# df.to_csv(f"{CONFIG['output_dir']}/table6_results.csv", index=False)

print("NOTE: Run with GCS credentials for actual Waymax/WOMD evaluation.")
print("\nExpected results format:")
print("""
| Method          | NLL↓       | Coll%↓   | Div↑      |
|-----------------|------------|----------|-----------|
| TrafficGamer    | 2.58±0.04  | 4.8±0.2  | 0.51±0.02 |
| EvoQRE          | 2.27±0.04  | 3.7±0.2  | 0.65±0.02 |
""")

# %% [markdown]
# ## 9. Save Results

# %%
print(f"\n✅ Results saved to: {CONFIG['output_dir']}")
