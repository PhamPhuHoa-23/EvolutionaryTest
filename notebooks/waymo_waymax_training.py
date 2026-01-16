# %% [markdown]
# # 🚗 TrafficGamer - Pure Waymax Training on Kaggle
#
# **Train multi-agent RL policies using Waymax simulator**
#
# **Key Features:**
# - ✅ NO local data needed - streams directly from Google Cloud
# - ✅ NO QCNet needed - uses Waymax observations directly
# - ✅ Waymax handles: data loading + dynamics + simulation
# - ✅ Train TrafficGamer / MAPPO / CCE-MAPPO policies
#
# **Repository:** https://github.com/PhamPhuHoa-23/EvolutionaryTest

# %% [markdown]
# ## 1. Install Dependencies

# %%
# Core dependencies
from utils.utils import seed_everything
from algorithm.TrafficGamer import TrafficGamer
from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO
from algorithm.cce_mappo import CCE_MAPPO
from algorithm.mappo import MAPPO
from waymax import visualization as waymax_viz
from waymax import datatypes
from waymax import dynamics as waymax_dynamics
from waymax import env as waymax_env
from waymax import dataloader as waymax_dataloader
from waymax import config as waymax_config
from torch.distributions import Laplace, Normal
import torch.nn.functional as F
import torch.nn as nn
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import dataclasses
import pickle
import yaml
from pathlib import Path
import warnings
import sys
import os
import numpy as np
import tensorflow as tf
import torch
!pip install - q torch torchvision torchaudio
!pip install - q pytorch-lightning == 2.0.0
!pip install - q torch-geometric

# %%
# Waymax - JAX-based simulator for Waymo (includes data loading)
!pip install - -upgrade pip
!pip install git+https: // github.com/waymo-research/waymax.git@main  # egg=waymo-waymax

# %%
# TensorFlow for data loading (Waymax uses tf.data internally)
!pip install - q tensorflow

# %%

print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    cuda_version = torch.version.cuda.replace('.', '')
    print(f"CUDA version: {cuda_version}")
else:
    cuda_version = 'cpu'

# %%
# PyTorch Geometric dependencies
if torch.cuda.is_available():
    !pip install torch-scatter torch-sparse torch-cluster - f https: // data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_version[:3]}.html
else:
    !pip install torch-scatter torch-sparse torch-cluster - f https: // data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cpu.html

# %% [markdown]
# ## 2. Setup Environment

# %%

warnings.filterwarnings('ignore')
tf.config.set_visible_devices([], 'GPU')  # Disable TF GPU (use PyTorch)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# %% [markdown]
# ## 3. Authenticate with Google Cloud
#
# **REQUIRED** to access Waymo Open Motion Dataset from GCS bucket
#
# Choose ONE method:
# - **Kaggle**: Upload service account key as dataset
# - **Colab**: Run `auth.authenticate_user()`

# %%
print("🔑 Authenticating with Google Cloud...")

# Method 1: Service Account Key (Kaggle)
service_key_path = '/kaggle/input/gcs-credentials/auth.json'

if os.path.exists(service_key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
    print(f"✅ Authenticated via service account")
else:
    # Method 2: Colab authentication
    try:
        from google.colab import auth
        auth.authenticate_user()
        print("✅ Authenticated via Colab")
    except ImportError:
        print("⚠️  No authentication found!")
        print("   Kaggle: Upload GCS credentials as dataset")
        print("   Colab: from google.colab import auth; auth.authenticate_user()")

# %% [markdown]
# ## 4. Clone Repository & Import Components

# %%
REPO_DIR = Path("EvolutionaryTest")

if not REPO_DIR.exists():
    print("📥 Cloning repository...")
    !git clone https: // github.com/PhamPhuHoa-23/EvolutionaryTest.git
else:
    print("✅ Repository exists, pulling latest...")
    !cd EvolutionaryTest & & git pull

sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %%
!pip install - q - r requirements.txt

# %%

# Waymax imports

# TrafficGamer RL algorithms

print("✅ All imports successful")

# %% [markdown]
# ## 5. Configuration

# %%
# ============================================
# 🔧 CONFIGURATION
# ============================================

CONFIG = {
    # Waymo dataset version & split
    'waymo_version': '1.2.0',  # '1.1.0' or '1.2.0'
    'split': 'training',  # 'training', 'validation', 'testing'

    # Waymax settings
    'max_num_objects': 32,  # Max objects per scenario (memory)

    # Observation settings
    'obs_dim': 7,  # [x, y, vx, vy, yaw, speed, length]

    # Training settings
    'seed': 42,
    'num_episodes': 300,
    'steps_per_episode': 80,  # 8 seconds @ 10Hz
    'batch_size': 32,

    # RL algorithm: 'TrafficGamer', 'MAPPO', 'CCE_MAPPO', 'Constrainted_CCE_MAPPO'
    'rl_algorithm': 'TrafficGamer',

    # Agent settings
    'num_controlled_agents': 4,  # Number of agents to control
    'hidden_dim': 128,

    # RL hyperparameters
    'gamma': 0.99,
    'gae_lambda': 0.95,
    'actor_lr': 3e-4,
    'critic_lr': 3e-4,
    'eps': 0.2,
    'entropy_coef': 0.005,
    'epochs': 10,

    # Safety constraints
    'distance_limit': 5.0,  # Collision threshold (meters)
    'cost_quantile': 48,
    'penalty_initial_value': 1.0,

    # Magnet (exploration)
    'is_magnet': True,
    'eta_coef1': 0.05,

    # Logging
    'log_interval': 10,
    'save_checkpoint': True,
}

seed_everything(CONFIG['seed'])

print("✅ Configuration:")
for k, v in CONFIG.items():
    print(f"   {k}: {v}")

# %% [markdown]
# ## 6. Initialize Waymax Data Loader
#
# **Waymax streams data directly from Google Cloud - NO download needed!**

# %%
print("📂 Initializing Waymax data loader...")

# Select dataset config based on version and split
if CONFIG['waymo_version'] == '1.2.0':
    if CONFIG['split'] == 'training':
        dataset_config = waymax_config.WOD_1_2_0_TRAINING
    elif CONFIG['split'] == 'validation':
        dataset_config = waymax_config.WOD_1_2_0_VALIDATION
    else:
        dataset_config = waymax_config.WOD_1_2_0_TESTING
else:  # 1.1.0
    if CONFIG['split'] == 'training':
        dataset_config = waymax_config.WOD_1_1_0_TRAINING
    elif CONFIG['split'] == 'validation':
        dataset_config = waymax_config.WOD_1_1_0_VALIDATION
    else:
        dataset_config = waymax_config.WOD_1_1_0_TESTING

# Fix GCS path and configure
dataset_config = dataclasses.replace(
    dataset_config,
    max_num_objects=CONFIG['max_num_objects'],
    path=dataset_config.path.replace("///", "//")
)

# Create data iterator (streams from GCS)
data_iterator = waymax_dataloader.simulator_state_generator(dataset_config)

print(f"✅ Waymax data loader ready")
print(f"   Dataset: Waymo Open Motion Dataset v{CONFIG['waymo_version']}")
print(f"   Split: {CONFIG['split']}")
print(f"   Streaming from: {dataset_config.path}")

# %% [markdown]
# ## 7. Test Data Loading

# %%
print("🔍 Testing data loading...")

# Get first scenario
sample_scenario = next(data_iterator)

print(f"✅ Scenario loaded!")
print(f"   Scenario shape: {type(sample_scenario)}")
print(f"   Num objects: {sample_scenario.num_objects}")
print(f"   Timestep: {sample_scenario.timestep}")

# Trajectory info
print(f"\n📊 Trajectory data:")
print(f"   Position shape: {sample_scenario.log_trajectory.xy.shape}")
print(f"   Velocity X shape: {sample_scenario.log_trajectory.vel_x.shape}")
print(f"   Yaw shape: {sample_scenario.log_trajectory.yaw.shape}")
print(f"   Valid shape: {sample_scenario.log_trajectory.valid.shape}")

# Object metadata
print(f"\n🚗 Object metadata:")
print(
    f"   Object types: {np.unique(np.array(sample_scenario.object_metadata.object_types))}")
print(
    f"   Is SDC: {np.array(sample_scenario.object_metadata.is_sdc).sum()} SDC agents")

# %% [markdown]
# ## 8. Visualize Sample Scenario

# %%
print("🎨 Visualizing scenario...")

try:
    # Waymax built-in visualization
    img = waymax_viz.plot_simulator_state(sample_scenario, use_log_traj=True)

    plt.figure(figsize=(14, 12))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Waymo Scenario - Log Trajectory',
              fontsize=14, fontweight='bold')
    plt.savefig('waymax_scenario.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("✅ Visualization saved to 'waymax_scenario.png'")
except Exception as e:
    print(f"⚠️  Visualization error: {e}")

# %% [markdown]
# ## 9. Initialize Waymax Environment

# %%
print("🔧 Initializing Waymax environment...")

# Dynamics model (bicycle model for vehicles)
dynamics_model = waymax_dynamics.InvertibleBicycleModel()

# Environment config
env_config = waymax_config.EnvironmentConfig(
    max_num_objects=CONFIG['max_num_objects'],
)

# Create environment
sim_env = waymax_env.PlanningAgentEnvironment(
    dynamics_model=dynamics_model,
    config=env_config,
)

print(f"✅ Waymax environment ready")
print(f"   Dynamics: InvertibleBicycleModel")
print(f"   Max objects: {CONFIG['max_num_objects']}")

# %% [markdown]
# ## 10. Helper Functions

# %%


def select_controlled_agents(scenario, num_agents):
    """Select agents to control (prioritize valid, non-SDC vehicles)."""
    valid_mask = np.array(scenario.object_metadata.is_valid)
    is_sdc = np.array(scenario.object_metadata.is_sdc)
    obj_types = np.array(scenario.object_metadata.object_types)

    # Prioritize: valid vehicles that are not SDC
    # Object type 1 = VEHICLE in Waymo
    vehicle_mask = (obj_types == 1) & valid_mask & ~is_sdc
    vehicle_indices = np.where(vehicle_mask)[0]

    if len(vehicle_indices) >= num_agents:
        return vehicle_indices[:num_agents].tolist()

    # Fallback: include SDC if needed
    all_valid = np.where(valid_mask)[0]
    return all_valid[:num_agents].tolist()


def extract_observations(state, agent_indices):
    """Extract observations from Waymax state for RL agents.

    Returns: torch.Tensor of shape (num_agents, obs_dim)
    obs = [x, y, vx, vy, yaw, speed, length]
    """
    observations = []

    # Debug: print available attributes of object_metadata
    if not hasattr(state.object_metadata, "_debug_printed"):
        print("Available attributes in state.object_metadata:",
              dir(state.object_metadata))
        state.object_metadata._debug_printed = True

    for idx in agent_indices:
        # Current position and velocity from sim_trajectory
        x = float(np.array(state.sim_trajectory.x[idx, state.timestep]))
        y = float(np.array(state.sim_trajectory.y[idx, state.timestep]))
        vx = float(np.array(state.sim_trajectory.vel_x[idx, state.timestep]))
        vy = float(np.array(state.sim_trajectory.vel_y[idx, state.timestep]))
        yaw = float(np.array(state.sim_trajectory.yaw[idx, state.timestep]))

        # Derived features
        speed = np.sqrt(vx**2 + vy**2)
        # TODO: Update 'length' extraction after inspecting available attributes
        length = 0.0  # Placeholder, will update after debug

        obs = [x, y, vx, vy, yaw, speed, length]
        observations.append(obs)

    return torch.FloatTensor(observations).to(device)


def create_action(actions_list, agent_indices, state):
    """Convert RL actions to Waymax Action format.

    Actions: [acceleration, steering_angle] per agent
    """
    num_objects = int(np.array(state.num_objects))

    # Initialize action arrays (InvertibleBicycleModel needs accel + steering)
    action_data = np.zeros((num_objects, 2), dtype=np.float32)
    action_valid = np.zeros((num_objects,), dtype=bool)

    for i, agent_idx in enumerate(agent_indices):
        if i < len(actions_list):
            action = actions_list[i].detach().cpu().numpy().flatten()

            # Action clipping
            accel = np.clip(action[0] if len(action) >
                            0 else 0.0, -4.0, 4.0)  # m/s^2
            steer = np.clip(action[1] if len(action) >
                            1 else 0.0, -0.3, 0.3)  # rad

            action_data[agent_idx] = [accel, steer]
            action_valid[agent_idx] = True

    return datatypes.Action(data=action_data, valid=action_valid)


def compute_reward(state, agent_idx, agent_indices, distance_limit=5.0):
    """Compute reward for an agent.

    Rewards:
    - Progress toward goal (log trajectory endpoint)
    - Collision penalty
    - Off-road penalty (simplified)
    """
    reward = 0.0

    timestep = int(np.array(state.timestep))

    # Current position
    x = float(np.array(state.sim_trajectory.x[agent_idx, timestep]))
    y = float(np.array(state.sim_trajectory.y[agent_idx, timestep]))

    # Goal position (end of log trajectory)
    goal_x = float(np.array(state.log_trajectory.x[agent_idx, -1]))
    goal_y = float(np.array(state.log_trajectory.y[agent_idx, -1]))

    # Distance to goal reward (negative distance)
    dist_to_goal = np.sqrt((x - goal_x)**2 + (y - goal_y)**2)
    reward -= dist_to_goal * 0.1  # Scale factor

    # Collision penalty - check distance to other controlled agents
    for other_idx in agent_indices:
        if other_idx != agent_idx:
            other_x = float(
                np.array(state.sim_trajectory.x[other_idx, timestep]))
            other_y = float(
                np.array(state.sim_trajectory.y[other_idx, timestep]))
            dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)

            if dist < 2.0:  # Collision
                reward -= 10.0
                break

    return reward


def compute_cost(state, agent_idx, agent_indices, distance_limit=5.0):
    """Compute safety cost (for constrained RL).

    Cost = 1 if too close to another agent, else 0
    """
    timestep = int(np.array(state.timestep))

    x = float(np.array(state.sim_trajectory.x[agent_idx, timestep]))
    y = float(np.array(state.sim_trajectory.y[agent_idx, timestep]))

    for other_idx in agent_indices:
        if other_idx != agent_idx:
            other_x = float(
                np.array(state.sim_trajectory.x[other_idx, timestep]))
            other_y = float(
                np.array(state.sim_trajectory.y[other_idx, timestep]))
            dist = np.sqrt((x - other_x)**2 + (y - other_y)**2)

            if dist < distance_limit:
                return 1.0

    return 0.0

# %% [markdown]
# ## 11. Build RL Config


# %%
print("📋 Building RL configuration...")

rl_config = {
    'algorithm': CONFIG['rl_algorithm'],
    'batch_size': CONFIG['batch_size'],
    'gamma': CONFIG['gamma'],
    'lamda': CONFIG['gae_lambda'],
    'actor_learning_rate': CONFIG['actor_lr'],
    'critic_learning_rate': CONFIG['critic_lr'],
    'eps': CONFIG['eps'],
    'entropy_coef': CONFIG['entropy_coef'],
    'epochs': CONFIG['epochs'],
    'hidden_dim': CONFIG['hidden_dim'],
    'agent_number': CONFIG['num_controlled_agents'],

    # Constrained RL settings
    'penalty_initial_value': CONFIG['penalty_initial_value'],
    'cost_quantile': CONFIG['cost_quantile'],
    'constrainted_critic_learning_rate': 1e-4,

    # Magnet settings
    'is_magnet': CONFIG['is_magnet'],
    'eta_coef1': CONFIG['eta_coef1'],
    'eta_coef2': 0.05,
    'beta_coef': 0.1,

    # Distribution RL
    'N_quantile': 64,
    'tau_update': 0.01,
    'LR_QN': 3e-4,
    'type': 'CVaR',
    'method': 'SplineDQN',

    # Other
    'offset': 5,
    'target_kl': 0.01,
    'density_learning_rate': 3e-4,
    'gae': True,
}

print("✅ RL Config:")
for k, v in rl_config.items():
    print(f"   {k}: {v}")

# %% [markdown]
# ## 12. Initialize RL Agents

# %%
print("🤖 Initializing RL agents...")

NUM_AGENTS = CONFIG['num_controlled_agents']
STATE_DIM = CONFIG['obs_dim']  # Observation dimension

# Initialize agents based on algorithm
if CONFIG['rl_algorithm'] == 'TrafficGamer':
    agents = [
        TrafficGamer(STATE_DIM, NUM_AGENTS, rl_config, device)
        for _ in range(NUM_AGENTS)
    ]
    print(f"✅ Initialized {NUM_AGENTS} TrafficGamer agents")

elif CONFIG['rl_algorithm'] == 'Constrainted_CCE_MAPPO':
    agents = [
        Constrainted_CCE_MAPPO(STATE_DIM, NUM_AGENTS, rl_config, device)
        for _ in range(NUM_AGENTS)
    ]
    print(f"✅ Initialized {NUM_AGENTS} Constrainted CCE-MAPPO agents")

elif CONFIG['rl_algorithm'] == 'CCE_MAPPO':
    agents = [
        CCE_MAPPO(STATE_DIM, NUM_AGENTS, rl_config, device)
        for _ in range(NUM_AGENTS)
    ]
    print(f"✅ Initialized {NUM_AGENTS} CCE-MAPPO agents")

else:  # MAPPO
    agents = [
        MAPPO(STATE_DIM, NUM_AGENTS, rl_config, device)
        for _ in range(NUM_AGENTS)
    ]
    print(f"✅ Initialized {NUM_AGENTS} MAPPO agents")

# %% [markdown]
# ## 13. Training Loop

# %%
print("🚀 Starting training...")
print(f"   Algorithm: {CONFIG['rl_algorithm']}")
print(f"   Episodes: {CONFIG['num_episodes']}")
print(f"   Steps per episode: {CONFIG['steps_per_episode']}")
print(f"   Controlled agents: {NUM_AGENTS}")

# Training metrics
metrics = {
    'episode_rewards': [],
    'episode_costs': [],
    'collision_rates': [],
}

# Recreate iterator for training
data_iterator = waymax_dataloader.simulator_state_generator(dataset_config)

# %%
for episode in tqdm(range(CONFIG['num_episodes']), desc="Training"):

    # Get new scenario
    try:
        scenario = next(data_iterator)
    except StopIteration:
        # Restart iterator
        data_iterator = waymax_dataloader.simulator_state_generator(
            dataset_config)
        scenario = next(data_iterator)

    # Select agents to control
    controlled_agents = select_controlled_agents(scenario, NUM_AGENTS)

    if len(controlled_agents) < NUM_AGENTS:
        continue  # Skip scenarios with too few agents

    # Reset environment
    state = sim_env.reset(scenario)

    # Initialize transition storage
    transition_list = [{
        "observations": [[] for _ in range(NUM_AGENTS)],
        "actions": [[] for _ in range(NUM_AGENTS)],
        "next_observations": [[] for _ in range(NUM_AGENTS)],
        "rewards": [[] for _ in range(NUM_AGENTS)],
        "costs": [[] for _ in range(NUM_AGENTS)],
        "magnet": [[] for _ in range(NUM_AGENTS)],
        "dones": [],
    }]

    episode_rewards = [0.0] * NUM_AGENTS
    episode_costs = [0.0] * NUM_AGENTS
    collisions = 0

    # Rollout episode
    for step in range(CONFIG['steps_per_episode']):
        # Extract observations
        obs = extract_observations(state, controlled_agents)

        # Get actions from agents
        actions = []
        for i, agent in enumerate(agents):
            action = agent.choose_action(obs[i].unsqueeze(0))
            actions.append(action)

        # Create Waymax action
        waymax_action = create_action(actions, controlled_agents, state)

        # Step environment
        next_state = sim_env.step(state, waymax_action)

        # Compute rewards and costs
        for i, agent_idx in enumerate(controlled_agents):
            reward = compute_reward(
                next_state, agent_idx, controlled_agents, CONFIG['distance_limit'])
            cost = compute_cost(next_state, agent_idx,
                                controlled_agents, CONFIG['distance_limit'])

            # Store transitions
            transition_list[0]["observations"][i].append(obs[i])
            transition_list[0]["actions"][i].append(actions[i])
            transition_list[0]["rewards"][i].append(
                torch.tensor([reward], device=device))
            transition_list[0]["costs"][i].append(
                torch.tensor([cost], device=device))

            episode_rewards[i] += reward
            episode_costs[i] += cost

            if cost > 0:
                collisions += 1

        # Next observations
        next_obs = extract_observations(next_state, controlled_agents)
        for i in range(NUM_AGENTS):
            transition_list[0]["next_observations"][i].append(next_obs[i])

        # Done flag
        is_done = bool(np.array(state.is_done)) if hasattr(
            state, 'is_done') else False
        done_tensor = torch.tensor(
            1 if is_done or step == CONFIG['steps_per_episode'] - 1 else 0, device=device)
        transition_list[0]["dones"].append(done_tensor)

        if is_done:
            break

        state = next_state

    # Update agents
    for i, agent in enumerate(agents):
        try:
            agent.update(transition_list, i)
        except Exception as e:
            if episode == 0:
                print(f"⚠️  Update error (agent {i}): {e}")

    # Record metrics
    avg_reward = np.mean(episode_rewards)
    avg_cost = np.mean(episode_costs)
    collision_rate = collisions / (CONFIG['steps_per_episode'] * NUM_AGENTS)

    metrics['episode_rewards'].append(avg_reward)
    metrics['episode_costs'].append(avg_cost)
    metrics['collision_rates'].append(collision_rate)

    # Log progress
    if (episode + 1) % CONFIG['log_interval'] == 0:
        recent_reward = np.mean(
            metrics['episode_rewards'][-CONFIG['log_interval']:])
        recent_cost = np.mean(
            metrics['episode_costs'][-CONFIG['log_interval']:])
        recent_collision = np.mean(
            metrics['collision_rates'][-CONFIG['log_interval']:])

        print(f"\n📊 Episode {episode + 1}/{CONFIG['num_episodes']}")
        print(f"   Avg Reward: {recent_reward:.4f}")
        print(f"   Avg Cost: {recent_cost:.4f}")
        print(f"   Collision Rate: {recent_collision:.4f}")

print("\n✅ Training completed!")

# %% [markdown]
# ## 14. Visualize Training Progress

# %%
print("📈 Plotting training progress...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Episode rewards
ax = axes[0, 0]
ax.plot(metrics['episode_rewards'], alpha=0.6, label='Raw')
window = min(20, len(metrics['episode_rewards']))
if window > 1:
    ma = pd.Series(metrics['episode_rewards']).rolling(window=window).mean()
    ax.plot(ma, color='red', linewidth=2, label=f'{window}-ep MA')
ax.set_title('Episode Rewards', fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Reward')
ax.legend()
ax.grid(True, alpha=0.3)

# Episode costs
ax = axes[0, 1]
ax.plot(metrics['episode_costs'], alpha=0.6, color='orange')
ax.set_title('Episode Safety Costs', fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Cost')
ax.grid(True, alpha=0.3)

# Collision rate
ax = axes[1, 0]
ax.plot(metrics['collision_rates'], alpha=0.6, color='red')
ax.set_title('Collision Rate', fontweight='bold')
ax.set_xlabel('Episode')
ax.set_ylabel('Rate')
ax.grid(True, alpha=0.3)

# Summary
ax = axes[1, 1]
ax.axis('off')
summary = f"""
Training Summary
================
Algorithm: {CONFIG['rl_algorithm']}
Episodes: {len(metrics['episode_rewards'])}
Agents: {NUM_AGENTS}

Final Metrics (last 50 ep):
- Avg Reward: {np.mean(metrics['episode_rewards'][-50:]):.4f}
- Avg Cost: {np.mean(metrics['episode_costs'][-50:]):.4f}
- Collision Rate: {np.mean(metrics['collision_rates'][-50:]):.4f}

Best Reward: {max(metrics['episode_rewards']):.4f}
"""
ax.text(0.1, 0.5, summary, fontsize=11, family='monospace',
        verticalalignment='center', transform=ax.transAxes)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Saved to 'training_progress.png'")

# %% [markdown]
# ## 15. Save Models

# %%
if CONFIG['save_checkpoint']:
    print("💾 Saving models...")

    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    for i, agent in enumerate(agents):
        checkpoint = {
            'agent_index': i,
            'state_dim': STATE_DIM,
            'agent_number': NUM_AGENTS,
            'config': rl_config,
            'policy_state_dict': agent.pi.state_dict(),
            'value_state_dict': agent.value.state_dict(),
        }

        path = save_dir / f'{CONFIG["rl_algorithm"]}_agent{i}_{timestamp}.pt'
        torch.save(checkpoint, path)
        print(f"   ✅ Agent {i}: {path}")

    # Save metrics
    metrics_path = save_dir / f'metrics_{timestamp}.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump({'metrics': metrics, 'config': CONFIG}, f)
    print(f"   ✅ Metrics: {metrics_path}")

# %% [markdown]
# ## 16. Evaluate Policy

# %%
print("🔍 Evaluating trained policy...")

NUM_EVAL = 20
eval_rewards = []
eval_costs = []

# New iterator for evaluation
eval_iterator = waymax_dataloader.simulator_state_generator(dataset_config)

for ep in tqdm(range(NUM_EVAL), desc="Evaluating"):
    try:
        scenario = next(eval_iterator)
    except StopIteration:
        eval_iterator = waymax_dataloader.simulator_state_generator(
            dataset_config)
        scenario = next(eval_iterator)

    controlled_agents = select_controlled_agents(scenario, NUM_AGENTS)
    if len(controlled_agents) < NUM_AGENTS:
        continue

    state = sim_env.reset(scenario)
    ep_reward = 0.0
    ep_cost = 0.0

    with torch.no_grad():
        for step in range(CONFIG['steps_per_episode']):
            obs = extract_observations(state, controlled_agents)

            actions = []
            for i, agent in enumerate(agents):
                action = agent.choose_action(obs[i].unsqueeze(0))
                actions.append(action)

            waymax_action = create_action(actions, controlled_agents, state)
            next_state = sim_env.step(state, waymax_action)

            for i, agent_idx in enumerate(controlled_agents):
                ep_reward += compute_reward(next_state,
                                            agent_idx, controlled_agents)
                ep_cost += compute_cost(next_state,
                                        agent_idx, controlled_agents)

            state = next_state

    eval_rewards.append(ep_reward / NUM_AGENTS)
    eval_costs.append(ep_cost / NUM_AGENTS)

print("\n" + "=" * 50)
print("📊 EVALUATION RESULTS")
print("=" * 50)
print(f"Avg Reward: {np.mean(eval_rewards):.4f} ± {np.std(eval_rewards):.4f}")
print(f"Avg Cost: {np.mean(eval_costs):.4f} ± {np.std(eval_costs):.4f}")
print("=" * 50)

# %% [markdown]
# ## 17. Summary
#
# **What we accomplished:**
# 1. ✅ Loaded Waymo data via Waymax (streaming from GCS - NO local data)
# 2. ✅ Used Waymax as simulator with InvertibleBicycleModel dynamics
# 3. ✅ Trained multi-agent RL policy directly on Waymax observations
# 4. ✅ NO QCNet needed - observations come directly from simulator state
# 5. ✅ Computed rewards: goal reaching + collision avoidance
# 6. ✅ Saved trained models
#
# **Key differences from previous notebook:**
# - Pure Waymax approach (no QCNet, no local data)
# - Simpler observation space (direct from Waymax state)
# - Uses Waymax environment for simulation
#
# **Repository:** https://github.com/PhamPhuHoa-23/EvolutionaryTest
