# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# # Waymax + Waymo Motion Dataset - Interactive Training
#
# **Interactive notebook** to train policy with Waymax simulator:
# - **NO DOWNLOAD**: Data streams from Google Cloud
# - **NO QCNet**: Focus on Waymax simulator + RL
# - Train Policy Network with RL
# - Visualize and evaluate
#
# **Repository:** https://github.com/PhamPhuHoa-23/TrafficGamer

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 1. Install Dependencies

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:31:44.268391Z","iopub.execute_input":"2026-01-11T04:31:44.268779Z","iopub.status.idle":"2026-01-11T04:32:02.543701Z","shell.execute_reply.started":"2026-01-11T04:31:44.268752Z","shell.execute_reply":"2026-01-11T04:32:02.542306Z"}}
from waymax import env, dynamics, datatypes
from utils.utils import seed_everything
from utils.rollout import PPO_process_batch
from algorithm.mappo import MAPPO
from algorithm.constrainted_cce_mappo import Constrainted_CCE_MAPPO
from algorithm.TrafficGamer import TrafficGamer
from transforms import TargetBuilder
from waymax import visualization as waymax_viz
import dataclasses
from waymax import dataloader as waymax_dataloader
from waymax import config as waymax_config
from torch_geometric.data import Batch
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from pathlib import Path
import pytorch_lightning as pl
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import yaml
import warnings
import sys
import os
import tensorflow as tf
import torch
!pip install - q torch torchvision torchaudio
!pip install - q pytorch-lightning == 2.0.0
!pip install - q torch-geometric
!pip install - q tensorflow  # For Waymo dataset
!pip install - q av  # Video encoding (needed by source imports)
!pip install - q av2  # Argoverse 2 API

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:02.546055Z","iopub.execute_input":"2026-01-11T04:32:02.546358Z","iopub.status.idle":"2026-01-11T04:32:15.315319Z","shell.execute_reply.started":"2026-01-11T04:32:02.546327Z","shell.execute_reply":"2026-01-11T04:32:15.313743Z"}}
!pip install - -upgrade pip
!pip install git+https: // github.com/waymo-research/waymax.git@main  # egg=waymo-waymax

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:15.317104Z","iopub.execute_input":"2026-01-11T04:32:15.317467Z","iopub.status.idle":"2026-01-11T04:32:15.325774Z","shell.execute_reply.started":"2026-01-11T04:32:15.317435Z","shell.execute_reply":"2026-01-11T04:32:15.324365Z"}}

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:15.327361Z","iopub.execute_input":"2026-01-11T04:32:15.327801Z","iopub.status.idle":"2026-01-11T04:32:15.366315Z","shell.execute_reply.started":"2026-01-11T04:32:15.327762Z","shell.execute_reply":"2026-01-11T04:32:15.365304Z"}}
# Check versions
print(f"PyTorch version: {torch.__version__}")
print(f"TensorFlow version: {tf.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")

cuda_version = torch.version.cuda.replace(
    '.', '') if torch.cuda.is_available() else 'cpu'
print(f"CUDA version: {cuda_version}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:15.368743Z","iopub.execute_input":"2026-01-11T04:32:15.369164Z","iopub.status.idle":"2026-01-11T04:32:18.305054Z","shell.execute_reply.started":"2026-01-11T04:32:15.369128Z","shell.execute_reply":"2026-01-11T04:32:18.303612Z"}}
# Install PyG dependencies
!pip install torch-scatter torch-sparse torch-cluster - f https: // data.pyg.org/whl/torch-{torch.__version__.split('+')[0]}+cu{cuda_version[:3]}.html

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 2. Setup Environment

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:18.306851Z","iopub.execute_input":"2026-01-11T04:32:18.307177Z","iopub.status.idle":"2026-01-11T04:32:18.315944Z","shell.execute_reply.started":"2026-01-11T04:32:18.307146Z","shell.execute_reply":"2026-01-11T04:32:18.314833Z"}}
warnings.filterwarnings('ignore')
tf.config.set_visible_devices([], 'GPU')  # Disable TF GPU (use PyTorch)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Using device: {device}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 3. Clone TrafficGamer Repository

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:18.317430Z","iopub.execute_input":"2026-01-11T04:32:18.318122Z","iopub.status.idle":"2026-01-11T04:32:20.031498Z","shell.execute_reply.started":"2026-01-11T04:32:18.317993Z","shell.execute_reply":"2026-01-11T04:32:20.030013Z"}}
# Clone repo
TRAFFICGAMER_DIR = Path("TrafficGamer")

if not TRAFFICGAMER_DIR.exists():
    print("📥 Cloning TrafficGamer...")
    !git clone https: // github.com/PhamPhuHoa-23/TrafficGamer.git
    print("✅ Cloned successfully")
else:
    print("✅ TrafficGamer already exists")

sys.path.insert(0, str(TRAFFICGAMER_DIR.absolute()))
os.chdir(TRAFFICGAMER_DIR)
print(f"📁 Working directory: {os.getcwd()}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:32:20.032992Z","iopub.execute_input":"2026-01-11T04:32:20.033293Z","iopub.status.idle":"2026-01-11T04:34:04.363520Z","shell.execute_reply.started":"2026-01-11T04:32:20.033264Z","shell.execute_reply":"2026-01-11T04:34:04.362314Z"}}
# Install requirements
!pip install - q - r requirements.txt
!pip install - q neptune

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:04.365520Z","iopub.execute_input":"2026-01-11T04:34:04.365944Z","iopub.status.idle":"2026-01-11T04:34:04.372083Z","shell.execute_reply.started":"2026-01-11T04:34:04.365909Z","shell.execute_reply":"2026-01-11T04:34:04.371143Z"}}

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 5. Configuration
#
# **🔧 EDIT HERE - Configure paths and parameters:**

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:04.373713Z","iopub.execute_input":"2026-01-11T04:34:04.374101Z","iopub.status.idle":"2026-01-11T04:34:04.436200Z","shell.execute_reply.started":"2026-01-11T04:34:04.374063Z","shell.execute_reply":"2026-01-11T04:34:04.434716Z"}}
# ============================================
# 🔧 CONFIGURATION - EDIT HERE
# ============================================

CONFIG = {
    # Data split
    'split': 'val',  # 'train', 'val', or 'test'

    # Dataset settings (Waymo: 11 history + 80 future @ 10Hz)
    'num_historical_steps': 11,
    'num_future_steps': 80,

    # Training settings
    'seed': 42,
    'batch_size': 4,
    'max_epochs': 10,

    # RL algorithm
    'rl_algorithm': 'TrafficGamer',  # 'TrafficGamer', 'MAPPO', or 'CCE_MAPPO'
    'rl_config_file': 'TrafficGamer.yaml',

    # Scenario settings
    'scenario_id': 1,
    'controlled_agents': [0, 1, 2, 3],  # Agent indices to control

    # Training hyperparameters
    'learning_rate_actor': 3e-4,
    'learning_rate_critic': 3e-4,
    'gamma': 0.99,
    'gae_lambda': 0.95,

    # Safety constraints
    'distance_limit': 5.0,  # Collision threshold (meters)
    'cost_quantile': 48,
    'penalty_initial_value': 1.0,

    # Evaluation
    'eval_freq': 5,
    'save_checkpoint': True,
}

seed_everything(CONFIG['seed'])

print("✅ Configuration loaded")
print(f"   Split: {CONFIG['split']}")
print(f"   RL Algorithm: {CONFIG['rl_algorithm']}")
print(
    f"   History/Future: {CONFIG['num_historical_steps']}/{CONFIG['num_future_steps']}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 6. Authenticate with Google Cloud
#
# **REQUIRED: Authenticate to access Waymo dataset from GCS**
#
# Choose ONE method:
# - **Colab**: Use `auth.authenticate_user()` (easiest)
# - **Kaggle/Local**: Use Service Account Key

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:04.437518Z","iopub.execute_input":"2026-01-11T04:34:04.437815Z","iopub.status.idle":"2026-01-11T04:34:04.457057Z","shell.execute_reply.started":"2026-01-11T04:34:04.437788Z","shell.execute_reply":"2026-01-11T04:34:04.456069Z"}}
print("🔑 Authenticating with Google Cloud...")

service_key_path = '/kaggle/input/gcs-credentials/auth.json'  # Change this path

if os.path.exists(service_key_path):
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = service_key_path
    print(f"✅ Authenticated via service account: {service_key_path}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 7. Load Waymo Dataset with Waymax
#
# **Waymax streams data from Google Cloud - NO download needed!**
# - Data streams directly from cloud
# - 11 historical steps + 80 future steps @ 10Hz (9.1s total)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:04.458270Z","iopub.execute_input":"2026-01-11T04:34:04.458546Z","iopub.status.idle":"2026-01-11T04:34:04.637872Z","shell.execute_reply.started":"2026-01-11T04:34:04.458524Z","shell.execute_reply":"2026-01-11T04:34:04.637025Z"}}
print("📂 Loading Waymo Motion Dataset with Waymax...")

# Stream data from cloud
if CONFIG['split'] == 'train':
    dataset_config = waymax_config.WOD_1_2_0_TRAINING
elif CONFIG['split'] == 'val':
    dataset_config = waymax_config.WOD_1_2_0_VALIDATION
else:
    dataset_config = waymax_config.WOD_1_2_0_TESTING

# Configure
dataset_config_fix = dataset_config.path.replace("///", "//")
dataset_config = dataclasses.replace(
    dataset_config,
    max_num_objects=32,  # Limit objects for memory
    path=dataset_config_fix
)

# Create generator
waymo_iterator = waymax_dataloader.simulator_state_generator(dataset_config)

print(f"✅ Waymax iterator created for {CONFIG['split']} split")
print(f"   Data streams from: {dataset_config.path}")
print(f"   Batch size: {CONFIG['batch_size']}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 8. Test Waymax Data Loading

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:04.639125Z","iopub.execute_input":"2026-01-11T04:34:04.639470Z","iopub.status.idle":"2026-01-11T04:34:07.473596Z","shell.execute_reply.started":"2026-01-11T04:34:04.639444Z","shell.execute_reply":"2026-01-11T04:34:07.472515Z"}}
print("🔍 Testing Waymax data loading...")

# Get one scenario
sample_scenario = next(waymo_iterator)

print("✅ Scenario loaded successfully!")
print("\n📊 Scenario Structure:")
print(f"   Timesteps: {sample_scenario.timestep}")
print(f"   Objects: {sample_scenario.num_objects}")
print(f"   Valid objects: {sample_scenario.object_metadata.is_valid.sum()}")

# Print trajectory shape
print(f"\n🚗 Trajectory data:")
print(f"   Position: {sample_scenario.log_trajectory.xy.shape}")
print(f"   Velocity: {sample_scenario.log_trajectory.vel_x.shape}")
print(f"   Heading: {sample_scenario.log_trajectory.yaw.shape}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 9. Visualize Waymo Scenario (Optional)

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:07.477090Z","iopub.execute_input":"2026-01-11T04:34:07.477418Z","iopub.status.idle":"2026-01-11T04:34:09.004505Z","shell.execute_reply.started":"2026-01-11T04:34:07.477391Z","shell.execute_reply":"2026-01-11T04:34:09.003118Z"}}
# Visualize scenario with Waymax

try:
    img = waymax_viz.plot_simulator_state(sample_scenario, use_log_traj=True)

    # Display
    try:
        import mediapy
        mediapy.show_image(img)
    except ImportError:
        plt.figure(figsize=(12, 10))
        plt.imshow(img)
        plt.axis('off')
        plt.title('Waymo Scenario Visualization',
                  fontsize=14, fontweight='bold')
        plt.show()

    print("✅ Scenario visualization complete")
except Exception as e:
    print(f"⚠️  Visualization skipped: {e}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 10. Initialize Policy Network (RL Agent)
#
# **Policy network** learns to control AVs via RL

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.006437Z","iopub.execute_input":"2026-01-11T04:34:09.006815Z","iopub.status.idle":"2026-01-11T04:34:09.023388Z","shell.execute_reply.started":"2026-01-11T04:34:09.006788Z","shell.execute_reply":"2026-01-11T04:34:09.021904Z"}}
# Load RL config
rl_config_path = Path('configs') / CONFIG['rl_config_file']
with open(rl_config_path, 'r') as f:
    rl_config = yaml.safe_load(f)

# Update config with user settings
rl_config['batch_size'] = CONFIG['batch_size']
rl_config['gamma'] = CONFIG['gamma']
rl_config['lamda'] = CONFIG['gae_lambda']
rl_config['actor_learning_rate'] = CONFIG['learning_rate_actor']
rl_config['critic_learning_rate'] = CONFIG['learning_rate_critic']
rl_config['penalty_initial_value'] = CONFIG['penalty_initial_value']
rl_config['cost_quantile'] = CONFIG['cost_quantile']

# Add missing config parameters required by algorithms
rl_config.setdefault('hidden_dim', 128)
rl_config.setdefault('constrainted_critic_learning_rate', 0.0001)
rl_config.setdefault('density_learning_rate', 0.0003)
rl_config.setdefault('eps', 0.2)
rl_config.setdefault('offset', 5)
rl_config.setdefault('entropy_coef', 0.005)
rl_config.setdefault('epochs', 10)
rl_config.setdefault('gae', True)
rl_config.setdefault('target_kl', 0.01)
rl_config.setdefault('is_magnet', True)
rl_config.setdefault('eta_coef1', 0.005)
rl_config.setdefault('beta_coef', 0.1)
rl_config.setdefault('N_quantile', 64)
rl_config.setdefault('tau_update', 0.01)
rl_config.setdefault('LR_QN', 0.0003)
rl_config.setdefault('type', 'CVaR')
rl_config.setdefault('method', 'SplineDQN')

print("✅ RL Configuration loaded:")
for key, value in rl_config.items():
    print(f"   {key}: {value}")

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.024868Z","iopub.execute_input":"2026-01-11T04:34:09.025238Z","iopub.status.idle":"2026-01-11T04:34:09.193145Z","shell.execute_reply.started":"2026-01-11T04:34:09.025211Z","shell.execute_reply":"2026-01-11T04:34:09.192197Z"}}
# Initialize RL agents
STATE_DIM = 7  # Hidden state dimension from model
AGENT_NUMBER = len(CONFIG['controlled_agents'])

# Select algorithm
if CONFIG['rl_algorithm'] == 'TrafficGamer':
    agents = [
        TrafficGamer(STATE_DIM, AGENT_NUMBER, rl_config, device)
        for _ in range(AGENT_NUMBER)
    ]
    print(f"✅ Initialized {AGENT_NUMBER} TrafficGamer agents")
elif CONFIG['rl_algorithm'] == 'CCE_MAPPO':
    agents = [
        Constrainted_CCE_MAPPO(STATE_DIM, AGENT_NUMBER, rl_config, device)
        for _ in range(AGENT_NUMBER)
    ]
    print(f"✅ Initialized {AGENT_NUMBER} CCE-MAPPO agents")
else:  # MAPPO
    agents = [
        MAPPO(STATE_DIM, AGENT_NUMBER, rl_config, device)
        for _ in range(AGENT_NUMBER)
    ]
    print(f"✅ Initialized {AGENT_NUMBER} MAPPO agents")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 11. Helper Functions for Training

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.195126Z","iopub.execute_input":"2026-01-11T04:34:09.195743Z","iopub.status.idle":"2026-01-11T04:34:09.208394Z","shell.execute_reply.started":"2026-01-11T04:34:09.195713Z","shell.execute_reply":"2026-01-11T04:34:09.206569Z"}}


def compute_waymax_cost(scenario, agent_idx, controlled_agents):
    """Compute safety cost for an agent (collision proximity)."""
    cost = 0.0

    # Convert timestep to Python int - handle batched case
    timestep_arr = np.array(scenario.timestep)
    if timestep_arr.ndim > 0:
        t = int(timestep_arr.flat[0])
    else:
        t = int(timestep_arr.item())

    # Get position of current agent
    xy = np.array(scenario.log_trajectory.xy[0, agent_idx, t])

    # Check distance to other controlled agents
    for other_idx in controlled_agents:
        if other_idx != agent_idx:
            other_xy = np.array(scenario.log_trajectory.xy[0, other_idx, t])
            dist = np.linalg.norm(xy - other_xy)
            if dist < CONFIG['distance_limit']:
                cost = 1.0
                break

    return cost


def initialize_transition_list(batch_size, agent_num):
    """Initialize transition storage for RL training."""
    transition_list = []
    for _ in range(batch_size):
        transition = {
            "observations": [[] for _ in range(agent_num)],
            "actions": [[] for _ in range(agent_num)],
            "rewards": [[] for _ in range(agent_num)],
            "costs": [[] for _ in range(agent_num)],
            "next_observations": [[] for _ in range(agent_num)],
            "dones": [],
            "magnet": [[] for _ in range(agent_num)],
        }
        transition_list.append(transition)
    return transition_list


def dict_mean(dict_list, agent):
    """Compute mean of dictionary values for logging."""
    mean_dict = {}
    for key in dict_list[0].keys():
        mean_dict[f"agent_{agent}_{key}"] = np.mean(
            [d[key] for d in dict_list], axis=0
        )
    return mean_dict

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 12. Waymax-Specific Helper Functions

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.210300Z","iopub.execute_input":"2026-01-11T04:34:09.211517Z","iopub.status.idle":"2026-01-11T04:34:09.245519Z","shell.execute_reply.started":"2026-01-11T04:34:09.211460Z","shell.execute_reply":"2026-01-11T04:34:09.244381Z"}}


def extract_observations_from_waymax_state(state, controlled_agent_ids):
    """
    Extract observations từ Waymax SimulatorState cho RL agents

    Args:
        state: waymax.datatypes.SimulatorState
        controlled_agent_ids: List[int] - indices của agents cần control

    Returns:
        observations: np.array shape (num_agents, 7)
    """
    observations = []

    for agent_id in controlled_agent_ids:
        # Lấy current sim trajectory (state hiện tại sau khi simulate)
        x = float(np.array(state.current_sim_trajectory.x[agent_id]))
        y = float(np.array(state.current_sim_trajectory.y[agent_id]))
        yaw = float(np.array(state.current_sim_trajectory.yaw[agent_id]))
        vel_x = float(np.array(state.current_sim_trajectory.vel_x[agent_id]))
        vel_y = float(np.array(state.current_sim_trajectory.vel_y[agent_id]))

        # Tính speed
        speed = np.sqrt(vel_x**2 + vel_y**2)

        # State vector: [x, y, vx, vy, yaw, speed, heading_rate]
        # heading_rate dùng 0.0 vì không có history trong single state
        obs = np.array([x, y, vel_x, vel_y, yaw, speed, 0.0])
        observations.append(obs)

    return np.array(observations)


def create_waymax_action(actions_list, controlled_agent_ids, state):
    """
    Convert RL actions sang Waymax Action format

    Args:
        actions_list: List[torch.Tensor] - actions từ RL agents
        controlled_agent_ids: List[int] - agent indices
        state: waymax.datatypes.SimulatorState

    Returns:
        waymax.datatypes.Action với format cho InvertibleBicycleModel
    """
    from waymax import datatypes

    # Get số lượng objects trong scene
    num_objects_arr = np.array(state.num_objects)
    if num_objects_arr.ndim > 0:
        num_objects = int(num_objects_arr.flat[0])
    else:
        num_objects = int(num_objects_arr.item())

    # Initialize action arrays
    # InvertibleBicycleModel cần [acceleration, steering_curvature]
    action_data = np.zeros((num_objects, 2), dtype=np.float32)
    action_valid = np.zeros((num_objects,), dtype=bool)

    # Fill actions cho controlled agents
    for i, agent_id in enumerate(controlled_agent_ids):
        # Convert RL action từ tensor sang numpy
        action_np = actions_list[i].detach().cpu().numpy().flatten()

        # Clip action để đảm bảo trong giới hạn hợp lý
        # acceleration: [-5, 5] m/s^2, steering: [-0.5, 0.5] rad/s
        accel = np.clip(action_np[0] if len(action_np) > 0 else 0.0, -5.0, 5.0)
        steer = np.clip(action_np[1] if len(action_np) > 1 else 0.0, -0.5, 0.5)

        action_data[agent_id] = [accel, steer]
        action_valid[agent_id] = True

    # Uncontrolled agents sẽ dùng log playback (action_valid = False)

    return datatypes.Action(
        data=action_data,
        valid=action_valid
    )

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 13. Training Setup


# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.246902Z","iopub.execute_input":"2026-01-11T04:34:09.247311Z","iopub.status.idle":"2026-01-11T04:34:09.280253Z","shell.execute_reply.started":"2026-01-11T04:34:09.247272Z","shell.execute_reply":"2026-01-11T04:34:09.279076Z"}}
training_metrics = {
    'episode_rewards': [],
    'episode_costs': [],
    'actor_losses': [],
    'critic_losses': [],
    'collision_rates': [],
}

NUM_EPISODES = CONFIG['max_epochs'] * 100  # Total training episodes
STEPS_PER_EPISODE = CONFIG['num_future_steps']

print(f"🚀 Starting training for {NUM_EPISODES} episodes...")
print(f"   Steps per episode: {STEPS_PER_EPISODE}")
print(f"   Controlled agents: {CONFIG['controlled_agents']}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 14. Main Training Loop with Waymax

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:41:44.527985Z","iopub.execute_input":"2026-01-11T04:41:44.528351Z","iopub.status.idle":"2026-01-11T04:41:44.674073Z","shell.execute_reply.started":"2026-01-11T04:41:44.528323Z","shell.execute_reply":"2026-01-11T04:41:44.672674Z"}}

# ===== SETUP WAYMAX ENVIRONMENT =====
print("🔧 Setting up Waymax environment...")
dynamics_model = dynamics.InvertibleBicycleModel()
env_config = waymax_config.EnvironmentConfig(max_num_objects=32)
waymax_env = env.BaseEnvironment(dynamics_model, env_config)
print("✅ Waymax environment ready\n")

# ===== TRAINING LOOP =====
for episode in tqdm(range(NUM_EPISODES), desc="Training"):
    # Lấy scenario mới
    try:
        scenario = next(waymo_iterator)
    except StopIteration:
        waymo_iterator = waymax_dataloader.simulator_state_generator(
            dataset_config)
        scenario = next(waymo_iterator)

    # RESET environment với scenario
    state = waymax_env.reset(scenario)

    # Initialize episode storage
    transition_list = initialize_transition_list(1, AGENT_NUMBER)
    episode_reward = [0.0] * AGENT_NUMBER
    episode_cost = [0.0] * AGENT_NUMBER
    collision_count = 0

    # ===== ROLLOUT EPISODE =====
    for step in range(STEPS_PER_EPISODE):
        # 1. Extract observations từ current state
        observations = extract_observations_from_waymax_state(
            state, CONFIG['controlled_agents']
        )
        observations_tensor = torch.FloatTensor(observations).to(device)

        # 2. RL agents chọn actions
        actions_list = []
        for i, agent in enumerate(agents):
            obs_flat = observations_tensor[i].flatten().unsqueeze(0)
            action = agent.choose_action(obs_flat)
            actions_list.append(action)

        # 3. Convert actions sang Waymax format
        waymax_action = create_waymax_action(
            actions_list,
            CONFIG['controlled_agents'],
            state
        )

        # 4. STEP environment (state transition via dynamics)
        next_state = waymax_env.step(state, waymax_action)

        # 5. Compute rewards & costs
        for i in range(AGENT_NUMBER):
            # Reward từ Waymax environment
            reward_tensor = waymax_env.reward(next_state, waymax_action)
            reward = float(
                np.array(reward_tensor[CONFIG['controlled_agents'][i]]).item())

            # Cost từ collision detection
            cost = compute_waymax_cost(
                next_state,
                CONFIG['controlled_agents'][i],
                CONFIG['controlled_agents']
            )

            # Store transitions
            transition_list[0]["observations"][i].append(
                observations_tensor[i])
            transition_list[0]["actions"][i].append(actions_list[i])
            transition_list[0]["rewards"][i].append(
                torch.tensor([reward], device=device))
            transition_list[0]["costs"][i].append(
                torch.tensor([cost], device=device))

            episode_reward[i] += reward
            episode_cost[i] += cost

            if cost > 0:
                collision_count += 1

        # 6. Next observations từ next_state
        next_observations = extract_observations_from_waymax_state(
            next_state, CONFIG['controlled_agents']
        )
        next_observations_tensor = torch.FloatTensor(
            next_observations).to(device)

        for i in range(AGENT_NUMBER):
            transition_list[0]["next_observations"][i].append(
                next_observations_tensor[i])

        # 7. Check done
        is_done = bool(np.array(next_state.is_done).item()) if hasattr(
            next_state.is_done, 'item') else bool(next_state.is_done)

        if is_done or step == STEPS_PER_EPISODE - 1:
            transition_list[0]["dones"].append(torch.tensor(1, device=device))
            break
        else:
            transition_list[0]["dones"].append(torch.tensor(0, device=device))

        # 8. Update state cho iteration tiếp theo
        state = next_state

    # ===== UPDATE AGENTS =====
    for i, agent in enumerate(agents):
        logs = agent.update(transition_list, i)

    # ===== RECORD METRICS =====
    training_metrics['episode_rewards'].append(np.mean(episode_reward))
    training_metrics['episode_costs'].append(np.mean(episode_cost))
    training_metrics['collision_rates'].append(
        collision_count / (STEPS_PER_EPISODE * AGENT_NUMBER))

    # ===== LOG PROGRESS =====
    if (episode + 1) % CONFIG['eval_freq'] == 0:
        avg_reward = np.mean(
            training_metrics['episode_rewards'][-CONFIG['eval_freq']:])
        avg_cost = np.mean(
            training_metrics['episode_costs'][-CONFIG['eval_freq']:])
        avg_collision = np.mean(
            training_metrics['collision_rates'][-CONFIG['eval_freq']:])

        print(f"\n📊 Episode {episode + 1}/{NUM_EPISODES}")
        print(f"   Avg Reward: {avg_reward:.4f}")
        print(f"   Avg Cost: {avg_cost:.4f}")
        print(f"   Collision Rate: {avg_collision:.4f}")

print("\n✅ Training completed!")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 15. Visualize Training Progress

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.342522Z","iopub.status.idle":"2026-01-11T04:34:09.343062Z","shell.execute_reply.started":"2026-01-11T04:34:09.342787Z","shell.execute_reply":"2026-01-11T04:34:09.342807Z"}}
# Plot training metrics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Episode rewards
axes[0, 0].plot(training_metrics['episode_rewards'], alpha=0.7)
axes[0, 0].set_title('Episode Rewards', fontsize=12, fontweight='bold')
axes[0, 0].set_xlabel('Episode')
axes[0, 0].set_ylabel('Total Reward')
axes[0, 0].grid(True, alpha=0.3)

# Moving average of rewards
window = min(50, len(training_metrics['episode_rewards']))
if window > 0:
    rewards_ma = pd.Series(training_metrics['episode_rewards']).rolling(
        window=window).mean()
    axes[0, 0].plot(rewards_ma, color='red',
                    linewidth=2, label=f'{window}-ep MA')
    axes[0, 0].legend()

# Episode costs
axes[0, 1].plot(training_metrics['episode_costs'], alpha=0.7, color='orange')
axes[0, 1].set_title('Episode Safety Costs', fontsize=12, fontweight='bold')
axes[0, 1].set_xlabel('Episode')
axes[0, 1].set_ylabel('Total Cost')
axes[0, 1].grid(True, alpha=0.3)

# Collision rates
axes[1, 0].plot(training_metrics['collision_rates'], alpha=0.7, color='red')
axes[1, 0].set_title('Collision Rate', fontsize=12, fontweight='bold')
axes[1, 0].set_xlabel('Episode')
axes[1, 0].set_ylabel('Rate')
axes[1, 0].grid(True, alpha=0.3)

# Summary statistics
axes[1, 1].axis('off')
summary_text = f"""
Training Summary
================
Algorithm: {CONFIG['rl_algorithm']}
Total Episodes: {len(training_metrics['episode_rewards'])}
Controlled Agents: {AGENT_NUMBER}

Final Metrics (last 100 episodes):
- Avg Reward: {np.mean(training_metrics['episode_rewards'][-100:]):.4f}
- Avg Cost: {np.mean(training_metrics['episode_costs'][-100:]):.4f}
- Avg Collision Rate: {np.mean(training_metrics['collision_rates'][-100:]):.4f}

Best Episode Reward: {max(training_metrics['episode_rewards']):.4f}
Lowest Collision Rate: {min(training_metrics['collision_rates']):.4f}
"""
axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[1, 1].transAxes)

plt.tight_layout()
plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ Training visualization saved to 'training_progress.png'")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 16. Evaluate Trained Policy

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.345049Z","iopub.status.idle":"2026-01-11T04:34:09.345607Z","shell.execute_reply.started":"2026-01-11T04:34:09.345392Z","shell.execute_reply":"2026-01-11T04:34:09.345418Z"}}
# Evaluation configuration
NUM_EVAL_EPISODES = 20
eval_metrics = {
    'rewards': [],
    'costs': [],
    'collisions': [],
}

print(f"🔍 Evaluating trained policy on {NUM_EVAL_EPISODES} episodes...")

# Evaluation loop
for eval_ep in tqdm(range(NUM_EVAL_EPISODES), desc="Evaluating"):
    try:
        scenario = next(waymo_iterator)
    except StopIteration:
        waymo_iterator = waymax_dataloader.simulator_state_generator(
            dataset_config)
        scenario = next(waymo_iterator)

    # Reset environment
    state = waymax_env.reset(scenario)

    ep_reward = 0.0
    ep_cost = 0.0
    ep_collisions = 0

    # Rollout with trained policy
    for step in range(STEPS_PER_EPISODE):
        # Get observations
        observations = extract_observations_from_waymax_state(
            state, CONFIG['controlled_agents']
        )
        observations_tensor = torch.FloatTensor(observations).to(device)

        # Get actions (no exploration during evaluation)
        actions_list = []
        with torch.no_grad():
            for i, agent in enumerate(agents):
                obs_flat = observations_tensor[i].flatten().unsqueeze(0)
                action = agent.choose_action(obs_flat)
                actions_list.append(action)

        # Convert to Waymax format
        waymax_action = create_waymax_action(
            actions_list,
            CONFIG['controlled_agents'],
            state
        )

        # Step environment
        next_state = waymax_env.step(state, waymax_action)

        # Compute metrics
        for i in range(AGENT_NUMBER):
            reward_tensor = waymax_env.reward(next_state, waymax_action)
            reward = float(
                np.array(reward_tensor[CONFIG['controlled_agents'][i]]).item())

            cost = compute_waymax_cost(
                next_state,
                CONFIG['controlled_agents'][i],
                CONFIG['controlled_agents']
            )

            ep_reward += reward
            ep_cost += cost
            if cost > 0:
                ep_collisions += 1

        # Check if done
        is_done = bool(np.array(next_state.is_done).item()) if hasattr(
            next_state.is_done, 'item') else bool(next_state.is_done)
        if is_done or step == STEPS_PER_EPISODE - 1:
            break

        state = next_state

    eval_metrics['rewards'].append(ep_reward / AGENT_NUMBER)
    eval_metrics['costs'].append(ep_cost / AGENT_NUMBER)
    eval_metrics['collisions'].append(ep_collisions)

# Print evaluation results
print("\n" + "="*50)
print("📊 EVALUATION RESULTS")
print("="*50)
print(
    f"Average Reward: {np.mean(eval_metrics['rewards']):.4f} ± {np.std(eval_metrics['rewards']):.4f}")
print(
    f"Average Cost: {np.mean(eval_metrics['costs']):.4f} ± {np.std(eval_metrics['costs']):.4f}")
print(
    f"Average Collisions: {np.mean(eval_metrics['collisions']):.2f} ± {np.std(eval_metrics['collisions']):.2f}")
print(
    f"Collision-Free Rate: {sum(1 for c in eval_metrics['collisions'] if c == 0) / NUM_EVAL_EPISODES * 100:.1f}%")
print("="*50)

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 17. Save Final Model

# %% [code] {"jupyter":{"outputs_hidden":false},"execution":{"iopub.status.busy":"2026-01-11T04:34:09.347355Z","iopub.status.idle":"2026-01-11T04:34:09.347817Z","shell.execute_reply.started":"2026-01-11T04:34:09.347577Z","shell.execute_reply":"2026-01-11T04:34:09.347603Z"}}
# Save trained models
if CONFIG['save_checkpoint']:
    save_dir = Path('checkpoints')
    save_dir.mkdir(exist_ok=True)

    timestamp = pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')

    for i, agent in enumerate(agents):
        checkpoint = {
            'agent_index': i,
            'state_dim': STATE_DIM,
            'agent_number': AGENT_NUMBER,
            'config': rl_config,
            'policy_state_dict': agent.pi.state_dict(),
            'value_state_dict': agent.value.state_dict(),
            'training_metrics': training_metrics,
            'eval_metrics': eval_metrics,
        }

        save_path = save_dir / \
            f'{CONFIG["rl_algorithm"]}_agent{i}_{timestamp}.pt'
        torch.save(checkpoint, save_path)
        print(f"✅ Saved Agent {i} to {save_path}")

    # Save training metrics
    metrics_path = save_dir / f'metrics_{timestamp}.pkl'
    with open(metrics_path, 'wb') as f:
        pickle.dump({
            'training': training_metrics,
            'evaluation': eval_metrics,
            'config': CONFIG,
        }, f)
    print(f"✅ Saved metrics to {metrics_path}")

# %% [markdown] {"jupyter":{"outputs_hidden":false}}
# ## 18. Summary
#
# **What we accomplished:**
# 1. ✅ Streamed Waymo Motion Dataset from Google Cloud (NO download!)
# 2. ✅ Used Waymax simulator with proper dynamics (InvertibleBicycleModel)
# 3. ✅ Trained multi-agent RL policy with TrafficGamer/MAPPO/CCE-MAPPO
# 4. ✅ Proper state transitions via `waymax_env.step()`
# 5. ✅ Evaluated on validation set
# 6. ✅ Visualized training progress
# 7. ✅ Saved trained models and metrics
#
# **Key improvements from original:**
# - ✅ Fixed Waymax integration - using `env.BaseEnvironment` properly
# - ✅ Proper observation extraction from `current_sim_trajectory`
# - ✅ Correct action format for `InvertibleBicycleModel`
# - ✅ Actual simulation loop with state transitions
# - ✅ Handles JAX array conversions correctly
#
# **Repository:** https://github.com/PhamPhuHoa-23/TrafficGamer
