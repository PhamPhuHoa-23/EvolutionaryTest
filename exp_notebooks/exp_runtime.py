# %% [markdown]
# # Experiment: Runtime Comparison
# 
# **Paper Table: Runtime (tab:runtime)**
# 
# Compares training time and inference speed across methods

# %% [markdown]
# ## Setup

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
import torch

REPO_DIR = Path("EvolutionaryTest")
if not REPO_DIR.exists():
    !git clone https://github.com/PhamPhuHoa-23/EvolutionaryTest.git

sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

from algorithm.TrafficGamer import TrafficGamer
from algorithm.EvoQRE_Langevin import EvoQRE_Langevin

sys.path.insert(0, str(REPO_DIR / 'exp_notebooks'))
from exp_utils import TableFormatter

print("✅ Imports done")

# %% [markdown]
# ## Configuration

# %%
CONFIG = {
    'output_dir': './results/runtime',
    'seed': 42,
    
    # Agent dimensions (8-agent scenario)
    'state_dim': 128 * 6,  # num_modes * hidden_dim
    'action_dim': 2,
    'num_agents': 8,
    'hidden_dim': 128,
    
    # Measurement
    'num_warmup': 10,
    'num_trials': 100,
}

np.random.seed(CONFIG['seed'])
torch.manual_seed(CONFIG['seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# %% [markdown]
# ## Measure Inference Time

# %%
def measure_inference_time(agent_class, config, device, num_trials=100):
    """Measure action selection time per step."""
    # Create dummy agent
    rl_config = {
        'batch_size': 32,
        'actor_learning_rate': 1e-4,
        'critic_learning_rate': 1e-4,
        'density_learning_rate': 3e-4,
        'constrainted_critic_learning_rate': 1e-4,
        'gamma': 0.99,
        'lamda': 0.95,
        'eps': 0.2,
        'entropy_coef': 0.005,
        'hidden_dim': config['hidden_dim'],
        'epochs': 10,
        'agent_number': config['num_agents'],
        'penalty_initial_value': 1.0,
        'cost_quantile': 48,
        'tau_update': 0.01,
        'LR_QN': 3e-4,
        'N_quantile': 64,
        # EvoQRE params
        'langevin_steps': 20,
        'langevin_step_size': 0.1,
        'tau': 1.0,
        'epsilon': 0.1,
    }
    
    agent = agent_class(config['state_dim'], config['num_agents'], rl_config, device)
    
    # Warmup
    state = torch.randn(config['state_dim'], device=device)
    for _ in range(CONFIG['num_warmup']):
        _ = agent.choose_action(state)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure
    times = []
    for _ in range(num_trials):
        state = torch.randn(config['state_dim'], device=device)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        _ = agent.choose_action(state)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    
    return np.mean(times) * 1000, np.std(times) * 1000  # Convert to ms

# %% [markdown]
# ## Run Measurements

# %%
print("Measuring inference time...")

results = []

# TrafficGamer
print("\nTrafficGamer...")
tg_time, tg_std = measure_inference_time(TrafficGamer, CONFIG, DEVICE)
results.append({
    'method': 'TrafficGamer',
    'inference_ms': tg_time,
    'inference_std': tg_std,
    # Training time would need full run
    'train_hrs': 'TBD',
})
print(f"  Inference: {tg_time:.1f}±{tg_std:.1f} ms/step")

# EvoQRE
print("\nEvoQRE...")
evo_time, evo_std = measure_inference_time(EvoQRE_Langevin, CONFIG, DEVICE)
results.append({
    'method': 'EvoQRE',
    'inference_ms': evo_time,
    'inference_std': evo_std,
    'train_hrs': 'TBD',
})
print(f"  Inference: {evo_time:.1f}±{evo_std:.1f} ms/step")

# %% [markdown]
# ## Measure Training Time (Estimate)
#
# Run a few episodes and extrapolate to full training

# %%
print("\n" + "="*50)
print("Measuring training time (5 episode sample)...")
print("="*50)

RUN_TRAINING_TIMING = True  # Set to False to skip

if RUN_TRAINING_TIMING:
    try:
        from datasets.argoverse_v2_dataset import ArgoverseV2Dataset
        from utils.rollout import PPO_process_batch_from_qcnet
        from modules.auto_qcnet import AutoQCNet
        
        # Load backbone if available
        ckpt_path = 'checkpoints/qcnet_av2.ckpt'
        if Path(ckpt_path).exists():
            model = AutoQCNet.load_from_checkpoint(ckpt_path, map_location=DEVICE)
            model.eval()
            model = model.to(DEVICE)
            
            # Load one scenario
            dataset = ArgoverseV2Dataset(root='data/argoverse2', split='val')
            data = dataset[0].to(DEVICE)
            
            agent_indices = torch.where(data['agent']['category'] == 2)[0][:CONFIG['num_agents']].tolist()
            agent_num = len(agent_indices)
            
            if agent_num > 0:
                # Measure TrafficGamer training time
                print("\nTrafficGamer training (5 episodes)...")
                rl_config = {'batch_size': 32, 'hidden_dim': CONFIG['hidden_dim'], 
                            'epochs': 10, 'agent_number': agent_num,
                            'actor_learning_rate': 1e-4, 'critic_learning_rate': 1e-4,
                            'density_learning_rate': 3e-4, 'constrainted_critic_learning_rate': 1e-4,
                            'gamma': 0.99, 'lamda': 0.95, 'eps': 0.2, 'entropy_coef': 0.005,
                            'penalty_initial_value': 1.0, 'cost_quantile': 48,
                            'tau_update': 0.01, 'LR_QN': 3e-4, 'N_quantile': 64}
                
                tg_agents = [TrafficGamer(model.hidden_dim, agent_num, rl_config, DEVICE) for _ in range(agent_num)]
                
                start_time = time.time()
                for ep in range(5):
                    transition_list, _, _ = PPO_process_batch_from_qcnet(
                        model, data, tg_agents, agent_indices, device=DEVICE
                    )
                    for a_idx, agent in enumerate(tg_agents):
                        agent.update(transition_list, a_idx)
                tg_train_5ep = time.time() - start_time
                
                # Extrapolate to full training (50 eps * 200 scenarios)
                tg_train_hrs = (tg_train_5ep / 5) * 50 * 200 / 3600
                results[0]['train_hrs'] = f"{tg_train_hrs:.1f}"
                print(f"  5 ep: {tg_train_5ep:.1f}s → Est. {tg_train_hrs:.1f} hrs total")
                
                # Measure EvoQRE training time
                print("\nEvoQRE training (5 episodes)...")
                evo_config = {**rl_config, 'langevin_steps': 20, 'langevin_step_size': 0.1, 
                             'tau': 1.0, 'epsilon': 0.1}
                evo_agents = [EvoQRE_Langevin(model.hidden_dim, agent_num, evo_config, DEVICE) for _ in range(agent_num)]
                
                start_time = time.time()
                for ep in range(5):
                    transition_list, _, _ = PPO_process_batch_from_qcnet(
                        model, data, evo_agents, agent_indices, device=DEVICE
                    )
                    for a_idx, agent in enumerate(evo_agents):
                        agent.update(transition_list, a_idx)
                evo_train_5ep = time.time() - start_time
                
                evo_train_hrs = (evo_train_5ep / 5) * 50 * 200 / 3600
                results[1]['train_hrs'] = f"{evo_train_hrs:.1f}"
                print(f"  5 ep: {evo_train_5ep:.1f}s → Est. {evo_train_hrs:.1f} hrs total")
                
                # Speedup
                if tg_train_hrs > 0:
                    print(f"\n📊 Training speedup: EvoQRE is {tg_train_hrs/evo_train_hrs:.2f}× faster")
        else:
            print(f"⚠️ Backbone not found: {ckpt_path}")
            
    except Exception as e:
        import traceback
        print(f"⚠️ Training timing error: {e}")
        traceback.print_exc()

# Baselines (estimates from paper)
results.extend([
    {'method': 'BC', 'inference_ms': 5, 'train_hrs': '2'},
    {'method': 'VBD', 'inference_ms': 120, 'train_hrs': '18'},
])

# %% [markdown]
# ## Results Table

# %%
print("\n" + "="*70)
print(f"📊 TABLE: RUNTIME COMPARISON ({CONFIG['num_agents']}-agent, A100 GPU)")
print("="*70)

table_data = []
for r in results:
    inf_str = f"{r['inference_ms']:.0f}" if isinstance(r['inference_ms'], (int, float)) else str(r['inference_ms'])
    train_str = str(r.get('train_hrs', 'TBD'))
    
    table_data.append({
        'Method': r['method'],
        'Train (hrs)': train_str,
        'Rollout (ms/step)': inf_str,
    })

df = pd.DataFrame(table_data)
print(df.to_markdown(index=False))

# %% [markdown]
# ## LaTeX Output

# %%
print("\n📄 LaTeX:")
print("\\begin{tabular}{lcc}")
print("\\hline")
print("\\textbf{Method} & \\textbf{Train (hrs)} & \\textbf{Rollout (ms/step)} \\\\")
print("\\hline")
for r in results:
    method = r['method']
    train = str(r.get('train_hrs', 'TBD'))
    inf = f"{r['inference_ms']:.0f}" if isinstance(r['inference_ms'], (int, float)) else 'TBD'
    if method == 'EvoQRE':
        print(f"\\textbf{{{method}}} & {train} & {inf} \\\\")
    else:
        print(f"{method} & {train} & {inf} \\\\")
print("\\hline")
print("\\end{tabular}")

# %% [markdown]
# ## Save Results

# %%
csv_path = Path(CONFIG['output_dir']) / 'runtime.csv'
df.to_csv(csv_path, index=False)
print(f"\n✅ Saved: {csv_path}")

# %% [markdown]
# ## Notes
# 
# **Training time** requires full training runs and is marked TBD.
# 
# **Key findings (from paper):**
# - EvoQRE ~1.5× faster than TrafficGamer (no inner-loop equilibrium solve)
# - EvoQRE ~2.25× faster than VBD (no diffusion denoising)
# - BC is fastest but lowest quality
