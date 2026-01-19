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
