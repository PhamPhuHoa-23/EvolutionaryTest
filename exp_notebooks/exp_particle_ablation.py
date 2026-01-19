# %% [markdown]
# # Experiment: Particle Count M Ablation
# 
# **Paper Table: Ablation Particles (tab:ablation_particles)**
# 
# Effect of particle count M on NLL, Collision%, and Time

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

from algorithm.EvoQRE_Langevin import EvoQRE_Langevin, ConcaveQNetwork, LangevinSampler
from utils.utils import seed_everything

sys.path.insert(0, str(REPO_DIR / 'exp_notebooks'))
from exp_utils import ResultsSaver, TableFormatter

seed_everything(42)
print("✅ Imports done")

# %% [markdown] 
# ## Configuration

# %%
CONFIG = {
    'output_dir': './results/particle_ablation',
    
    # M values to test (from paper Table IX)
    'M_values': [10, 25, 50, 100],
    
    # Agent settings
    'state_dim': 128,
    'action_dim': 2,
    'hidden_dim': 256,
    'langevin_steps': 20,
    'step_size': 0.1,
    'tau': 1.0,
    'epsilon': 0.1,
    
    # Measurement
    'num_timing_trials': 100,
    'warmup_trials': 10,
}

os.makedirs(CONFIG['output_dir'], exist_ok=True)

# %% [markdown]
# ## Measure Timing per M

# %%
print("Measuring inference time for each M value...")

results = []

for M in CONFIG['M_values']:
    print(f"\n{'='*50}")
    print(f"Testing M = {M} particles")
    print(f"{'='*50}")
    
    # Create sampler with M particles
    sampler = LangevinSampler(
        num_steps=CONFIG['langevin_steps'],
        step_size=CONFIG['step_size'],
        temperature=CONFIG['tau'],
        action_bound=1.0,
    )
    
    # Create Q-network
    q_network = ConcaveQNetwork(
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        hidden_dim=CONFIG['hidden_dim'],
        epsilon=CONFIG['epsilon'],
        use_spectral_norm=True
    ).to(DEVICE)
    
    # Warmup
    print(f"  Warmup ({CONFIG['warmup_trials']} trials)...")
    for _ in range(CONFIG['warmup_trials']):
        state = torch.randn(CONFIG['state_dim'], device=DEVICE)
        _ = sampler.sample(q_network, state)
    
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    
    # Measure timing
    print(f"  Timing ({CONFIG['num_timing_trials']} trials)...")
    times = []
    for _ in range(CONFIG['num_timing_trials']):
        state = torch.randn(CONFIG['state_dim'], device=DEVICE)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        
        # Sample M particles (simulate by running M times or adjusting sampler)
        for _ in range(M):
            _ = sampler.sample(q_network, state)
        
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        times.append(elapsed)
    
    avg_time_s = np.mean(times)
    std_time_s = np.std(times)
    
    print(f"  Average time: {avg_time_s:.4f}s ± {std_time_s:.4f}s")
    
    results.append({
        'M': M,
        'time_s': avg_time_s,
        'time_std': std_time_s,
        # Note: NLL and Collision need full training run
        # Here we estimate based on expected scaling
    })

# %% [markdown]
# ## Add Estimated Quality Metrics
# 
# Note: Real NLL/Collision values require full training.
# These are placeholders based on expected scaling from paper.

# %%
# Placeholder scaling (fill with real values after full training)
# From paper pattern: more particles → better quality, diminishing returns

for r in results:
    M = r['M']
    # Placeholder: NLL improves with log(M)
    # r['nll'] = 2.45 - 0.1 * np.log2(M/10)  # Example
    # r['coll'] = 4.2 - 0.3 * np.log2(M/10)  # Example
    r['nll'] = 'TBD'  # To be filled after training
    r['coll'] = 'TBD'

# %% [markdown]
# ## Results Table

# %%
print("\n" + "="*70)
print("📊 TABLE: ABLATION - PARTICLE COUNT M")
print("="*70)

table_data = []
for r in results:
    table_data.append({
        'M': r['M'],
        'NLL↓': r['nll'],
        'Coll.%↓': r['coll'],
        'Time (s)↓': f"{r['time_s']:.2f}",
    })

df = pd.DataFrame(table_data)
print(df.to_markdown(index=False))

# %% [markdown]
# ## LaTeX Output

# %%
latex = TableFormatter.format_ablation_results(
    param_name='$M$',
    param_values=[str(r['M']) for r in results],
    nll_values=[str(r['nll']) for r in results],
    coll_values=[str(r['coll']) for r in results],
    time_values=[f"{r['time_s']:.1f}" for r in results],
)
print("\n📄 LaTeX:")
print(latex)

# %% [markdown]
# ## Save Results

# %%
csv_path = Path(CONFIG['output_dir']) / 'particle_ablation.csv'
df.to_csv(csv_path, index=False)

latex_path = Path(CONFIG['output_dir']) / 'table_particle_ablation.tex'
with open(latex_path, 'w') as f:
    f.write(latex)

print(f"\n✅ Saved CSV: {csv_path}")
print(f"✅ Saved LaTeX: {latex_path}")

# %% [markdown]
# ## Note
# 
# **Important:** NLL and Collision% values marked "TBD" require full training runs.
# Run `exp_main_results.py` with different M values to get actual quality metrics.
# Time measurements above are real.
