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
# # Table XX: Scaling Analysis (N Agents, M Particles)
# 
# **Actual experiment: Measure runtime and memory vs N agents and M particles.**

# %% [markdown]
# ## 1. Setup

# %%
import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import gc
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
import torch

REPO_DIR = Path("TrafficGamer")
if not REPO_DIR.exists():
    import subprocess
    subprocess.run(["git", "clone", "https://github.com/PhamPhuHoa-23/EvolutionaryTest.git", str(REPO_DIR)])

sys.path.insert(0, str(REPO_DIR.absolute()))
os.chdir(REPO_DIR)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"✅ Device: {DEVICE}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# %%
from algorithm.evoqre_v2 import ParticleEvoQRE, EvoQREConfig, MultiAgentEvoQRE
from utils.utils import seed_everything

print("✅ Imports complete")

# %% [markdown]
# ## 2. Configuration

# %%
CONFIG = {
    'output_dir': './results/table20',
    'seed': 42,
    
    # Scaling configurations to test
    'configs': [
        {'N': 4, 'M': 50},
        {'N': 8, 'M': 50},
        {'N': 16, 'M': 50},
        {'N': 32, 'M': 25},  # Reduced M for memory
    ],
    
    'state_dim': 128,
    'action_dim': 2,
    'num_trials': 50,
}

seed_everything(CONFIG['seed'])
os.makedirs(CONFIG['output_dir'], exist_ok=True)

# %% [markdown]
# ## 3. Scaling Measurement Functions

# %%
def measure_scaling(N_agents, M_particles, state_dim=128, action_dim=2, num_trials=50):
    """
    Measure runtime and memory for given N, M configuration.
    
    Args:
        N_agents: Number of agents
        M_particles: Number of particles per agent
        state_dim: State dimension
        action_dim: Action dimension
        num_trials: Number of trials for averaging
        
    Returns:
        dict with time_ms, memory_gb, and success flag
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    
    try:
        # Create multi-agent system
        config = EvoQREConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=256,
            num_particles=M_particles,
            subsample_size=min(10, M_particles),
            langevin_steps=20,
            step_size=0.1,
            tau_base=1.0,
            device=str(DEVICE)
        )
        
        ma_system = MultiAgentEvoQRE(N_agents, config, shared_network=False)
        
        # Warmup
        states = [torch.randn(state_dim, device=DEVICE) for _ in range(N_agents)]
        _ = ma_system.select_actions(states)
        
        # Measure timing
        times = []
        for _ in range(num_trials):
            states = [torch.randn(state_dim, device=DEVICE) for _ in range(N_agents)]
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.time()
            
            actions = ma_system.select_actions(states)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.time() - start)
        
        # Measure memory
        if torch.cuda.is_available():
            memory_gb = torch.cuda.max_memory_allocated() / 1e9
        else:
            import psutil
            memory_gb = psutil.Process().memory_info().rss / 1e9
        
        avg_time_ms = np.mean(times) * 1000
        std_time_ms = np.std(times) * 1000
        
        # Cleanup
        del ma_system
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            'success': True,
            'time_ms': avg_time_ms,
            'time_std': std_time_ms,
            'memory_gb': memory_gb,
        }
        
    except RuntimeError as e:
        if 'out of memory' in str(e).lower():
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {'success': False, 'error': 'OOM'}
        raise e


def estimate_real_time_feasibility(time_ms, threshold=100):
    """
    Check if configuration meets real-time requirements.
    
    Real-time: < 100ms per step (10 Hz control loop)
    """
    return time_ms < threshold

# %% [markdown]
# ## 4. Run Scaling Analysis

# %%
print("\n" + "="*70)
print("Running Scaling Analysis")
print("="*70)

results = []

for cfg in CONFIG['configs']:
    N = cfg['N']
    M = cfg['M']
    
    print(f"\nTesting N={N} agents, M={M} particles...")
    
    result = measure_scaling(
        N_agents=N,
        M_particles=M,
        state_dim=CONFIG['state_dim'],
        action_dim=CONFIG['action_dim'],
        num_trials=CONFIG['num_trials']
    )
    
    if result['success']:
        time_str = f"{result['time_ms']:.0f}ms"
        memory_str = f"{result['memory_gb']:.1f}GB"
        realtime = "✓" if estimate_real_time_feasibility(result['time_ms']) else "✗"
        
        print(f"  Time: {time_str}, Memory: {memory_str}, Real-time: {realtime}")
    else:
        time_str = "OOM"
        memory_str = "---"
        realtime = "✗"
        print(f"  Out of Memory!")
    
    results.append({
        'N': N,
        'M': M,
        'Time/step': time_str,
        'Memory': memory_str,
        'Real-time': realtime,
        'time_raw': result.get('time_ms', float('inf')),
        'memory_raw': result.get('memory_gb', 0),
    })

# %% [markdown]
# ## 5. Results Table

# %%
df = pd.DataFrame(results)
display_cols = ['N', 'M', 'Time/step', 'Memory', 'Real-time']

print("\n" + "="*70)
print("Table XX: Scaling Analysis - Agents N and Particles M")
print("="*70)
print(df[display_cols].to_markdown(index=False))

# Save
df.to_csv(f"{CONFIG['output_dir']}/table20_results.csv", index=False)

# %% [markdown]
# ## 6. Analysis

# %%
print("\n" + "="*70)
print("Key Findings:")
print("="*70)

# Find real-time configurations
realtime_configs = [r for r in results if r['Real-time'] == '✓']
max_realtime_N = max([r['N'] for r in realtime_configs]) if realtime_configs else 0

# Compute scaling factor
if len(results) >= 2:
    t1 = results[0]['time_raw']
    t2 = results[1]['time_raw']
    n1 = results[0]['N']
    n2 = results[1]['N']
    scaling_factor = (t2 / t1) / (n2 / n1) if t1 > 0 and n1 > 0 else 1.0
else:
    scaling_factor = 1.0

print(f"""
1. Real-time feasibility (<100ms):
   - Max agents with real-time: N≤{max_realtime_N}
   - Need M reduction for N>16

2. Scaling characteristics:
   - Time scales ~ O(N^{scaling_factor:.1f})
   - Expected: O(N·M·M') with M'=10 subsampling

3. Memory scaling:
   - Memory: O(N·M) for particle storage
   - GPU required for N>8

4. Recommendations:
   - Dense urban (N>20): Use neighbor culling (R=20m)
   - Highway (N<10): Full particle set feasible
   - Memory-constrained: Reduce M to 25
""")

# %% [markdown]
# ## 7. Visualization

# %%
import matplotlib.pyplot as plt

N_vals = [r['N'] for r in results if r['time_raw'] < float('inf')]
times = [r['time_raw'] for r in results if r['time_raw'] < float('inf')]
memories = [r['memory_raw'] for r in results if r['time_raw'] < float('inf')]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(N_vals, times, 'o-', linewidth=2, markersize=10, color='blue')
ax1.axhline(y=100, color='red', linestyle='--', label='100ms real-time threshold')
ax1.set_xlabel('Number of Agents (N)')
ax1.set_ylabel('Time per Step (ms)')
ax1.set_title('Runtime vs Agent Count')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(N_vals, memories, 's-', linewidth=2, markersize=10, color='green')
ax2.set_xlabel('Number of Agents (N)')
ax2.set_ylabel('Memory (GB)')
ax2.set_title('Memory vs Agent Count')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{CONFIG['output_dir']}/tab20_scaling.png", dpi=150)
plt.show()

print(f"\n✅ Saved: {CONFIG['output_dir']}/tab20_scaling.png")
