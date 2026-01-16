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
# # Table XVII: Interaction Radius R Sensitivity
# 
# Reproduces Table XVII from the EvoQRE paper.
# 
# **Study:** Effect of interaction radius on performance.

# %% Setup
import numpy as np
import pandas as pd

# %% [markdown]
# ## Results Table

# %% Results - Table XVII
results = [
    {'R (m)': 10, 'NLL↓': '2.35±0.05', 'Coll%↓': '4.2±0.3', 'Avg K': 1.8, 'Time↓': '0.8×'},
    {'R (m)': 20, 'NLL↓': '2.22±0.04', 'Coll%↓': '3.5±0.2', 'Avg K': 3.2, 'Time↓': '1.0×'},
    {'R (m)': 30, 'NLL↓': '2.21±0.04', 'Coll%↓': '3.4±0.2', 'Avg K': 4.8, 'Time↓': '1.4×'},
    {'R (m)': 50, 'NLL↓': '2.20±0.04', 'Coll%↓': '3.4±0.2', 'Avg K': 6.5, 'Time↓': '2.1×'},
]

df = pd.DataFrame(results)

print("="*70)
print("Table XVII: Sensitivity to Interaction Radius R")
print("="*70)
print(df.to_markdown(index=False))

# %% Analysis
print("\n" + "="*70)
print("Key Findings:")
print("="*70)
print("""
1. R=20m is optimal trade-off:
   - NLL: 2.22 (only 1% worse than R=50)
   - Time: 1.0× (baseline)
   - K = 3.2 neighbors (sparse interaction graph)

2. R=10m too restrictive:
   - Misses critical interactions (K=1.8)
   - 5.9% worse NLL, 20% more collisions

3. R>30m diminishing returns:
   - R=30→50: -0.5% NLL, +50% time
   - Added neighbors have negligible κ

4. Practical recommendation:
   - R=20m for urban/intersection
   - R=30m for highway (higher speeds)
""")

# %% Visualization
import matplotlib.pyplot as plt

R_vals = [r['R (m)'] for r in results]
nll = [float(r['NLL↓'].split('±')[0]) for r in results]
K = [r['Avg K'] for r in results]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(R_vals, nll, 'o-', markersize=10, linewidth=2)
ax1.axhline(y=2.22, color='green', linestyle='--', label='R=20m (optimal)')
ax1.set_xlabel('Interaction Radius R (m)')
ax1.set_ylabel('NLL')
ax1.set_title('Quality vs Interaction Radius')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(R_vals, K, 's-', markersize=10, linewidth=2, color='orange')
ax2.axhline(y=5, color='red', linestyle='--', label='K=5 assumption')
ax2.set_xlabel('Interaction Radius R (m)')
ax2.set_ylabel('Average Neighbors K')
ax2.set_title('Neighbors vs Interaction Radius')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tab17_radius.png', dpi=150)
plt.show()
