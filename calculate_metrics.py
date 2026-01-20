
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Optional

@dataclass
class ScenarioResult:
    scenario_id: str
    method: str
    num_agents: int
    reward: float
    cost: float
    nll: float
    collision_rate: float
    off_road_rate: float
    diversity: float
    speed_mean: float
    speed_std: float
    accel_mean: float
    accel_std: float
    following_dist_mean: float
    following_dist_std: float
    lane_changes_per_km: float
    alpha: float
    kappa: float
    stability_satisfied: bool
    train_time_s: float
    inference_time_ms: float
    trajectory_path: Optional[str] = None
    checkpoint_path: Optional[str] = None

# Results from Batch 2 (Dense, Roundabout, Y-junction)
batch2_tg = [
  ScenarioResult(scenario_id='cb0133ff-f7ad-43b7-b260-7068ace15307', method='TrafficGamer', num_agents=5, reward=np.float64(-6701.370490167322), cost=np.float64(0.03), nll=np.float64(5.88680253102106), collision_rate=0.0, off_road_rate=0.8, diversity=np.float32(33.035927), speed_mean=np.float32(0.41682678), speed_std=np.float32(0.48165432), accel_mean=np.float32(1.6527996), accel_std=np.float32(2.2303636), following_dist_mean=np.float32(20.220942), following_dist_std=np.float32(2.4687202), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=1169.5630466938019, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='3856ed37-4a05-4131-9b12-c4f4716fec92', method='TrafficGamer', num_agents=5, reward=np.float64(-1058.242191848722), cost=np.float64(0.24000000000000005), nll=np.float64(2.30548430161152), collision_rate=0.0, off_road_rate=0.4, diversity=np.float32(31.731165), speed_mean=np.float32(3.4781964), speed_std=np.float32(3.6034307), accel_mean=np.float32(1.5749748), accel_std=np.float32(2.2189803), following_dist_mean=np.float32(12.558942), following_dist_std=np.float32(6.6811266), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=1229.132869720459, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='cdf70cc8-d13d-470b-bb39-4f1812acc146', method='TrafficGamer', num_agents=5, reward=np.float64(-39.29195743893917), cost=np.float64(0.57375), nll=np.float64(1.9016125186585406), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(25.631878), speed_mean=np.float32(3.1623762), speed_std=np.float32(2.3687968), accel_mean=np.float32(1.545586), accel_std=np.float32(1.9586757), following_dist_mean=np.float32(11.740821), following_dist_std=np.float32(2.829279), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=1473.660481929779, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None)
]
batch2_evo = [
  ScenarioResult(scenario_id='cb0133ff-f7ad-43b7-b260-7068ace15307', method='EvoQRE', num_agents=5, reward=np.float64(-6702.506781966574), cost=np.float64(0.03125), nll=np.float64(2.62164209746902), collision_rate=0.0, off_road_rate=0.8, diversity=np.float32(32.937958), speed_mean=np.float32(0.67710745), speed_std=np.float32(0.68657184), accel_mean=np.float32(1.5188848), accel_std=np.float32(1.9663352), following_dist_mean=np.float32(20.2955), following_dist_std=np.float32(2.00042), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=3483.963696241379, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='3856ed37-4a05-4131-9b12-c4f4716fec92', method='EvoQRE', num_agents=5, reward=np.float64(-1065.4854250407593), cost=np.float64(1.44875), nll=np.float64(2.3463332852285657), collision_rate=0.0, off_road_rate=0.4, diversity=np.float32(31.970947), speed_mean=np.float32(3.594966), speed_std=np.float32(3.492128), accel_mean=np.float32(1.6852787), accel_std=np.float32(2.2269506), following_dist_mean=np.float32(12.812262), following_dist_std=np.float32(6.7742834), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=3699.562202692032, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='cdf70cc8-d13d-470b-bb39-4f1812acc146', method='EvoQRE', num_agents=5, reward=np.float64(-36.98222723511335), cost=np.float64(1.44375), nll=np.float64(1.7495704904759366), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(25.478159), speed_mean=np.float32(2.9472039), speed_std=np.float32(2.1846752), accel_mean=np.float32(1.5213318), accel_std=np.float32(1.8700137), following_dist_mean=np.float32(11.556523), following_dist_std=np.float32(3.059853), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=3899.21847987175, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None)
]

# Results from Batch 1 (Merge, Dual-lane, T-junction)
batch1_tg = [
  ScenarioResult(scenario_id='236df665-eec6-4c25-8822-950a6150eade', method='TrafficGamer', num_agents=5, reward=np.float64(-1997.9356556335429), cost=np.float64(0.70625), nll=np.float64(2.7312397411168394), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(47.407917), speed_mean=np.float32(4.182868), speed_std=np.float32(4.855919), accel_mean=np.float32(1.6438366), accel_std=np.float32(2.2153955), following_dist_mean=np.float32(22.61271), following_dist_std=np.float32(18.098059), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=1695.9712636470795, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='00a50e9f-63a1-4678-a4fe-c6109721ecba', method='TrafficGamer', num_agents=3, reward=np.float64(-4077.1403460728625), cost=np.float64(0.14166666666666666), nll=np.float64(2.486347201467767), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(23.13682), speed_mean=np.float32(6.1946464), speed_std=np.float32(4.0090194), accel_mean=np.float32(2.1422317), accel_std=np.float32(2.6116142), following_dist_mean=np.float32(15.298481), following_dist_std=np.float32(7.3601284), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=959.3774945735931, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84', method='TrafficGamer', num_agents=5, reward=np.float64(-30.059581452001584), cost=np.float64(0.2825), nll=np.float64(2.0319406102373407), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(36.926582), speed_mean=np.float32(2.0078614), speed_std=np.float32(2.5121543), accel_mean=np.float32(1.5978595), accel_std=np.float32(2.181289), following_dist_mean=np.float32(15.164797), following_dist_std=np.float32(8.255497), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=1288.423309803009, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None)
]
batch1_evo = [
  ScenarioResult(scenario_id='236df665-eec6-4c25-8822-950a6150eade', method='EvoQRE', num_agents=5, reward=np.float64(-1995.946302494956), cost=np.float64(1.80125), nll=np.float64(2.733511929840885), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(47.075554), speed_mean=np.float32(4.4117465), speed_std=np.float32(5.0519543), accel_mean=np.float32(1.612783), accel_std=np.float32(2.0673046), following_dist_mean=np.float32(22.212482), following_dist_std=np.float32(18.44227), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=3908.318207502365, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='00a50e9f-63a1-4678-a4fe-c6109721ecba', method='EvoQRE', num_agents=3, reward=np.float64(-4071.9279659287376), cost=np.float64(0.3208333333333333), nll=np.float64(2.5001143812175957), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(23.374136), speed_mean=np.float32(7.640742), speed_std=np.float32(3.885625), accel_mean=np.float32(2.0476923), accel_std=np.float32(2.2443376), following_dist_mean=np.float32(15.781155), following_dist_std=np.float32(6.940365), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=2542.947018623352, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None),
  ScenarioResult(scenario_id='d1f6b01e-3b4a-4790-88ed-6d85fb1c0b84', method='EvoQRE', num_agents=5, reward=np.float64(-33.2572925561294), cost=np.float64(1.1887500000000002), nll=np.float64(2.0913481765761897), collision_rate=0.0, off_road_rate=0.0, diversity=np.float32(36.513653), speed_mean=np.float32(2.421391), speed_std=np.float32(2.4919195), accel_mean=np.float32(1.3583604), accel_std=np.float32(1.6893296), following_dist_mean=np.float32(15.07776), following_dist_std=np.float32(7.9439), lane_changes_per_km=0.0, alpha=0.0, kappa=0.0, stability_satisfied=False, train_time_s=3752.4787118434906, inference_time_ms=0.0, trajectory_path=None, checkpoint_path=None)
]

# Combine
all_tg = batch1_tg + batch2_tg
all_evo = batch1_evo + batch2_evo

def compute_stats(results):
    nll = [r.nll for r in results]
    coll = [r.collision_rate * 100 for r in results] # Convert to %
    off = [r.off_road_rate * 100 for r in results] # Convert to %
    div = [r.diversity for r in results]
    speed = [r.speed_mean for r in results]
    accel = [r.accel_mean for r in results]
    
    return {
        'nll_mean': np.mean(nll), 'nll_std': np.std(nll),
        'coll_mean': np.mean(coll), 'coll_std': np.std(coll),
        'off_mean': np.mean(off), 'off_std': np.std(off),
        'div_mean': np.mean(div), 'div_std': np.std(div),
        'speed_mean': np.mean(speed), 'speed_std': np.std(speed),
        'accel_mean': np.mean(accel), 'accel_std': np.std(accel),
    }

stats_tg = compute_stats(all_tg)
stats_evo = compute_stats(all_evo)

print("=== FINAL METRICS (6 Scenarios, 30 Episodes) ===")
print("TrafficGamer:")
print(f"  NLL: {stats_tg['nll_mean']:.2f} ± {stats_tg['nll_std']:.2f}")
print(f"  Coll: {stats_tg['coll_mean']:.2f}% ± {stats_tg['coll_std']:.2f}")
print(f"  Off:  {stats_tg['off_mean']:.2f}% ± {stats_tg['off_std']:.2f}")
print(f"  Div:  {stats_tg['div_mean']:.2f} ± {stats_tg['div_std']:.2f}")
print("EvoQRE:")
print(f"  NLL: {stats_evo['nll_mean']:.2f} ± {stats_evo['nll_std']:.2f}")
print(f"  Coll: {stats_evo['coll_mean']:.2f}% ± {stats_evo['coll_std']:.2f}")
print(f"  Off:  {stats_evo['off_mean']:.2f}% ± {stats_evo['off_std']:.2f}")
print(f"  Div:  {stats_evo['div_mean']:.2f} ± {stats_evo['div_std']:.2f}")
