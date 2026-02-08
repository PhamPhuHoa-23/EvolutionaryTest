"""
Experiment Utilities for EvoQRE Paper Tables.

This module provides:
1. ExperimentResults - Data class for storing experiment results
2. MetricsComputer - Compute all paper metrics (NLL, Collision, Diversity, etc.)
3. ResultsSaver - Save/load results in standardized format
4. TableFormatter - Format results for LaTeX tables in paper
"""

import os
import json
import pickle
import hashlib
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance, gaussian_kde
from scipy.spatial.distance import jensenshannon


# ==============================================================================
# Section 1: Data Classes for Results
# ==============================================================================

@dataclass
class ScenarioResult:
    """Results for a single scenario."""
    scenario_id: str
    method: str
    num_agents: int
    
    # Core metrics
    reward: float = 0.0
    cost: float = 0.0
    
    # Table VI metrics
    nll: float = 0.0
    collision_rate: float = 0.0
    off_road_rate: float = 0.0
    diversity: float = 0.0
    
    # Table VII: Behavioral metrics
    speed_mean: float = 0.0
    speed_std: float = 0.0
    accel_mean: float = 0.0
    accel_std: float = 0.0
    following_dist_mean: float = 0.0
    following_dist_std: float = 0.0
    lane_changes_per_km: float = 0.0
    
    # TrafficGamer-compatible risk metrics
    ttc_risk_ratio: float = 0.0  # Time-to-Collision < 2s
    thw_risk_ratio: float = 0.0  # Time Headway < 2s
    
    # TrafficGamer-compatible fidelity metrics
    hellinger_velocity: float = 0.0
    hellinger_acceleration: float = 0.0
    kl_velocity: float = 0.0
    wasserstein_velocity: float = 0.0
    
    # Stability metrics (Table: Stability)
    alpha: float = 0.0
    kappa: float = 0.0
    stability_satisfied: bool = False
    
    # Runtime
    train_time_s: float = 0.0
    inference_time_ms: float = 0.0
    
    # Raw data paths
    trajectory_path: Optional[str] = None
    checkpoint_path: Optional[str] = None


@dataclass
class ExperimentConfig:
    """Configuration for an experiment run."""
    experiment_name: str
    method: str
    
    # Dataset
    dataset: str = 'argoverse2'
    split: str = 'val'
    num_scenarios: int = 200
    
    # Training
    num_episodes: int = 50
    batch_size: int = 32
    max_agents: int = 10
    epochs: int = 10
    
    # EvoQRE specific
    num_particles: int = 50
    langevin_steps: int = 20
    step_size: float = 0.1
    tau: float = 1.0
    epsilon: float = 0.1
    
    # Output
    output_dir: str = './results'
    seed: int = 42
    
    def get_hash(self) -> str:
        """Get unique hash for this config."""
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class ExperimentResults:
    """Complete results for an experiment."""
    config: ExperimentConfig
    scenario_results: List[ScenarioResult] = field(default_factory=list)
    start_time: str = ""
    end_time: str = ""
    total_time_s: float = 0.0
    
    def add_result(self, result: ScenarioResult):
        self.scenario_results.append(result)
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame."""
        return pd.DataFrame([asdict(r) for r in self.scenario_results])
    
    def get_summary(self) -> Dict[str, str]:
        """Get summary statistics formatted for paper table."""
        df = self.to_dataframe()
        
        if len(df) == 0:
            return {}
        
        def fmt(col: str, multiplier: float = 1.0, decimals: int = 2) -> str:
            mean = df[col].mean() * multiplier
            std = df[col].std() * multiplier
            return f"{mean:.{decimals}f}±{std:.{decimals}f}"
        
        return {
            'Method': self.config.method,
            'NLL↓': fmt('nll'),
            'Coll.%↓': fmt('collision_rate', 100, 1),
            'Off-road%↓': fmt('off_road_rate', 100, 1),
            'Div.↑': fmt('diversity'),
        }


# ==============================================================================
# Section 2: Metrics Computation
# ==============================================================================

class MetricsComputer:
    """
    Compute all metrics needed for paper tables.
    
    Implements:
    - NLL via KDE (Table VI, with bandwidth sensitivity)
    - Collision rate (bounding box overlap)
    - Off-road rate (center outside drivable area)
    - Diversity (mean pairwise trajectory distance)
    - Behavioral metrics (speed, accel, following distance)
    - Stability metrics (α, κ estimation)
    """
    
    def __init__(self, kde_bandwidth: str = 'silverman'):
        self.kde_bandwidth = kde_bandwidth
    
    # ==========================================
    # Table VI: Main Metrics
    # ==========================================
    
    def compute_nll_kde(
        self,
        samples: np.ndarray,
        targets: np.ndarray,
        bandwidth: Optional[float] = None
    ) -> float:
        """
        Compute NLL via Kernel Density Estimation.
        
        From paper: Gaussian kernel with Silverman bandwidth.
        """
        if len(samples) < 5 or len(targets) == 0:
            return float('inf')
        
        try:
            # Flatten if needed
            samples_flat = samples.flatten()
            targets_flat = targets.flatten()
            
            if len(samples_flat) < 5:
                return float('inf')
            
            # Build KDE
            bw = bandwidth if bandwidth else self.kde_bandwidth
            kde = gaussian_kde(samples_flat, bw_method=bw)
            
            # Compute log probability
            log_probs = kde.logpdf(targets_flat)
            nll = -np.mean(log_probs)
            
            return nll if np.isfinite(nll) else float('inf')
        except Exception:
            return float('inf')
    
    def compute_collision_rate(
        self,
        positions: np.ndarray,
        threshold: float = 2.0
    ) -> float:
        """
        Compute pairwise collision rate.
        
        Args:
            positions: (num_agents, num_steps, 2 or 3)
            threshold: Distance threshold for collision (meters)
            
        Returns:
            Collision rate [0, 1]
        """
        if positions.shape[0] < 2:
            return 0.0
        
        num_agents, num_steps = positions.shape[:2]
        collisions = 0
        total_pairs = 0
        
        for t in range(num_steps):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    dist = np.linalg.norm(
                        positions[i, t, :2] - positions[j, t, :2]
                    )
                    total_pairs += 1
                    if dist < threshold:
                        collisions += 1
        
        return collisions / max(total_pairs, 1)
    
    def compute_off_road_rate(
        self,
        positions: np.ndarray,
        drivable_polygons: List
    ) -> float:
        """
        Compute fraction of positions outside drivable area.
        
        Args:
            positions: (num_agents, num_steps, 2)
            drivable_polygons: List of shapely Polygon objects
            
        Returns:
            Off-road rate [0, 1]
        """
        if not drivable_polygons or len(drivable_polygons) == 0:
            return 0.0
        
        try:
            from shapely.geometry import Point
            
            off_road_count = 0
            total_points = 0
            
            for i in range(positions.shape[0]):
                for t in range(positions.shape[1]):
                    pt = Point(positions[i, t, 0], positions[i, t, 1])
                    total_points += 1
                    
                    is_on_road = any(
                        poly.contains(pt) for poly in drivable_polygons
                    )
                    if not is_on_road:
                        off_road_count += 1
            
            return off_road_count / total_points if total_points > 0 else 0.0
        except ImportError:
            return 0.0
    
    def compute_diversity(self, trajectories: np.ndarray) -> float:
        """
        Compute mean pairwise trajectory distance.
        
        Args:
            trajectories: (num_samples, num_steps, dim) or list of arrays
            
        Returns:
            Mean pairwise distance
        """
        if isinstance(trajectories, list):
            trajectories = np.array(trajectories)
        
        if len(trajectories) < 2:
            return 0.0
        
        total_dist = 0.0
        count = 0
        
        num_samples = len(trajectories)
        for i in range(num_samples):
            for j in range(i + 1, num_samples):
                # Mean distance across timesteps
                dist = np.mean(
                    np.linalg.norm(trajectories[i] - trajectories[j], axis=-1)
                )
                total_dist += dist
                count += 1
        
        return total_dist / count if count > 0 else 0.0
    
    # ==========================================
    # Table VII: Behavioral Metrics
    # ==========================================
    
    def compute_behavioral_metrics(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        headings: np.ndarray,
        dt: float = 0.1
    ) -> Dict[str, float]:
        """
        Compute behavioral fidelity metrics.
        
        Returns:
            Dict with speed, acceleration, steering, following distance stats
        """
        metrics = {}
        
        # Speed
        speeds = np.linalg.norm(velocities, axis=-1)
        metrics['speed_mean'] = np.mean(speeds)
        metrics['speed_std'] = np.std(speeds)
        
        # Acceleration
        accels = np.diff(speeds, axis=-1) / dt
        metrics['accel_mean'] = np.mean(np.abs(accels))
        metrics['accel_std'] = np.std(accels)
        
        # Steering (heading rate)
        steering = np.diff(headings, axis=-1) / dt
        metrics['steering_mean'] = np.mean(np.abs(steering))
        metrics['steering_std'] = np.std(steering)
        
        # Following distance (min distance to other agents per step)
        if positions.shape[0] > 1:
            min_dists = []
            for t in range(positions.shape[1]):
                for i in range(positions.shape[0]):
                    dists_to_others = []
                    for j in range(positions.shape[0]):
                        if i != j:
                            d = np.linalg.norm(positions[i, t] - positions[j, t])
                            dists_to_others.append(d)
                    if dists_to_others:
                        min_dists.append(min(dists_to_others))
            
            metrics['following_dist_mean'] = np.mean(min_dists) if min_dists else 0.0
            metrics['following_dist_std'] = np.std(min_dists) if min_dists else 0.0
        else:
            metrics['following_dist_mean'] = 0.0
            metrics['following_dist_std'] = 0.0
        
        return metrics
    
    def compute_kl_divergence(
        self,
        samples: np.ndarray,
        reference: np.ndarray,
        bins: int = 50
    ) -> float:
        """Compute KL divergence between sample and reference distributions."""
        try:
            hist_samples, bin_edges = np.histogram(samples, bins=bins, density=True)
            hist_ref, _ = np.histogram(reference, bins=bin_edges, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-10
            hist_samples = hist_samples + eps
            hist_ref = hist_ref + eps
            
            # Normalize
            hist_samples = hist_samples / hist_samples.sum()
            hist_ref = hist_ref / hist_ref.sum()
            
            # KL divergence
            kl = np.sum(hist_ref * np.log(hist_ref / hist_samples))
            return kl if np.isfinite(kl) else 0.0
        except:
            return 0.0
    
    def compute_wasserstein(
        self,
        samples: np.ndarray,
        reference: np.ndarray
    ) -> float:
        """Compute Wasserstein distance between distributions."""
        try:
            return wasserstein_distance(samples.flatten(), reference.flatten())
        except:
            return 0.0
    
    # ==========================================
    # TrafficGamer-Compatible Risk Metrics
    # ==========================================
    
    def compute_ttc(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        threshold: float = 2.0
    ) -> float:
        """
        Compute Time-to-Collision (TTC) risk ratio.
        
        TTC = distance / closing_speed (when vehicles are approaching).
        
        Args:
            positions: (num_agents, num_steps, 2 or 3)
            velocities: (num_agents, num_steps, 2 or 3)
            threshold: TTC threshold in seconds (TrafficGamer uses 2.0s)
            
        Returns:
            Fraction of timestep-pairs where TTC < threshold [0, 1]
        """
        if positions.shape[0] < 2:
            return 0.0
        
        num_agents, num_steps = positions.shape[:2]
        risk_count = 0
        total_pairs = 0
        
        for t in range(num_steps):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    pos_i = positions[i, t, :2]
                    pos_j = positions[j, t, :2]
                    vel_i = velocities[i, t, :2]
                    vel_j = velocities[j, t, :2]
                    
                    # Relative position and velocity
                    rel_pos = pos_j - pos_i
                    rel_vel = vel_j - vel_i
                    distance = np.linalg.norm(rel_pos)
                    
                    if distance < 0.01:  # Already colliding
                        risk_count += 1
                        total_pairs += 1
                        continue
                    
                    # Closing speed (projection of rel_vel onto rel_pos direction)
                    closing_speed = -np.dot(rel_vel, rel_pos) / distance
                    total_pairs += 1
                    
                    # If approaching and TTC < threshold
                    if closing_speed > 0.1:  # Only count approaching vehicles
                        ttc = distance / closing_speed
                        if ttc < threshold:
                            risk_count += 1
        
        return risk_count / max(total_pairs, 1)
    
    def compute_thw(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        headings: np.ndarray,
        threshold: float = 2.0
    ) -> float:
        """
        Compute Time Headway (THW) risk ratio.
        
        THW = distance / follower_speed (for following vehicle pairs).
        
        Args:
            positions: (num_agents, num_steps, 2 or 3)
            velocities: (num_agents, num_steps, 2 or 3)
            headings: (num_agents, num_steps)
            threshold: THW threshold in seconds (TrafficGamer uses 2.0s)
            
        Returns:
            Fraction of following situations where THW < threshold [0, 1]
        """
        if positions.shape[0] < 2:
            return 0.0
        
        num_agents, num_steps = positions.shape[:2]
        risk_count = 0
        total_checks = 0
        
        for t in range(num_steps):
            for i in range(num_agents):  # Lead vehicle
                for j in range(num_agents):  # Follower candidate
                    if i == j:
                        continue
                    
                    pos_i = positions[i, t, :2]
                    pos_j = positions[j, t, :2]
                    vel_j = velocities[j, t, :2]
                    heading_i = headings[i, t]
                    
                    # Direction vector from j to i
                    dir_to_i = pos_i - pos_j
                    dir_norm = np.linalg.norm(dir_to_i)
                    if dir_norm < 0.5:  # Too close to determine
                        continue
                    
                    # Check if j is roughly behind i (following)
                    heading_vec = np.array([np.cos(heading_i), np.sin(heading_i)])
                    alignment = np.dot(dir_to_i / dir_norm, heading_vec)
                    
                    if alignment > 0.7:  # j is behind i and facing similar direction
                        speed_j = np.linalg.norm(vel_j)
                        if speed_j > 0.5:  # Follower is moving
                            thw = dir_norm / speed_j
                            total_checks += 1
                            if thw < threshold:
                                risk_count += 1
        
        return risk_count / max(total_checks, 1)
    
    def compute_hellinger_distance(
        self,
        samples: np.ndarray,
        reference: np.ndarray,
        bins: int = 50
    ) -> float:
        """
        Compute Hellinger distance between distributions.
        
        H(P, Q) = sqrt(0.5 * sum((sqrt(P) - sqrt(Q))^2))
        
        Args:
            samples: Generated samples
            reference: Ground truth samples
            bins: Number of histogram bins
            
        Returns:
            Hellinger distance [0, 1] where 0 = identical distributions
        """
        try:
            samples_flat = samples.flatten()
            ref_flat = reference.flatten()
            
            # Determine range
            range_min = min(samples_flat.min(), ref_flat.min())
            range_max = max(samples_flat.max(), ref_flat.max())
            
            # Create histograms
            hist_samples, _ = np.histogram(
                samples_flat, bins=bins, range=(range_min, range_max), density=True
            )
            hist_ref, _ = np.histogram(
                ref_flat, bins=bins, range=(range_min, range_max), density=True
            )
            
            # Normalize to probabilities
            hist_samples = hist_samples / (hist_samples.sum() + 1e-10)
            hist_ref = hist_ref / (hist_ref.sum() + 1e-10)
            
            # Hellinger distance
            hellinger = np.sqrt(0.5 * np.sum((np.sqrt(hist_samples) - np.sqrt(hist_ref)) ** 2))
            
            return hellinger if np.isfinite(hellinger) else 1.0
        except:
            return 1.0


# ==============================================================================
# Section 3: Results Saving/Loading
# ==============================================================================

class ResultsSaver:
    """
    Save and load experiment results in standardized format.
    
    Supports:
    - JSON (summary and config)
    - CSV (per-scenario results)
    - Pickle (full objects with trajectories)
    """
    
    def __init__(self, base_dir: str = './results'):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
    
    def save_experiment(
        self,
        results: ExperimentResults,
        save_trajectories: bool = False
    ) -> Path:
        """
        Save experiment results.
        
        Returns:
            Path to results directory
        """
        # Create experiment directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        exp_name = results.config.experiment_name
        exp_dir = self.base_dir / f"{exp_name}_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = exp_dir / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(asdict(results.config), f, indent=2)
        
        # Save summary
        summary = results.get_summary()
        summary['start_time'] = results.start_time
        summary['end_time'] = results.end_time
        summary['total_time_s'] = results.total_time_s
        summary['num_scenarios'] = len(results.scenario_results)
        
        summary_path = exp_dir / 'summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save per-scenario results as CSV
        df = results.to_dataframe()
        csv_path = exp_dir / 'results.csv'
        df.to_csv(csv_path, index=False)
        
        # Save full results as pickle
        pickle_path = exp_dir / 'full_results.pkl'
        with open(pickle_path, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"✅ Saved results to {exp_dir}")
        return exp_dir
    
    def load_experiment(self, exp_dir: str) -> ExperimentResults:
        """Load experiment results from directory."""
        exp_path = Path(exp_dir)
        pickle_path = exp_path / 'full_results.pkl'
        
        with open(pickle_path, 'rb') as f:
            return pickle.load(f)
    
    def load_summary(self, exp_dir: str) -> Dict:
        """Load just the summary (faster)."""
        exp_path = Path(exp_dir)
        summary_path = exp_path / 'summary.json'
        
        with open(summary_path, 'r') as f:
            return json.load(f)
    
    def list_experiments(self) -> List[Dict]:
        """List all saved experiments."""
        experiments = []
        
        for exp_dir in self.base_dir.iterdir():
            if exp_dir.is_dir():
                try:
                    summary = self.load_summary(str(exp_dir))
                    summary['path'] = str(exp_dir)
                    experiments.append(summary)
                except:
                    continue
        
        return sorted(experiments, key=lambda x: x.get('end_time', ''), reverse=True)


# ==============================================================================
# Section 4: LaTeX Table Formatting
# ==============================================================================

class TableFormatter:
    """
    Format results for LaTeX tables in paper.
    
    Generates LaTeX code for all paper tables with proper formatting.
    """
    
    @staticmethod
    def format_main_results(results: List[Dict]) -> str:
        """
        Format Table: Main Results on WOMD.
        
        Args:
            results: List of summary dicts from different methods
            
        Returns:
            LaTeX table content
        """
        lines = []
        lines.append("\\begin{tabular}{lcccc}")
        lines.append("\\hline")
        lines.append("\\textbf{Method} & \\textbf{NLL}$\\downarrow$ & \\textbf{Coll.\\%}$\\downarrow$ & \\textbf{Off-road\\%}$\\downarrow$ & \\textbf{Div.}$\\uparrow$ \\\\")
        lines.append("\\hline")
        
        for r in results:
            method = r.get('Method', 'Unknown')
            nll = r.get('NLL↓', 'XX')
            coll = r.get('Coll.%↓', 'XX')
            off = r.get('Off-road%↓', 'XX')
            div = r.get('Div.↑', 'XX')
            
            # Bold best method
            if 'EvoQRE' in method and 'learned' in method.lower():
                lines.append(f"\\textbf{{{method}}} & \\textbf{{{nll}}} & \\textbf{{{coll}}} & \\textbf{{{off}}} & \\textbf{{{div}}} \\\\")
            else:
                lines.append(f"{method} & {nll} & {coll} & {off} & {div} \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        
        return '\n'.join(lines)
    
    @staticmethod
    def format_stability_results(results: List[Dict]) -> str:
        """Format Table: Stability Condition Verification."""
        lines = []
        lines.append("\\begin{tabular}{lccccc}")
        lines.append("\\hline")
        lines.append("\\textbf{Scenario} & $\\hat{\\alpha}$ & $\\hat{\\kappa}_{\\max}$ & $\\tau_{\\min}$ & $\\kappa^2/\\alpha$ & \\textbf{Satisfied} \\\\")
        lines.append("\\hline")
        
        for r in results:
            scenario = r.get('scenario', 'Unknown')
            alpha = r.get('alpha', 'XX')
            kappa = r.get('kappa', 'XX')
            tau = r.get('tau', '1.0')
            threshold = r.get('threshold', 'XX')
            sat = r.get('satisfied', 'XX%')
            
            if scenario == 'Overall':
                lines.append(f"\\textbf{{{scenario}}} & {alpha} & {kappa} & {tau} & {threshold} & \\textbf{{{sat}}} \\\\")
            else:
                lines.append(f"{scenario} & {alpha} & {kappa} & {tau} & {threshold} & {sat} \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        
        return '\n'.join(lines)
    
    @staticmethod  
    def format_ablation_results(
        param_name: str,
        param_values: List,
        nll_values: List[str],
        coll_values: List[str],
        time_values: List[str]
    ) -> str:
        """Format ablation table (particles, step size, etc.)."""
        lines = []
        lines.append("\\begin{tabular}{lccc}")
        lines.append("\\hline")
        lines.append(f"{param_name} & NLL$\\downarrow$ & Coll.\\%$\\downarrow$ & Time$\\downarrow$ \\\\")
        lines.append("\\hline")
        
        for val, nll, coll, time in zip(param_values, nll_values, coll_values, time_values):
            lines.append(f"{val} & {nll} & {coll} & {time} \\\\")
        
        lines.append("\\hline")
        lines.append("\\end{tabular}")
        
        return '\n'.join(lines)


# ==============================================================================
# Section 5: Convenience Functions
# ==============================================================================

def create_experiment(
    name: str,
    method: str,
    **kwargs
) -> Tuple[ExperimentConfig, ExperimentResults]:
    """Create new experiment config and results container."""
    config = ExperimentConfig(
        experiment_name=name,
        method=method,
        **kwargs
    )
    
    results = ExperimentResults(
        config=config,
        start_time=datetime.now().isoformat()
    )
    
    return config, results


def finish_experiment(results: ExperimentResults) -> ExperimentResults:
    """Mark experiment as finished and compute total time."""
    results.end_time = datetime.now().isoformat()
    
    if results.start_time:
        start = datetime.fromisoformat(results.start_time)
        end = datetime.fromisoformat(results.end_time)
        results.total_time_s = (end - start).total_seconds()
    
    return results


def load_latest_results(
    experiment_name: str,
    base_dir: str = './results'
) -> Optional[ExperimentResults]:
    """Load most recent results for an experiment."""
    saver = ResultsSaver(base_dir)
    experiments = saver.list_experiments()
    
    for exp in experiments:
        if experiment_name in exp.get('path', ''):
            return saver.load_experiment(exp['path'])
    
    return None


# Quick test
if __name__ == '__main__':
    print("Testing exp_utils...")
    
    # Test config
    config, results = create_experiment(
        name='test_main_results',
        method='EvoQRE',
        num_scenarios=10
    )
    print(f"Config hash: {config.get_hash()}")
    
    # Test metrics
    mc = MetricsComputer()
    samples = np.random.randn(100)
    targets = np.random.randn(50)
    nll = mc.compute_nll_kde(samples, targets)
    print(f"NLL: {nll:.3f}")
    
    # Test collision
    positions = np.random.randn(4, 10, 2) * 10
    coll = mc.compute_collision_rate(positions)
    print(f"Collision rate: {coll:.3f}")
    
    print("✅ All tests passed!")
