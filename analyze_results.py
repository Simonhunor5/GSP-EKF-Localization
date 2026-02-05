"""
Analyze and visualize experiment results.

Usage:
    # Analyze a sweep result
    python analyze_results.py results/sweep_20260204_105118.json
    
    # Analyze a single experiment
    python analyze_results.py results/experiment_20260204_105035.json
    
    # Just print summary (no plots)
    python analyze_results.py results/sweep_*.json --no-plot
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np


def load_results(filepath):
    """Load results from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def analyze_single(data, show_plots=True):
    """Analyze a single experiment result."""
    print("=" * 60)
    print("SINGLE EXPERIMENT ANALYSIS")
    print("=" * 60)
    
    config = data.get("config", {})
    metrics = data.get("metrics", {})
    
    print("\nConfiguration:")
    print(f"   sigma_r (noise):  {config.get('sigma_r', 'N/A')}")
    print(f"   Outlier prob:     {config.get('outlier_prob', 'N/A')}")
    print(f"   Heuristic beta:   {config.get('heuristic_beta', 'N/A')}")
    print(f"   GSP gamma:        {config.get('gsp_gamma', 'N/A')}")
    
    print("\nResults (RMSE in meters):")
    print(f"   (A) Fixed R:      {metrics.get('rmse_A', 'N/A'):.4f} m")
    print(f"   (B) Adaptive R:   {metrics.get('rmse_B', 'N/A'):.4f} m")
    print(f"   (C) GSP Smooth:   {metrics.get('rmse_C', 'N/A'):.4f} m")
    
    print("\nWinner:", end=" ")
    rmses = {'A': metrics['rmse_A'], 'B': metrics['rmse_B'], 'C': metrics['rmse_C']}
    winner = min(rmses, key=rmses.get)
    print(f"Method {winner} ({rmses[winner]:.4f} m)")
    
    imp_b = metrics.get('improvement_B_over_A', 0)
    imp_c = metrics.get('improvement_C_over_A', 0)
    print(f"\n   Improvement B over A: {imp_b:+.4f} m ({100*imp_b/metrics['rmse_A']:+.1f}%)")
    print(f"   Improvement C over A: {imp_c:+.4f} m ({100*imp_c/metrics['rmse_A']:+.1f}%)")


def analyze_sweep(data, show_plots=True):
    """Analyze parameter sweep results."""
    print("=" * 60)
    print("PARAMETER SWEEP ANALYSIS")
    print("=" * 60)
    print(f"\nTotal configurations: {len(data)}")
    
    # Extract unique parameter values
    sigma_rs = sorted(set(d['params']['sigma_r'] for d in data))
    outlier_probs = sorted(set(d['params']['outlier_prob'] for d in data))
    betas = sorted(set(d['params']['heuristic_beta'] for d in data))
    gammas = sorted(set(d['params']['gsp_gamma'] for d in data))
    
    print(f"\nParameter ranges:")
    print(f"   σ_r (noise):      {sigma_rs}")
    print(f"   Outlier prob:     {outlier_probs}")
    print(f"   Heuristic β:      {betas}")
    print(f"   GSP γ:            {gammas}")
    
    # Find best configurations for each method
    print("\n" + "=" * 60)
    print("BEST CONFIGURATIONS")
    print("=" * 60)
    
    # Best B improvement
    best_b = max(data, key=lambda d: d['metrics']['improvement_B_mean'])
    print(f"\nBest for Method B (Heuristic Adaptive R):")
    print(f"   Params: sigma={best_b['params']['sigma_r']}, out={best_b['params']['outlier_prob']}, beta={best_b['params']['heuristic_beta']}")
    print(f"   RMSE:   {best_b['metrics']['rmse_B_mean']:.4f} +/- {best_b['metrics']['rmse_B_std']:.4f} m")
    print(f"   Improvement over A: {best_b['metrics']['improvement_B_mean']:+.4f} m")
    
    # Best C improvement
    best_c = max(data, key=lambda d: d['metrics']['improvement_C_mean'])
    print(f"\nBest for Method C (GSP Smoothing):")
    print(f"   Params: sigma={best_c['params']['sigma_r']}, out={best_c['params']['outlier_prob']}, gamma={best_c['params']['gsp_gamma']}")
    print(f"   RMSE:   {best_c['metrics']['rmse_C_mean']:.4f} +/- {best_c['metrics']['rmse_C_std']:.4f} m")
    print(f"   Improvement over A: {best_c['metrics']['improvement_C_mean']:+.4f} m")
    
    # Count wins
    print("\n" + "=" * 60)
    print("METHOD COMPARISON (which method wins most often)")
    print("=" * 60)
    
    wins = {'A': 0, 'B': 0, 'C': 0}
    b_beats_c = 0
    c_beats_b = 0
    
    for d in data:
        m = d['metrics']
        rmses = {'A': m['rmse_A_mean'], 'B': m['rmse_B_mean'], 'C': m['rmse_C_mean']}
        winner = min(rmses, key=rmses.get)
        wins[winner] += 1
        
        if m['rmse_B_mean'] < m['rmse_C_mean']:
            b_beats_c += 1
        else:
            c_beats_b += 1
    
    total = len(data)
    print(f"\n   Method A wins: {wins['A']:3d} / {total} ({100*wins['A']/total:.1f}%)")
    print(f"   Method B wins: {wins['B']:3d} / {total} ({100*wins['B']/total:.1f}%)")
    print(f"   Method C wins: {wins['C']:3d} / {total} ({100*wins['C']/total:.1f}%)")
    print(f"\n   B beats C: {b_beats_c} times")
    print(f"   C beats B: {c_beats_b} times")
    
    # Parameter sensitivity analysis
    print("\n" + "=" * 60)
    print("PARAMETER SENSITIVITY")
    print("=" * 60)
    
    # Effect of outlier probability
    print("\nEffect of OUTLIER PROBABILITY:")
    for op in outlier_probs:
        subset = [d for d in data if d['params']['outlier_prob'] == op]
        avg_imp_b = np.mean([d['metrics']['improvement_B_mean'] for d in subset])
        avg_imp_c = np.mean([d['metrics']['improvement_C_mean'] for d in subset])
        print(f"   out={op:.2f}: Avg improvement B={avg_imp_b:+.4f}m, C={avg_imp_c:+.4f}m")
    
    # Effect of noise level
    print("\nEffect of NOISE LEVEL (sigma_r):")
    for sr in sigma_rs:
        subset = [d for d in data if d['params']['sigma_r'] == sr]
        avg_imp_b = np.mean([d['metrics']['improvement_B_mean'] for d in subset])
        avg_imp_c = np.mean([d['metrics']['improvement_C_mean'] for d in subset])
        print(f"   sigma={sr:.2f}: Avg improvement B={avg_imp_b:+.4f}m, C={avg_imp_c:+.4f}m")
    
    # Effect of beta (for B)
    print("\nEffect of BETA (Method B sensitivity):")
    for beta in betas:
        subset = [d for d in data if d['params']['heuristic_beta'] == beta]
        avg_imp_b = np.mean([d['metrics']['improvement_B_mean'] for d in subset])
        avg_rmse_b = np.mean([d['metrics']['rmse_B_mean'] for d in subset])
        print(f"   β={beta:.1f}: Avg RMSE_B={avg_rmse_b:.4f}m, Avg improvement={avg_imp_b:+.4f}m")
    
    # Effect of gamma (for C)
    print("\nEffect of GAMMA (GSP smoothing strength):")
    for gamma in gammas:
        subset = [d for d in data if d['params']['gsp_gamma'] == gamma]
        avg_imp_c = np.mean([d['metrics']['improvement_C_mean'] for d in subset])
        avg_rmse_c = np.mean([d['metrics']['rmse_C_mean'] for d in subset])
        print(f"   γ={gamma:.1f}: Avg RMSE_C={avg_rmse_c:.4f}m, Avg improvement={avg_imp_c:+.4f}m")
    
    # Recommendations
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    
    # Find optimal beta
    beta_scores = {}
    for beta in betas:
        subset = [d for d in data if d['params']['heuristic_beta'] == beta]
        beta_scores[beta] = np.mean([d['metrics']['improvement_B_mean'] for d in subset])
    best_beta = max(beta_scores, key=beta_scores.get)
    
    # Find optimal gamma
    gamma_scores = {}
    for gamma in gammas:
        subset = [d for d in data if d['params']['gsp_gamma'] == gamma]
        gamma_scores[gamma] = np.mean([d['metrics']['improvement_C_mean'] for d in subset])
    best_gamma = max(gamma_scores, key=gamma_scores.get)
    
    print(f"\n   Best β for Method B: {best_beta} (avg improvement: {beta_scores[best_beta]:+.4f}m)")
    print(f"   Best γ for Method C: {best_gamma} (avg improvement: {gamma_scores[best_gamma]:+.4f}m)")
    
    # When to use which method
    print("\n   When to use each method:")
    
    # Check if C is better at low noise
    low_noise = [d for d in data if d['params']['sigma_r'] == min(sigma_rs) and d['params']['outlier_prob'] == 0]
    if low_noise:
        avg_c = np.mean([d['metrics']['rmse_C_mean'] for d in low_noise])
        avg_b = np.mean([d['metrics']['rmse_B_mean'] for d in low_noise])
        if avg_c < avg_b:
            print(f"   • Low noise, no outliers: Method C is better")
        else:
            print(f"   • Low noise, no outliers: Method B is better")
    
    # Check with outliers
    with_outliers = [d for d in data if d['params']['outlier_prob'] > 0]
    if with_outliers:
        c_wins_outlier = sum(1 for d in with_outliers if d['metrics']['rmse_C_mean'] < d['metrics']['rmse_B_mean'])
        if c_wins_outlier > len(with_outliers) / 2:
            print(f"   • With outliers: Method C tends to be better")
        else:
            print(f"   • With outliers: Method B tends to be better")
    
    if show_plots:
        plot_sweep_results(data, sigma_rs, outlier_probs, betas, gammas)


def plot_sweep_results(data, sigma_rs, outlier_probs, betas, gammas):
    """Generate visualization plots for sweep results."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("\n[WARNING] matplotlib not available, skipping plots")
        return
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: RMSE by outlier probability
    ax1 = axes[0, 0]
    for method, color, label in [('A', 'red', 'Fixed R'), ('B', 'blue', 'Adaptive R'), ('C', 'green', 'GSP')]:
        means = []
        stds = []
        for op in outlier_probs:
            subset = [d for d in data if d['params']['outlier_prob'] == op]
            means.append(np.mean([d['metrics'][f'rmse_{method}_mean'] for d in subset]))
            stds.append(np.mean([d['metrics'][f'rmse_{method}_std'] for d in subset]))
        ax1.errorbar(outlier_probs, means, yerr=stds, marker='o', label=label, color=color, capsize=3)
    ax1.set_xlabel('Outlier Probability')
    ax1.set_ylabel('RMSE [m]')
    ax1.set_title('RMSE vs Outlier Probability')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RMSE by noise level
    ax2 = axes[0, 1]
    for method, color, label in [('A', 'red', 'Fixed R'), ('B', 'blue', 'Adaptive R'), ('C', 'green', 'GSP')]:
        means = []
        stds = []
        for sr in sigma_rs:
            subset = [d for d in data if d['params']['sigma_r'] == sr]
            means.append(np.mean([d['metrics'][f'rmse_{method}_mean'] for d in subset]))
            stds.append(np.mean([d['metrics'][f'rmse_{method}_std'] for d in subset]))
        ax2.errorbar(sigma_rs, means, yerr=stds, marker='o', label=label, color=color, capsize=3)
    ax2.set_xlabel('Noise σ_r')
    ax2.set_ylabel('RMSE [m]')
    ax2.set_title('RMSE vs Noise Level')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Method B RMSE by beta
    ax3 = axes[1, 0]
    for op in outlier_probs:
        means = []
        for beta in betas:
            subset = [d for d in data if d['params']['heuristic_beta'] == beta and d['params']['outlier_prob'] == op]
            means.append(np.mean([d['metrics']['rmse_B_mean'] for d in subset]))
        ax3.plot(betas, means, marker='o', label=f'out={op}')
    ax3.set_xlabel('Beta (β)')
    ax3.set_ylabel('RMSE_B [m]')
    ax3.set_title('Method B: Effect of β')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Method C RMSE by gamma
    ax4 = axes[1, 1]
    for op in outlier_probs:
        means = []
        for gamma in gammas:
            subset = [d for d in data if d['params']['gsp_gamma'] == gamma and d['params']['outlier_prob'] == op]
            means.append(np.mean([d['metrics']['rmse_C_mean'] for d in subset]))
        ax4.plot(gammas, means, marker='o', label=f'out={op}')
    ax4.set_xlabel('Gamma (γ)')
    ax4.set_ylabel('RMSE_C [m]')
    ax4.set_title('Method C: Effect of γ')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/sweep_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nPlot saved to: results/sweep_analysis.png")
    plt.show()


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment results")
    parser.add_argument("file", type=str, help="Path to results JSON file")
    parser.add_argument("--no-plot", action="store_true", help="Skip plotting")
    args = parser.parse_args()
    
    filepath = Path(args.file)
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        sys.exit(1)
    
    data = load_results(filepath)
    show_plots = not args.no_plot
    
    # Detect if it's a sweep or single experiment
    if isinstance(data, list):
        analyze_sweep(data, show_plots)
    elif isinstance(data, dict) and "metrics" in data:
        analyze_single(data, show_plots)
    else:
        print("Error: Unknown result format")
        sys.exit(1)


if __name__ == "__main__":
    main()
