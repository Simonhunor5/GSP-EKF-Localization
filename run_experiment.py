"""
Run EKF localization experiments with configurable parameters.

Usage:
    # Single run with defaults
    python run_experiment.py
    
    # Single run with custom parameters
    python run_experiment.py --sigma_r 0.25 --outlier_prob 0.1
    
    # Parameter sweep
    python run_experiment.py --sweep
    
    # Show plots
    python run_experiment.py --plot
"""

import argparse
import json
import numpy as np
from datetime import datetime
from pathlib import Path
from itertools import product

from src.experiment import ExperimentConfig, run_experiment
from src.plotting import plot_all


def parse_args():
    parser = argparse.ArgumentParser(description="Run EKF localization experiments")
    
    # Run mode
    parser.add_argument("--sweep", action="store_true", help="Run parameter sweep")
    parser.add_argument("--plot", action="store_true", help="Show plots")
    parser.add_argument("--output", "-o", type=str, default="results", help="Output directory")
    
    # Simulation params
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=350, help="Number of time steps")
    parser.add_argument("--dt", type=float, default=0.1, help="Time step")
    
    # Noise params
    parser.add_argument("--sigma_r", type=float, default=0.18, help="Measurement noise std")
    parser.add_argument("--outlier_prob", type=float, default=0.06, help="Outlier probability")
    parser.add_argument("--outlier_sigma", type=float, default=1.2, help="Outlier noise std")
    
    # Method B params
    parser.add_argument("--heuristic_T_win", type=int, default=15)
    parser.add_argument("--heuristic_beta", type=float, default=2.0)
    
    # Method C params
    parser.add_argument("--gsp_T_win", type=int, default=20)
    parser.add_argument("--gsp_learn_every", type=int, default=10)
    parser.add_argument("--gsp_alpha", type=float, default=0.01)
    parser.add_argument("--gsp_mu", type=float, default=0.2)
    parser.add_argument("--gsp_gamma", type=float, default=1.0)
    
    return parser.parse_args()


def run_single(args) -> dict:
    """Run a single experiment with given args."""
    config = ExperimentConfig(
        seed=args.seed,
        T=args.T,
        dt=args.dt,
        sigma_r=args.sigma_r,
        outlier_prob=args.outlier_prob,
        outlier_sigma=args.outlier_sigma,
        heuristic_T_win=args.heuristic_T_win,
        heuristic_beta=args.heuristic_beta,
        gsp_T_win=args.gsp_T_win,
        gsp_learn_every=args.gsp_learn_every,
        gsp_alpha=args.gsp_alpha,
        gsp_mu=args.gsp_mu,
        gsp_gamma=args.gsp_gamma,
    )
    
    print("Running experiment with config:")
    print(f"  sigma_r={config.sigma_r}, outlier_prob={config.outlier_prob}")
    print(f"  heuristic: T_win={config.heuristic_T_win}, beta={config.heuristic_beta}")
    print(f"  GSP: T_win={config.gsp_T_win}, alpha={config.gsp_alpha}, gamma={config.gsp_gamma}")
    print()
    
    result = run_experiment(config, store_arrays=args.plot)
    
    print("=== Results ===")
    print(f"(A) EKF Fixed R:      RMSE = {result.rmse_A:.4f} m")
    print(f"(B) Heuristic Adapt:  RMSE = {result.rmse_B:.4f} m  (Δ = {result.improvement_B_over_A:+.4f} m)")
    print(f"(C) GSP Smoothing:    RMSE = {result.rmse_C:.4f} m  (Δ = {result.improvement_C_over_A:+.4f} m)")
    
    if not result.cvxpy_available:
        print("\n[WARNING] cvxpy not installed - Method C fell back to fixed R")
    
    if args.plot:
        anchors = np.array(config.anchors)
        plot_all(result, anchors, show=True)
    
    return result.to_dict()


def run_sweep(args) -> list:
    """Run parameter sweep experiments."""
    
    # Define sweep ranges
    sweep_params = {
        "sigma_r": [0.1, 0.18, 0.3],
        "outlier_prob": [0.0, 0.06, 0.12],
        "heuristic_beta": [1.0, 2.0, 4.0],
        "gsp_gamma": [0.5, 1.0, 2.0],
    }
    
    # Seeds for statistical averaging
    seeds = [0, 1, 2, 3, 4]
    
    results = []
    total = len(list(product(*sweep_params.values()))) * len(seeds)
    
    print(f"Running parameter sweep: {total} experiments")
    print(f"Sweep params: {list(sweep_params.keys())}")
    print()
    
    idx = 0
    for values in product(*sweep_params.values()):
        param_dict = dict(zip(sweep_params.keys(), values))
        
        seed_results = []
        for seed in seeds:
            idx += 1
            
            config = ExperimentConfig(
                seed=seed,
                T=args.T,
                dt=args.dt,
                sigma_r=param_dict.get("sigma_r", args.sigma_r),
                outlier_prob=param_dict.get("outlier_prob", args.outlier_prob),
                outlier_sigma=args.outlier_sigma,
                heuristic_T_win=args.heuristic_T_win,
                heuristic_beta=param_dict.get("heuristic_beta", args.heuristic_beta),
                gsp_T_win=args.gsp_T_win,
                gsp_learn_every=args.gsp_learn_every,
                gsp_alpha=args.gsp_alpha,
                gsp_mu=args.gsp_mu,
                gsp_gamma=param_dict.get("gsp_gamma", args.gsp_gamma),
            )
            
            result = run_experiment(config, store_arrays=False)
            seed_results.append(result)
            
            print(f"[{idx}/{total}] {param_dict} seed={seed} -> A={result.rmse_A:.4f}, B={result.rmse_B:.4f}, C={result.rmse_C:.4f}")
        
        # Aggregate over seeds
        rmse_A_mean = np.mean([r.rmse_A for r in seed_results])
        rmse_B_mean = np.mean([r.rmse_B for r in seed_results])
        rmse_C_mean = np.mean([r.rmse_C for r in seed_results])
        rmse_A_std = np.std([r.rmse_A for r in seed_results])
        rmse_B_std = np.std([r.rmse_B for r in seed_results])
        rmse_C_std = np.std([r.rmse_C for r in seed_results])
        
        results.append({
            "params": param_dict,
            "seeds": seeds,
            "metrics": {
                "rmse_A_mean": rmse_A_mean,
                "rmse_A_std": rmse_A_std,
                "rmse_B_mean": rmse_B_mean,
                "rmse_B_std": rmse_B_std,
                "rmse_C_mean": rmse_C_mean,
                "rmse_C_std": rmse_C_std,
                "improvement_B_mean": rmse_A_mean - rmse_B_mean,
                "improvement_C_mean": rmse_A_mean - rmse_C_mean,
            }
        })
    
    print("\n=== Sweep Complete ===")
    print(f"Total experiments: {total}")
    
    # Print summary table
    print("\nSummary (mean ± std over seeds):")
    print("-" * 100)
    for r in results:
        p = r["params"]
        m = r["metrics"]
        print(f"σ={p['sigma_r']:.2f} out={p['outlier_prob']:.2f} β={p['heuristic_beta']:.1f} γ={p['gsp_gamma']:.1f} | "
              f"A={m['rmse_A_mean']:.4f}±{m['rmse_A_std']:.4f}  "
              f"B={m['rmse_B_mean']:.4f}±{m['rmse_B_std']:.4f}  "
              f"C={m['rmse_C_mean']:.4f}±{m['rmse_C_std']:.4f}")
    
    return results


def main():
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.sweep:
        results = run_sweep(args)
        output_file = output_dir / f"sweep_{timestamp}.json"
    else:
        results = run_single(args)
        output_file = output_dir / f"experiment_{timestamp}.json"
    
    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
