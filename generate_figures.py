"""
Generate figures for documentation.

Usage:
    python generate_figures.py
    
Saves figures to figures/ directory.
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

plt.rcParams['font.family'] = 'DejaVu Sans'

from src.experiment import ExperimentConfig, run_experiment


def main():
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment
    config = ExperimentConfig(seed=0)
    print("Running experiment...")
    result = run_experiment(config, store_arrays=True)
    
    anchors = np.array(config.anchors)
    
    print(f"RMSE A: {result.rmse_A:.4f} m")
    print(f"RMSE B: {result.rmse_B:.4f} m")
    print(f"RMSE C: {result.rmse_C:.4f} m")
    
    # Figure 1: Trajectories
    plt.figure(figsize=(10, 8))
    plt.plot(result.true_xy[:, 0], result.true_xy[:, 1], 'k-', lw=2, label="Valódi pálya")
    plt.plot(result.est_A_xy[:, 0], result.est_A_xy[:, 1], '--', label=f"(A) EKF fix R (RMSE={result.rmse_A:.3f}m)")
    plt.plot(result.est_B_xy[:, 0], result.est_B_xy[:, 1], '--', label=f"(B) Adaptív R (RMSE={result.rmse_B:.3f}m)")
    plt.plot(result.est_C_xy[:, 0], result.est_C_xy[:, 1], '--', label=f"(C) GSP simítás (RMSE={result.rmse_C:.3f}m)")
    plt.scatter(anchors[:, 0], anchors[:, 1], marker="x", s=100, c='red', label="Anchorok")
    plt.title("Trajektóriák összehasonlítása")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    plt.savefig(output_dir / "traj.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/traj.png")
    
    # Figure 2: Position errors
    plt.figure(figsize=(12, 5))
    plt.plot(np.sqrt(result.err_A), label="(A) EKF fix R", alpha=0.8)
    plt.plot(np.sqrt(result.err_B), label="(B) Adaptív R", alpha=0.8)
    plt.plot(np.sqrt(result.err_C), label="(C) GSP simítás", alpha=0.8)
    plt.title("Pozícióhiba időben")
    plt.xlabel("Időlépések")
    plt.ylabel("Hiba [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "error.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/error.png")
    
    # Figure 3: Weights (Method B)
    M = result.weights_B.shape[1]
    plt.figure(figsize=(12, 5))
    for i in range(M):
        plt.plot(result.weights_B[:, i], label=f"Anchor {i}")
    plt.title("Heurisztikus súlyok időben (B módszer)")
    plt.xlabel("Időlépések")
    plt.ylabel("Súly")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "weights.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/weights.png")
    
    # Figure 4: Residual norms (Method C)
    plt.figure(figsize=(12, 5))
    plt.plot(result.resid_norm_raw, label="Nyers ||r||", alpha=0.7)
    plt.plot(result.resid_norm_smooth, label="Simított ||r̃||", alpha=0.7)
    plt.title("Reziduál norma: nyers vs GSP-simított (C módszer)")
    plt.xlabel("Időlépések")
    plt.ylabel("Norma")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig(output_dir / "residual_norm.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/residual_norm.png")
    
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
