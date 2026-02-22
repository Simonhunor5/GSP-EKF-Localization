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
from src.utils import wrap_pi


def main():
    # Create output directory
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run experiment (no anchor noise)
    config = ExperimentConfig(seed=0)
    print("Running experiment (no anchor noise)...")
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
    
    # Figure 5: Yaw estimation
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    ax = axes[0]
    ax.plot(np.degrees(result.true_yaw), 'k-', lw=2, label="Valódi yaw")
    ax.plot(np.degrees(result.est_A_yaw), '--', alpha=0.8, label="(A) EKF fix R")
    ax.plot(np.degrees(result.est_B_yaw), '--', alpha=0.8, label="(B) Adaptív R")
    ax.plot(np.degrees(result.est_C_yaw), '--', alpha=0.8, label="(C) GSP simítás")
    ax.set_ylabel("Yaw [°]")
    ax.set_title("Orientációs szög (yaw) becslés")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax = axes[1]
    yaw_err_A = np.array([np.degrees(wrap_pi(e - t)) for e, t in zip(result.est_A_yaw, result.true_yaw)])
    yaw_err_B = np.array([np.degrees(wrap_pi(e - t)) for e, t in zip(result.est_B_yaw, result.true_yaw)])
    yaw_err_C = np.array([np.degrees(wrap_pi(e - t)) for e, t in zip(result.est_C_yaw, result.true_yaw)])
    ax.plot(yaw_err_A, alpha=0.8, label=f"(A) RMSE={np.sqrt(np.mean(yaw_err_A**2)):.2f}°")
    ax.plot(yaw_err_B, alpha=0.8, label=f"(B) RMSE={np.sqrt(np.mean(yaw_err_B**2)):.2f}°")
    ax.plot(yaw_err_C, alpha=0.8, label=f"(C) RMSE={np.sqrt(np.mean(yaw_err_C**2)):.2f}°")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("Yaw hiba [°]")
    ax.set_title("Orientációs szög becslési hibája")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "yaw_estimation.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/yaw_estimation.png")
    
    # Figure 6: State components (x, y, yaw) separately
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    
    # x component
    ax = axes[0]
    ax.plot(result.true_xy[:, 0], 'k-', lw=2, label="Valódi")
    ax.plot(result.est_A_xy[:, 0], '--', alpha=0.7, label="(A) Fix R")
    ax.plot(result.est_B_xy[:, 0], '--', alpha=0.7, label="(B) Adaptív R")
    ax.plot(result.est_C_xy[:, 0], '--', alpha=0.7, label="(C) GSP")
    ax.set_ylabel("p_x [m]")
    ax.set_title("Állapotváltozók becslése")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # y component
    ax = axes[1]
    ax.plot(result.true_xy[:, 1], 'k-', lw=2, label="Valódi")
    ax.plot(result.est_A_xy[:, 1], '--', alpha=0.7, label="(A) Fix R")
    ax.plot(result.est_B_xy[:, 1], '--', alpha=0.7, label="(B) Adaptív R")
    ax.plot(result.est_C_xy[:, 1], '--', alpha=0.7, label="(C) GSP")
    ax.set_ylabel("p_y [m]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    # yaw component
    ax = axes[2]
    ax.plot(np.degrees(result.true_yaw), 'k-', lw=2, label="Valódi")
    ax.plot(np.degrees(result.est_A_yaw), '--', alpha=0.7, label="(A) Fix R")
    ax.plot(np.degrees(result.est_B_yaw), '--', alpha=0.7, label="(B) Adaptív R")
    ax.plot(np.degrees(result.est_C_yaw), '--', alpha=0.7, label="(C) GSP")
    ax.set_xlabel("Időlépések")
    ax.set_ylabel("yaw [°]")
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_dir / "state_components.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/state_components.png")
    
    # Figure 7: Anchor noise effect
    print("\nRunning anchor noise comparison...")
    sigma_a_values = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2]
    n_seeds = 10
    
    rmse_A_list = []
    rmse_B_list = []
    rmse_C_list = []
    
    for sigma_a in sigma_a_values:
        rA, rB, rC = [], [], []
        for seed in range(n_seeds):
            cfg = ExperimentConfig(seed=seed, anchor_noise_sigma=sigma_a)
            res = run_experiment(cfg, store_arrays=False)
            rA.append(res.rmse_A)
            rB.append(res.rmse_B)
            rC.append(res.rmse_C)
        rmse_A_list.append((np.mean(rA), np.std(rA)))
        rmse_B_list.append((np.mean(rB), np.std(rB)))
        rmse_C_list.append((np.mean(rC), np.std(rC)))
        print(f"  sigma_a = {sigma_a:.3f} m | A: {np.mean(rA):.4f} | B: {np.mean(rB):.4f} | C: {np.mean(rC):.4f}")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    means_A = [m for m, s in rmse_A_list]
    stds_A = [s for m, s in rmse_A_list]
    means_B = [m for m, s in rmse_B_list]
    stds_B = [s for m, s in rmse_B_list]
    means_C = [m for m, s in rmse_C_list]
    stds_C = [s for m, s in rmse_C_list]
    
    ax.errorbar(sigma_a_values, means_A, yerr=stds_A, marker='o', capsize=3, label="(A) EKF fix R")
    ax.errorbar(sigma_a_values, means_B, yerr=stds_B, marker='s', capsize=3, label="(B) Adaptív R")
    ax.errorbar(sigma_a_values, means_C, yerr=stds_C, marker='^', capsize=3, label="(C) GSP simítás")
    ax.set_xlabel("Anchor pozíció hiba szórása (σ_a) [m]")
    ax.set_ylabel("RMSE [m]")
    ax.set_title("Anchor pozíció bizonytalanság hatása az RMSE-re")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.savefig(output_dir / "anchor_noise.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/anchor_noise.png")
    
    print("\nAll figures saved to figures/")


if __name__ == "__main__":
    main()
