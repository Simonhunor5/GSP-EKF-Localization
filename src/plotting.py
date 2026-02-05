"""Plotting utilities for experiment visualization."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from .experiment import ExperimentResult


def plot_trajectories(result: ExperimentResult, anchors: np.ndarray, save_path: Optional[str] = None):
    """Plot estimated trajectories vs ground truth."""
    if result.true_xy is None:
        raise ValueError("Result does not contain trajectory data")
    
    plt.figure(figsize=(10, 8))
    plt.plot(result.true_xy[:, 0], result.true_xy[:, 1], 'k-', lw=2, label="True")
    plt.plot(result.est_A_xy[:, 0], result.est_A_xy[:, 1], '--', label=f"(A) Fixed R (RMSE={result.rmse_A:.3f}m)")
    plt.plot(result.est_B_xy[:, 0], result.est_B_xy[:, 1], '--', label=f"(B) Adaptive R (RMSE={result.rmse_B:.3f}m)")
    plt.plot(result.est_C_xy[:, 0], result.est_C_xy[:, 1], '--', label=f"(C) GSP Smooth (RMSE={result.rmse_C:.3f}m)")
    plt.scatter(anchors[:, 0], anchors[:, 1], marker="x", s=100, c='red', label="Anchors")
    
    plt.title("Trajectory Comparison")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.axis('equal')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_errors(result: ExperimentResult, save_path: Optional[str] = None):
    """Plot position error over time."""
    if result.err_A is None:
        raise ValueError("Result does not contain error data")
    
    plt.figure(figsize=(12, 5))
    plt.plot(np.sqrt(result.err_A), label="(A) Fixed R", alpha=0.8)
    plt.plot(np.sqrt(result.err_B), label="(B) Adaptive R", alpha=0.8)
    plt.plot(np.sqrt(result.err_C), label="(C) GSP Smooth", alpha=0.8)
    
    plt.title("Position Error Over Time")
    plt.xlabel("Time Step")
    plt.ylabel("Error [m]")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_weights(result: ExperimentResult, save_path: Optional[str] = None):
    """Plot heuristic weights over time (Method B)."""
    if result.weights_B is None:
        raise ValueError("Result does not contain weight data")
    
    M = result.weights_B.shape[1]
    plt.figure(figsize=(12, 5))
    
    for i in range(M):
        plt.plot(result.weights_B[:, i], label=f"Anchor {i}")
    
    plt.title("Heuristic Weights Over Time (Method B)")
    plt.xlabel("Time Step")
    plt.ylabel("Weight")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_residual_norms(result: ExperimentResult, save_path: Optional[str] = None):
    """Plot raw vs smoothed residual norms (Method C)."""
    if result.resid_norm_raw is None:
        raise ValueError("Result does not contain residual data")
    
    plt.figure(figsize=(12, 5))
    plt.plot(result.resid_norm_raw, label="Raw ||r||", alpha=0.7)
    plt.plot(result.resid_norm_smooth, label="Smoothed ||r̃||", alpha=0.7)
    
    plt.title("Residual Norm: Raw vs GSP-Smoothed (Method C)")
    plt.xlabel("Time Step")
    plt.ylabel("Norm")
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return plt.gcf()


def plot_all(result: ExperimentResult, anchors: np.ndarray, show: bool = True):
    """Generate all plots for an experiment result."""
    figs = []
    figs.append(plot_trajectories(result, anchors))
    figs.append(plot_errors(result))
    figs.append(plot_weights(result))
    figs.append(plot_residual_norms(result))
    
    if show:
        plt.show()
    
    return figs
