"""
Controllability and observability analysis of the EKF system.

Analyzes:
  - Controllability of F (state transition matrix)
  - Observability of (F, H) pair
  - Effect of robot position / anchor geometry on observability

Usage:
    python observability_analysis.py
"""

import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'DejaVu Sans'


def jacobian_F(v, dt, yaw):
    """State transition Jacobi matrix."""
    F = np.eye(3)
    F[0, 2] = -v * dt * np.sin(yaw)
    F[1, 2] = v * dt * np.cos(yaw)
    return F


def jacobian_H(px, py, anchors):
    """Measurement Jacobi matrix."""
    M = anchors.shape[0]
    H = np.zeros((M, 3))
    for i in range(M):
        dx = px - anchors[i, 0]
        dy = py - anchors[i, 1]
        d = np.sqrt(dx**2 + dy**2) + 1e-9
        H[i, 0] = dx / d
        H[i, 1] = dy / d
        H[i, 2] = 0.0
    return H


def controllability_matrix(F, G):
    """
    Compute controllability matrix: C = [G, F*G, F^2*G, ..., F^(n-1)*G].
    G is the input matrix (how control affects state).
    """
    n = F.shape[0]
    m = G.shape[1]
    C = np.zeros((n, n * m))
    col = G.copy()
    for i in range(n):
        C[:, i*m:(i+1)*m] = col
        col = F @ col
    return C


def observability_matrix(F, H):
    """
    Compute observability matrix: O = [H; H*F; H*F^2; ...; H*F^(n-1)].
    """
    n = F.shape[0]
    m = H.shape[0]
    O = np.zeros((n * m, n))
    row = H.copy()
    for i in range(n):
        O[i*m:(i+1)*m, :] = row
        row = row @ F
    return O


def analyze_controllability(v, dt, yaw):
    """Analyze controllability for given parameters."""
    F = jacobian_F(v, dt, yaw)
    
    # G: input matrix (how [v, w] affects [px, py, yaw])
    G = np.array([
        [dt * np.cos(yaw), 0],
        [dt * np.sin(yaw), 0],
        [0, dt]
    ])
    
    C = controllability_matrix(F, G)
    rank_C = np.linalg.matrix_rank(C)
    
    return rank_C, C, F, G


def analyze_observability(v, dt, yaw, px, py, anchors):
    """Analyze observability for given state and anchor config."""
    F = jacobian_F(v, dt, yaw)
    H = jacobian_H(px, py, anchors)
    
    O = observability_matrix(F, H)
    rank_O = np.linalg.matrix_rank(O)
    
    # Singular values give condition information
    sv = np.linalg.svd(O, compute_uv=False)
    
    return rank_O, O, sv


def main():
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    anchors = np.array([
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 8.0],
        [0.0, 8.0],
        [5.0, 4.0],
    ])
    
    v = 0.85
    dt = 0.1
    
    # =============================================
    # 1. Controllability analysis
    # =============================================
    print("=" * 60)
    print("CONTROLLABILITY ANALYSIS")
    print("=" * 60)
    
    yaw_values = np.linspace(-np.pi, np.pi, 12)
    for yaw in yaw_values:
        rank_C, C, F, G = analyze_controllability(v, dt, yaw)
        print(f"  yaw = {np.degrees(yaw):7.1f}°  |  rank(C) = {rank_C}  |  {'CONTROLLABLE' if rank_C == 3 else 'NOT CONTROLLABLE'}")
    
    print(f"\nConclusion: The system is controllable for all yaw values,")
    print(f"because the control inputs (v, w) reach all 3 state variables.")
    
    # Show F and G for a typical yaw
    yaw_ex = 0.2
    F_ex = jacobian_F(v, dt, yaw_ex)
    G_ex = np.array([
        [dt * np.cos(yaw_ex), 0],
        [dt * np.sin(yaw_ex), 0],
        [0, dt]
    ])
    print(f"\nExample (yaw = {np.degrees(yaw_ex):.1f}°):")
    print(f"F =\n{F_ex}")
    print(f"G =\n{G_ex}")
    C_ex = controllability_matrix(F_ex, G_ex)
    print(f"rank(C) = {np.linalg.matrix_rank(C_ex)}")
    
    # =============================================
    # 2. Observability analysis
    # =============================================
    print("\n" + "=" * 60)
    print("OBSERVABILITY ANALYSIS")
    print("=" * 60)
    
    # Test at different positions along the trajectory
    yaw_test = 0.2
    positions = [
        (2.0, 2.0, "Start position"),
        (5.0, 4.0, "Near center anchor"),
        (8.0, 6.0, "Towards top right"),
        (5.0, 0.0, "Bottom edge"),
        (0.0, 0.0, "On Anchor 0"),
    ]
    
    print(f"\nAnchors: {anchors.tolist()}")
    print(f"v = {v}, dt = {dt}, yaw = {np.degrees(yaw_test):.1f}°\n")
    
    for px, py, label in positions:
        rank_O, O, sv = analyze_observability(v, dt, yaw_test, px, py, anchors)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float('inf')
        print(f"  [{label:30s}] (px={px:.1f}, py={py:.1f}) | rank(O) = {rank_O} | "
              f"singular values: [{', '.join(f'{s:.4f}' for s in sv)}] | cond = {cond:.1f}")
    
    print(f"\nNote: The third column of the H matrix (yaw derivative) is 0 everywhere,")
    print(f"but yaw is indirectly observable through the F matrix,")
    print(f"because yaw changes affect position prediction.")
    
    # =============================================
    # 3. Observability heatmap
    # =============================================
    print("\n" + "=" * 60)
    print("GENERATING OBSERVABILITY HEATMAP...")
    print("=" * 60)
    
    nx, ny = 50, 40
    xs = np.linspace(-1, 11, nx)
    ys = np.linspace(-1, 9, ny)
    obs_map = np.zeros((ny, nx))
    cond_map = np.zeros((ny, nx))
    
    for ix, px in enumerate(xs):
        for iy, py in enumerate(ys):
            rank_O, O, sv = analyze_observability(v, dt, yaw_test, px, py, anchors)
            obs_map[iy, ix] = rank_O
            cond_map[iy, ix] = sv[-1]  # smallest singular value
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Rank map
    ax = axes[0]
    im = ax.contourf(xs, ys, obs_map, levels=[0.5, 1.5, 2.5, 3.5], colors=['red', 'orange', 'green'])
    ax.scatter(anchors[:, 0], anchors[:, 1], marker='x', s=100, c='black', label='Anchorok', zorder=5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Megfigyelhetőségi mátrix rangja')
    ax.legend()
    ax.set_aspect('equal')
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_ticks([1, 2, 3])
    cbar.set_ticklabels(['1', '2', '3'])
    
    # Smallest singular value (condition)
    ax = axes[1]
    im = ax.contourf(xs, ys, cond_map, levels=20, cmap='viridis')
    ax.scatter(anchors[:, 0], anchors[:, 1], marker='x', s=100, c='red', label='Anchorok', zorder=5)
    ax.set_xlabel('x [m]')
    ax.set_ylabel('y [m]')
    ax.set_title('Legkisebb szinguláris érték (megfigyelhetőség minősége)')
    ax.legend()
    ax.set_aspect('equal')
    plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.savefig(output_dir / "observability.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir}/observability.png")
    
    # =============================================
    # 4. Observability vs speed
    # =============================================
    print("\nObservability for different speed values (px=5, py=4):")
    px_t, py_t = 5.0, 4.0
    v_vals = [0.0, 0.1, 0.5, 0.85, 2.0, 5.0]
    for v_test in v_vals:
        rank_O, O, sv = analyze_observability(v_test, dt, yaw_test, px_t, py_t, anchors)
        cond = sv[0] / sv[-1] if sv[-1] > 1e-12 else float('inf')
        print(f"  v = {v_test:5.2f} m/s | rank(O) = {rank_O} | min_sv = {sv[-1]:.6f} | cond = {cond:.1f}")
    
    print(f"\nConclusion: At v=0, yaw is not directly observable")
    print(f"(because the yaw-position coupling term in the F matrix vanishes),")
    print(f"but for v>0, the system is fully observable.")
    
    print("\nDone!")


if __name__ == "__main__":
    main()
