"""Experiment runner for comparing EKF methods."""

import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any

from .models import motion_model, measurement_model
from .ekf import ekf_predict, ekf_update
from .adaptive import ResidualWeighting
from .gsp import try_import_cvxpy, learn_laplacian_dong, ResidualWindow, smooth_residual


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    # Simulation
    seed: int = 0
    T: int = 350
    dt: float = 0.1
    
    # Robot control
    v: float = 0.85
    w: float = 0.22
    
    # Initial state
    x0_true: List[float] = field(default_factory=lambda: [2.0, 2.0, 0.2])
    x0_est: List[float] = field(default_factory=lambda: [1.0, 1.0, 0.0])
    P0_diag: List[float] = field(default_factory=lambda: [2.25, 2.25, 0.122])  # [1.5^2, 1.5^2, (20deg)^2]
    
    # Process noise
    Q_diag: List[float] = field(default_factory=lambda: [0.0009, 0.0009, 0.0003])  # [0.03^2, 0.03^2, (1deg)^2]
    
    # Measurement noise
    sigma_r: float = 0.18
    outlier_prob: float = 0.06
    outlier_sigma: float = 1.2
    
    # Anchors
    anchors: List[List[float]] = field(default_factory=lambda: [
        [0.0, 0.0],
        [10.0, 0.0],
        [10.0, 8.0],
        [0.0, 8.0],
        [5.0, 4.0],
    ])
    
    # Method B: Heuristic adaptive R
    heuristic_T_win: int = 15
    heuristic_beta: float = 2.0
    heuristic_eps: float = 1e-3
    
    # Method C: GSP Dong-style
    gsp_T_win: int = 20
    gsp_learn_every: int = 10
    gsp_alpha: float = 1e-2
    gsp_mu: float = 0.2
    gsp_gamma: float = 1.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ExperimentResult:
    """Results from a single experiment run."""
    
    config: Dict[str, Any]
    rmse_A: float
    rmse_B: float
    rmse_C: float
    improvement_B_over_A: float
    improvement_C_over_A: float
    cvxpy_available: bool
    
    # Optional detailed data (not saved to JSON by default)
    true_xy: Optional[np.ndarray] = None
    est_A_xy: Optional[np.ndarray] = None
    est_B_xy: Optional[np.ndarray] = None
    est_C_xy: Optional[np.ndarray] = None
    err_A: Optional[np.ndarray] = None
    err_B: Optional[np.ndarray] = None
    err_C: Optional[np.ndarray] = None
    weights_B: Optional[np.ndarray] = None
    resid_norm_raw: Optional[np.ndarray] = None
    resid_norm_smooth: Optional[np.ndarray] = None
    
    def to_dict(self, include_arrays=False) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        d = {
            "config": self.config,
            "metrics": {
                "rmse_A": self.rmse_A,
                "rmse_B": self.rmse_B,
                "rmse_C": self.rmse_C,
                "improvement_B_over_A": self.improvement_B_over_A,
                "improvement_C_over_A": self.improvement_C_over_A,
            },
            "cvxpy_available": self.cvxpy_available,
        }
        if include_arrays and self.true_xy is not None:
            d["arrays"] = {
                "true_xy": self.true_xy.tolist(),
                "est_A_xy": self.est_A_xy.tolist(),
                "est_B_xy": self.est_B_xy.tolist(),
                "est_C_xy": self.est_C_xy.tolist(),
            }
        return d


def run_experiment(config: ExperimentConfig, store_arrays: bool = True) -> ExperimentResult:
    """
    Run a single experiment comparing methods A, B, C.
    
    Args:
        config: Experiment configuration
        store_arrays: Whether to store trajectory arrays in result
    
    Returns:
        ExperimentResult with metrics and optionally trajectory data
    """
    rng = np.random.default_rng(config.seed)
    
    # Setup
    anchors = np.array(config.anchors)
    M = anchors.shape[0]
    T = config.T
    dt = config.dt
    
    Q = np.diag(config.Q_diag)
    R_fixed = (config.sigma_r ** 2) * np.eye(M)
    
    # Initial state
    x_true = np.array(config.x0_true)
    
    def init_filter():
        return np.array(config.x0_est), np.diag(config.P0_diag)
    
    xA, PA = init_filter()
    xB, PB = init_filter()
    xC, PC = init_filter()
    
    # Method B: Heuristic
    rw = ResidualWeighting(
        M, 
        T_win=config.heuristic_T_win, 
        beta=config.heuristic_beta, 
        eps=config.heuristic_eps
    )
    
    # Method C: GSP
    use_cvxpy = try_import_cvxpy() is not None
    rwin = ResidualWindow(M, T_win=config.gsp_T_win)
    L_prev = None
    L_current = None
    
    # Storage
    true_xy = np.zeros((T, 2))
    estA_xy = np.zeros((T, 2))
    estB_xy = np.zeros((T, 2))
    estC_xy = np.zeros((T, 2))
    errA = np.zeros(T)
    errB = np.zeros(T)
    errC = np.zeros(T)
    weightsB = np.zeros((T, M))
    resid_norm_raw = np.zeros(T)
    resid_norm_smooth = np.zeros(T)
    
    # Main loop
    for k in range(T):
        u = np.array([config.v, config.w])
        
        # True state propagation
        x_true = motion_model(x_true, u, dt)
        
        # Measurement with noise
        z_true = measurement_model(x_true, anchors)
        z = z_true + rng.normal(0.0, config.sigma_r, size=M)
        
        # Outlier injection
        if rng.random() < config.outlier_prob:
            j = int(rng.integers(0, M))
            z[j] += rng.normal(0.0, config.outlier_sigma)
        
        # (A) Fixed R
        x_predA, P_predA = ekf_predict(xA, PA, u, Q, dt)
        xA, PA, _, _ = ekf_update(x_predA, P_predA, z, anchors, R_fixed)
        
        # (B) Heuristic adaptive R
        x_predB, P_predB = ekf_predict(xB, PB, u, Q, dt)
        z_hatB = measurement_model(x_predB, anchors)
        residualB = z - z_hatB
        rw.update(residualB)
        R_adapt, w_vec = rw.make_R(config.sigma_r)
        xB, PB, _, _ = ekf_update(x_predB, P_predB, z, anchors, R_adapt)
        weightsB[k] = w_vec
        
        # (C) GSP residual smoothing
        x_predC, P_predC = ekf_predict(xC, PC, u, Q, dt)
        z_hatC = measurement_model(x_predC, anchors)
        rC = z - z_hatC
        rwin.push(rC)
        resid_norm_raw[k] = float(np.linalg.norm(rC))
        
        # Learn Laplacian periodically
        if use_cvxpy and rwin.full() and (k % config.gsp_learn_every == 0):
            X = rwin.get_matrix()
            try:
                L_current = learn_laplacian_dong(
                    X, 
                    alpha=config.gsp_alpha, 
                    mu=config.gsp_mu, 
                    L_prev=L_prev
                )
                L_prev = L_current
            except Exception:
                pass
        
        # Smooth residual if L available
        if L_current is not None:
            rC_tilde = smooth_residual(L_current, rC, gamma=config.gsp_gamma)
        else:
            rC_tilde = rC.copy()
        
        resid_norm_smooth[k] = float(np.linalg.norm(rC_tilde))
        
        # Update with smoothed measurement
        z_tilde = z_hatC + rC_tilde
        xC, PC, _, _ = ekf_update(x_predC, P_predC, z_tilde, anchors, R_fixed)
        
        # Log
        true_xy[k] = x_true[:2]
        estA_xy[k] = xA[:2]
        estB_xy[k] = xB[:2]
        estC_xy[k] = xC[:2]
        errA[k] = np.sum((estA_xy[k] - true_xy[k]) ** 2)
        errB[k] = np.sum((estB_xy[k] - true_xy[k]) ** 2)
        errC[k] = np.sum((estC_xy[k] - true_xy[k]) ** 2)
    
    # Compute RMSE
    rmseA = float(np.sqrt(np.mean(errA)))
    rmseB = float(np.sqrt(np.mean(errB)))
    rmseC = float(np.sqrt(np.mean(errC)))
    
    result = ExperimentResult(
        config=config.to_dict(),
        rmse_A=rmseA,
        rmse_B=rmseB,
        rmse_C=rmseC,
        improvement_B_over_A=rmseA - rmseB,
        improvement_C_over_A=rmseA - rmseC,
        cvxpy_available=use_cvxpy,
    )
    
    if store_arrays:
        result.true_xy = true_xy
        result.est_A_xy = estA_xy
        result.est_B_xy = estB_xy
        result.est_C_xy = estC_xy
        result.err_A = errA
        result.err_B = errB
        result.err_C = errC
        result.weights_B = weightsB
        result.resid_norm_raw = resid_norm_raw
        result.resid_norm_smooth = resid_norm_smooth
    
    return result
