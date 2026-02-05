"""Extended Kalman Filter implementation."""

import numpy as np
from .utils import wrap_pi
from .models import motion_model, jacobian_F, measurement_model, jacobian_H


def ekf_predict(x, P, u, Q, dt):
    """
    EKF prediction step.
    
    Args:
        x: Current state estimate
        P: Current covariance
        u: Control input
        Q: Process noise covariance
        dt: Time step
    
    Returns:
        x_pred: Predicted state
        P_pred: Predicted covariance
    """
    x_pred = motion_model(x, u, dt)
    F = jacobian_F(x, u, dt)
    P_pred = F @ P @ F.T + Q
    return x_pred, P_pred


def ekf_update(x_pred, P_pred, z, anchors, R):
    """
    EKF update step.
    
    Args:
        x_pred: Predicted state
        P_pred: Predicted covariance
        z: Measurement vector
        anchors: Anchor positions
        R: Measurement noise covariance
    
    Returns:
        x_upd: Updated state
        P_upd: Updated covariance
        y: Innovation (residual)
        z_hat: Predicted measurement
    """
    z_hat = measurement_model(x_pred, anchors)
    H = jacobian_H(x_pred, anchors)
    y = z - z_hat
    S = H @ P_pred @ H.T + R
    K = P_pred @ H.T @ np.linalg.inv(S)
    x_upd = x_pred + K @ y
    x_upd[2] = wrap_pi(x_upd[2])
    P_upd = (np.eye(3) - K @ H) @ P_pred
    return x_upd, P_upd, y, z_hat
