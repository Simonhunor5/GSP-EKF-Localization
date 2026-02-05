"""Motion and measurement models for robot localization."""

import numpy as np
from .utils import wrap_pi


def motion_model(x, u, dt):
    """
    Unicycle motion model.
    
    Args:
        x: State [px, py, yaw]
        u: Control [v, w] (linear velocity, angular velocity)
        dt: Time step
    
    Returns:
        New state vector
    """
    px, py, yaw = x
    v, w = u
    px2 = px + v * dt * np.cos(yaw)
    py2 = py + v * dt * np.sin(yaw)
    yaw2 = wrap_pi(yaw + w * dt)
    return np.array([px2, py2, yaw2])


def jacobian_F(x, u, dt):
    """
    Jacobian of motion model w.r.t. state.
    
    Args:
        x: State [px, py, yaw]
        u: Control [v, w]
        dt: Time step
    
    Returns:
        3x3 Jacobian matrix F
    """
    px, py, yaw = x
    v, w = u
    F = np.eye(3)
    F[0, 2] = -v * dt * np.sin(yaw)
    F[1, 2] = v * dt * np.cos(yaw)
    return F


def measurement_model(x, anchors):
    """
    Range measurement model.
    
    Args:
        x: State [px, py, yaw]
        anchors: (M, 2) array of anchor positions
    
    Returns:
        (M,) array of ranges to each anchor
    """
    px, py, yaw = x
    dx = anchors[:, 0] - px
    dy = anchors[:, 1] - py
    return np.sqrt(dx * dx + dy * dy)


def jacobian_H(x, anchors):
    """
    Jacobian of measurement model w.r.t. state.
    
    Args:
        x: State [px, py, yaw]
        anchors: (M, 2) array of anchor positions
    
    Returns:
        (M, 3) Jacobian matrix H
    """
    px, py, yaw = x
    dx = px - anchors[:, 0]
    dy = py - anchors[:, 1]
    r = np.sqrt(dx * dx + dy * dy) + 1e-9
    H = np.zeros((anchors.shape[0], 3))
    H[:, 0] = dx / r
    H[:, 1] = dy / r
    return H
