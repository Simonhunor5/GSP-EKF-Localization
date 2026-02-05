"""Utility functions for EKF localization."""

import numpy as np


def wrap_pi(a):
    """Wrap angle to [-pi, pi]."""
    return (a + np.pi) % (2 * np.pi) - np.pi
