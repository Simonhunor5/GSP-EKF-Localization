"""Adaptive measurement weighting methods."""

import numpy as np


class ResidualWeighting:
    """
    Heuristic per-anchor reliability weighting.
    
    Computes weights based on recent residual history:
        e_i(k) = mean(r_i^2) over last T_win steps
        w_i(k) = exp(-beta * e_i(k))
        R_i(k) = sigma_r^2 / (w_i + eps)
    """
    
    def __init__(self, M, T_win=15, beta=2.0, eps=1e-3):
        """
        Args:
            M: Number of anchors
            T_win: Window size for averaging
            beta: Sensitivity parameter
            eps: Small constant to avoid division by zero
        """
        self.M = M
        self.T_win = T_win
        self.beta = beta
        self.eps = eps
        self.buf = np.zeros((M, T_win), dtype=float)
        self.ptr = 0
        self.count = 0

    def update(self, residual):
        """Add new residual to buffer."""
        self.buf[:, self.ptr] = residual
        self.ptr = (self.ptr + 1) % self.T_win
        self.count = min(self.count + 1, self.T_win)

    def get_weights(self):
        """Compute current weights from residual history."""
        if self.count == 0:
            return np.ones(self.M)
        r = self.buf[:, :self.count]
        e = np.mean(r ** 2, axis=1)
        w = np.exp(-self.beta * e)
        return np.clip(w, 1e-6, 1.0)

    def make_R(self, sigma_r):
        """
        Build adaptive R matrix.
        
        Returns:
            R: Diagonal measurement covariance matrix
            w: Current weights
        """
        w = self.get_weights()
        R_diag = (sigma_r ** 2) / (w + self.eps)
        return np.diag(R_diag), w

    def reset(self):
        """Reset the buffer."""
        self.buf.fill(0)
        self.ptr = 0
        self.count = 0
