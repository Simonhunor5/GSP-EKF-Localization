"""Graph Signal Processing methods for residual smoothing."""

import numpy as np


def try_import_cvxpy():
    """Try to import cvxpy, return None if not available."""
    try:
        import cvxpy as cp
        return cp
    except Exception:
        return None


def learn_laplacian_dong(X, alpha=1e-2, mu=0.0, L_prev=None, solver="SCS"):
    """
    Dong-style Laplacian learning from signals.
    
    Solves:
        min_L  tr(X^T L X) + alpha * ||L||_F^2 + mu * ||L - L_prev||_F^2
        s.t.   L is a valid Laplacian
    
    Args:
        X: (M, T) matrix of graph signals
        alpha: Regularization strength
        mu: Temporal smoothness (weight on deviation from L_prev)
        L_prev: Previous Laplacian (for temporal smoothness)
        solver: CVXPY solver to use
    
    Returns:
        L: (M, M) learned Laplacian matrix
    """
    cp = try_import_cvxpy()
    if cp is None:
        raise RuntimeError("cvxpy not installed. Run: pip install cvxpy")

    M, T = X.shape
    L = cp.Variable((M, M), symmetric=True)

    # Objective: smoothness + regularization
    smooth = cp.trace(X.T @ L @ X)
    reg = alpha * cp.sum_squares(L)

    temp = 0
    if mu > 0.0 and L_prev is not None:
        temp = mu * cp.sum_squares(L - L_prev)

    # Laplacian constraints
    constraints = [
        L @ np.ones(M) == 0,                # Zero row sum
        cp.diag(L) >= 0,                    # Diagonal nonnegative
        L - cp.diag(cp.diag(L)) <= 0,       # Off-diagonal <= 0
        cp.sum(cp.diag(L)) == M,            # Scaling (avoid L=0)
    ]

    prob = cp.Problem(cp.Minimize(smooth + reg + temp), constraints)
    prob.solve(solver=getattr(cp, solver), verbose=False)

    if L.value is None:
        raise RuntimeError("Laplacian learning failed (solver returned None)")

    return L.value


class ResidualWindow:
    """Ring buffer for storing residual vectors."""
    
    def __init__(self, M, T_win):
        """
        Args:
            M: Number of anchors (signal dimension)
            T_win: Window size
        """
        self.M = M
        self.T_win = T_win
        self.buf = np.zeros((M, T_win), dtype=float)
        self.ptr = 0
        self.count = 0

    def push(self, r):
        """Add new residual vector to buffer."""
        self.buf[:, self.ptr] = r
        self.ptr = (self.ptr + 1) % self.T_win
        self.count = min(self.count + 1, self.T_win)

    def full(self):
        """Check if buffer is full."""
        return self.count == self.T_win

    def get_matrix(self):
        """Get residual matrix in chronological order."""
        if self.count < self.T_win:
            return self.buf[:, :self.count].copy()
        idx = np.arange(self.ptr, self.ptr + self.T_win) % self.T_win
        return self.buf[:, idx].copy()

    def reset(self):
        """Reset the buffer."""
        self.buf.fill(0)
        self.ptr = 0
        self.count = 0


def smooth_residual(L, r, gamma=1.0):
    """
    Graph-based residual smoothing (low-pass filter).
    
    Computes: r_tilde = (I + gamma * L)^(-1) * r
    
    Args:
        L: Laplacian matrix
        r: Raw residual vector
        gamma: Smoothing strength
    
    Returns:
        Smoothed residual vector
    """
    M = L.shape[0]
    A = np.eye(M) + gamma * L
    return np.linalg.solve(A, r)
