import itertools as it
import numpy as np
import scipy.linalg as spl
from scipy.interpolate import splrep, splev


def eps(x):
    """
    Returns the machine precision at x.
    I.e., the distance to the nearest distinct finite-precision number.
    Applies coordinate-wise if x is a numpy.array.
    """
    return np.fabs(np.spacing(x))


def mldivide(A, B):
    """
    Returns x, where x solves Ax = B. (A\B in MATLAB)
    """
    return np.linalg.lstsq(A, B, rcond=None)[0]


def mrdivide(B, A):
    """
    Returns x, where x solves B = xA. (B/A in MATLAB)
    """
    return np.linalg.lstsq(A.T, B.T, rcond=None)[0].T


def solve(A, B):
    """
    Returns x, where x solves Ax = B.
    Assumes A is invertible.
    If A is a KxNxN stack of matrices, B should be KxN.
    """
    signature = "dd->d"
    if B.ndim == A.ndim - 1:
        return np.linalg.linalg._umath_linalg.solve1(A, B, signature=signature)
    else:
        return np.linalg.linalg._umath_linalg.solve(A, B, signature=signature)


def minimum_singular_value(A):
    """
    Returns (min_sv, low_rank), where
        min_sv is the minimum singular value of numpy.array A
        low_rank is True only if A is low rank
    """
    # slightly faster than np.linalg.norm/svd:
    min_eig = spl.eigh(A.T.dot(A), eigvals_only=True, eigvals=(0, 1))[0]
    low_rank = min_eig <= 0
    min_sv = 0 if low_rank else np.sqrt(min_eig)
    return min_sv, low_rank


def nr_solve(x, f, Df, ef, max_iterations=None):
    """
    Run Newton-Raphson iterations to solve f(x) = 0
    Inputs:
        x: an initial seed
        f: a function handle to the function f
        Df: a function handle computing the Jacobian of f
            Df(x)[:,:] is the Jacobian of f at x
        ef: a function handle computing the forward error of f
        max_iterations: optional maximum number of iterations
    Returns:
        x: the final solution point
        points[i]: x at the i^th iteration
        residuals[i]: residual error at the i^th iteration
    """
    points = []
    residuals = []
    for i in it.count():
        fx = f(x)
        efx = ef(x)
        points.append(x)
        residuals.append(np.fabs(fx).max())
        if i == max_iterations or not np.isfinite(fx).all():
            break
        if (np.fabs(fx) < efx).all():
            break
        Dfx = Df(x)
        if fx.shape[0] < x.shape[0]:
            x = x - mldivide(Dfx, fx)
        if fx.shape[0] > x.shape[0]:
            x = x - mrdivide(Dfx, fx)
        if fx.shape[0] == x.shape[0]:
            x = x - solve(Dfx, fx)
    return x, points, residuals


def nr_solves(X, f, Df, ef, max_iterations=None):
    """
    Solve multiple f(x) = 0 simultaneously with Newton-Raphson iterations
    Inputs:
        X[:,p]: the p^th initial seed
        f: a function handle to the function f
        Df: a function handle computing the derivative of f
            Df(X)[p,:,:] is the derivative of the p^th point
        ef: a function handle computing the forward error of f
        max_iterations: optional maximum number of iterations
    Returns:
        X: the final solution points
        done[i]: True iff the i^th point converged
        points[i]: X at the i^th iteration
        residuals[i]: maximum residual error at the i^th iteration
    """
    points = []
    residuals = []
    done = np.zeros(X.shape[1], dtype=bool)
    for i in it.count():
        fx = f(X[:, ~done])
        efx = ef(X[:, ~done])
        points.append(X)
        residuals.append(np.fabs(fx).max())
        if i == max_iterations or not np.isfinite(fx).all():
            break
        done_now = (np.fabs(fx) < efx).all(axis=0)
        done[~done] = done_now
        if done_now.all():
            break
        Dfx = Df(X[:, ~done])
        fx = fx[:, ~done_now]
        X[:, ~done] = X[:, ~done] - solve(Dfx, fx.T).T
    return X, done, points, residuals

def _spline_derivs(y, t, orders, k=4, s_smooth=0.0):
    """Coordinatewise B-spline derivatives of y(t).
    y: (N, ...) values, t: (N,) parameter, orders: list[int] derivative orders.
    """
    y = np.asarray(y, dtype=float)
    y_flat = y.reshape(len(t), -1)
    out = {o: np.empty_like(y_flat) for o in orders}
    for j in range(y_flat.shape[1]):
        tck = splrep(t, y_flat[:, j], k=k, s=s_smooth)
        for o in orders:
            out[o][:, j] = splev(t, tck, der=o)
    return {o: out[o].reshape(y.shape) for o in orders}

def curvature_findiff(X):
    """
    First curvature of a sampled curve in R^n via finite differences.

    Uses the parametrization-invariant formula
        kappa = sqrt(|v|^2 |a|^2 - (v . a)^2) / |v|^3
    where v and a are the first and second derivatives of the curve with
    respect to any parameter. This is the n-dim generalization of the
    classical 3D cross-product formula via the Lagrange identity
        |v x a|^2 = |v|^2 |a|^2 - (v . a)^2.

    Parameters
    ----------
    points : (N, n) array_like
        Sample points along the curve, any ambient dimension n >= 2.

    Returns
    -------
    kappa : (N,) ndarray
        Curvature at each sample. Endpoints are less accurate because
        np.gradient falls back to one-sided differences there.
    t : (N,) ndarray
        Cumulative chord-length parameter (useful for plotting kappa(t)).
    """
    X = np.asarray(X, dtype=float)

    # Parametrize by cumulative chord length. This handles non-uniform
    # spacing correctly; using the integer index as parameter would give
    # wrong curvatures whenever samples aren't equally spaced.
    segment_lengths = np.linalg.norm(np.diff(X, axis=0), axis=1)
    t = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    # Numerical derivatives along the curve.
    X_dot     = np.gradient(X,   t, axis=0)
    X_ddot = np.gradient(X_dot, t, axis=0)

    # Per-sample inner products.
    speed_sq = (X_dot     * X_dot    ).sum(axis=1)   # |v|^2
    acc_sq   = (X_ddot * X_ddot).sum(axis=1)   # |a|^2
    v_dot_a  = (X_dot     * X_ddot).sum(axis=1)   # v . a

    # Lagrange identity: |v wedge a|^2 = |v|^2|a|^2 - (v.a)^2.
    # The max(., 0) clip guards against tiny negative results from
    # floating-point cancellation when v and a are nearly parallel.
    wedge_sq = np.maximum(speed_sq * acc_sq - v_dot_a**2, 0.0)

    kappa = np.sqrt(wedge_sq) / speed_sq**1.5
    return kappa, t