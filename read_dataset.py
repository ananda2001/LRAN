import numpy as np
from scipy.io import loadmat


# ── Pendulum simulator ────────────────────────────────────────────────────────

def _pendulum(x, u, g=9.81, l=1.0):
    """Simple pendulum with direct torque: θ'' = -(g/l)sin(θ) + u"""
    return np.array([x[1], -(g / l) * np.sin(x[0]) + u])


def _rk4(f, x, u, dt):
    k1 = f(x, u)
    k2 = f(x + 0.5 * dt * k1, u)
    k3 = f(x + 0.5 * dt * k2, u)
    k4 = f(x + dt * k3, u)
    return x + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)


def generate_pendulum(n_traj=100, T=200, dt=0.05, u_max=6.0, seed=0):
    """
    Generate trajectories of the simple controlled pendulum.

    Returns
    -------
    X : (n_traj, T+1, 2)  states [θ, θ']
    U : (n_traj, T,   1)  control torques
    """
    rng = np.random.default_rng(seed)
    X = np.zeros((n_traj, T + 1, 2), dtype=np.float32)
    U = np.zeros((n_traj, T,     1), dtype=np.float32)
    for i in range(n_traj):
        x  = np.array([rng.uniform(-np.pi / 2, np.pi / 2), rng.uniform(-1.0, 1.0)])
        us = rng.uniform(-u_max, u_max, size=T).astype(np.float32)
        X[i, 0] = x
        for t in range(T):
            x = _rk4(_pendulum, x, us[t], dt)
            X[i, t + 1] = x
        U[i, :, 0] = us
    return X, U


# ── .mat loader ───────────────────────────────────────────────────────────────

def load_from_mat(path, x_key='X', u_key='U'):
    """
    Load state/control data from a .mat file.

    Expected shapes:
        X : (T, n_x)        or  (n_traj, T, n_x)
        U : (T-1, n_u)      or  (n_traj, T-1, n_u)

    T is the number of state snapshots; there are T-1 control steps between them.
    """
    data = loadmat(path)
    X = np.array(data[x_key], dtype=np.float32)
    U = np.array(data[u_key], dtype=np.float32)
    if X.ndim == 2:
        X, U = X[None], U[None]
    return X, U


# ── Normalization ─────────────────────────────────────────────────────────────

def normalize(X, U):
    """
    Min-max normalize X and U to [-1, 1] per feature.

    Returns normalized arrays and a scale dict needed for denormalization.
    """
    def _minmax(arr):
        flat = arr.reshape(-1, arr.shape[-1])
        lo   = flat.min(axis=0)
        rng  = np.ptp(flat, axis=0)
        rng[rng == 0] = 1.0
        return (2 * (arr - lo) / rng - 1).astype(np.float32), lo, rng

    X_n, x_lo, x_rng = _minmax(X)
    U_n, u_lo, u_rng = _minmax(U)
    scale = dict(x_lo=x_lo, x_rng=x_rng, u_lo=u_lo, u_rng=u_rng)
    return X_n, U_n, scale


def denormalize(arr, lo, rng):
    return (arr + 1) / 2 * rng + lo


# ── Windowing ─────────────────────────────────────────────────────────────────

def make_windows(X, U, steps):
    """
    Slice trajectories into overlapping windows of length steps+1.

    X : (n_traj, T, n_x)
    U : (n_traj, T-1, n_u)

    Returns a flat list of 2*steps+1 arrays:
        [x_0, x_1, ..., x_steps, u_0, ..., u_{steps-1}]
    each of shape (N_windows, dim), ready for TensorDataset.
    """
    T  = X.shape[1]
    xs = [[] for _ in range(steps + 1)]
    us = [[] for _ in range(steps)]
    for x_traj, u_traj in zip(X, U):
        for t in range(T - steps):
            for k in range(steps + 1):
                xs[k].append(x_traj[t + k])
            for k in range(steps):
                us[k].append(u_traj[t + k])
    xs = [np.stack(a) for a in xs]
    us = [np.stack(a) for a in us]
    return xs + us
