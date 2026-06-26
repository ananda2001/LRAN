import argparse
import os
import time
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.io import savemat
import matplotlib.pyplot as plt

from model import LRAN
from train import train as train_lran
from read_dataset import (generate_pendulum, load_from_mat,
                           normalize, denormalize, make_windows)

# ── Arguments ─────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description='LRAN — controlled dynamical systems')

# Data
parser.add_argument('--dataset',    default='pendulum',
                    help='"pendulum" or path to a .mat file')
parser.add_argument('--x_key',      default='X',   help='key for states in .mat file')
parser.add_argument('--u_key',      default='U',   help='key for controls in .mat file')
parser.add_argument('--n_traj',     type=int,   default=100,
                    help='number of trajectories to generate (pendulum only)')
parser.add_argument('--traj_len',   type=int,   default=200,
                    help='number of control steps per trajectory (pendulum only)')
parser.add_argument('--train_frac', type=float, default=0.8,
                    help='fraction of trajectories used for training')

# Architecture
parser.add_argument('--n_z',        type=int,   default=8,
                    help='latent (Koopman) dimension')
parser.add_argument('--alpha',      type=int,   default=4,
                    help='network width multiplier (hidden layer width = 16*alpha)')
parser.add_argument('--init_scale', type=float, default=0.99,
                    help='initial spectral radius of A')

# Training
parser.add_argument('--steps',      type=int,   default=8,
                    help='multi-step prediction horizon during training')
parser.add_argument('--epochs',     type=int,   default=500)
parser.add_argument('--batch_size', type=int,   default=128)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--wd',         type=float, default=1e-4)
parser.add_argument('--gradclip',   type=float, default=0.05)
parser.add_argument('--gamma_id',   type=float, default=1.0)
parser.add_argument('--gamma_fwd',  type=float, default=1.0)
parser.add_argument('--gamma_lin',  type=float, default=1.0)
parser.add_argument('--gamma_eig',  type=float, default=0.0,
                    help='weight on eigenvalue stability loss (0 = disabled)')

parser.add_argument('--seed',       type=int,   default=0)
parser.add_argument('--device',     type=str,   default='cpu')
parser.add_argument('--out_dir',    type=str,   default='',
                    help='directory to save per-run metrics (empty = skip)')
parser.add_argument('--no_plot',    action='store_true',
                    help='suppress figure output (use on HPC without display)')
args = parser.parse_args()

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# ── Data ──────────────────────────────────────────────────────────────────────
if args.dataset == 'pendulum':
    X, U = generate_pendulum(n_traj=args.n_traj, T=args.traj_len, seed=args.seed)
else:
    X, U = load_from_mat(args.dataset, x_key=args.x_key, u_key=args.u_key)

n_x, n_u = X.shape[-1], U.shape[-1]
X_n, U_n, scale = normalize(X, U)

n_train      = int(args.train_frac * len(X_n))
X_tr, U_tr   = X_n[:n_train], U_n[:n_train]
X_te, U_te   = X_n[n_train:], U_n[n_train:]

windows = make_windows(X_tr, U_tr, args.steps)
dataset = TensorDataset(*[torch.from_numpy(w) for w in windows])
loader  = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

print(f'Train trajectories: {n_train}  |  Test trajectories: {len(X_te)}')
print(f'Windows per epoch:  {len(dataset)}  |  Batches per epoch: {len(loader)}')

# ── Model ─────────────────────────────────────────────────────────────────────
model   = LRAN(n_x, n_u, args.n_z, args.alpha, args.init_scale)
n_param = sum(p.numel() for p in model.parameters())
print(f'LRAN | n_x={n_x}  n_u={n_u}  n_z={args.n_z}  alpha={args.alpha} (width={16*args.alpha}) | params={n_param:,}')

# ── Train ─────────────────────────────────────────────────────────────────────
t0 = time.time()
history = train_lran(
    model, loader, args.epochs,
    lr=args.lr, wd=args.wd, gradclip=args.gradclip,
    gamma_id=args.gamma_id, gamma_fwd=args.gamma_fwd,
    gamma_lin=args.gamma_lin, gamma_eig=args.gamma_eig,
    device=args.device, print_every=50,
)
elapsed = time.time() - t0

if args.out_dir:
    os.makedirs(args.out_dir, exist_ok=True)
model_path = os.path.join(args.out_dir, 'model.pt') if args.out_dir else 'lran_model.pt'
torch.save({'state_dict': model.state_dict(), 'args': vars(args),
            'scale': scale, 'n_x': n_x, 'n_u': n_u}, model_path)
print(f'Model saved → {model_path}')

# ── Evaluate: roll out a test trajectory from its initial condition ────────────
model.eval()
with torch.no_grad():
    x0   = torch.from_numpy(X_te[0, :1])           # (1, n_x)
    us   = torch.from_numpy(U_te[0]).unsqueeze(0)  # (1, T, n_u)
    z0   = model.encoder(x0)
    z_preds  = model.rollout(z0, us)
    x_hat = torch.cat(
        [x0] + [model.decoder(z) for z in z_preds], dim=0
    ).numpy()                                        # (T+1, n_x)

x_true = X_te[0]                                    # (T+1, n_x) normalized
T_eval = min(len(x_hat), len(x_true))
x_hat  = x_hat[:T_eval]
x_true = x_true[:T_eval]

err = (np.linalg.norm(x_hat - x_true, axis=1) /
       (np.linalg.norm(x_true, axis=1) + 1e-8))

print(f'Mean relative error: {err.mean():.4f}  |  Elapsed: {elapsed:.1f} s')

# ── Save per-run metrics ───────────────────────────────────────────────────────
if args.out_dir:
    savemat(os.path.join(args.out_dir, 'metrics.mat'), {
        'test_err_mean': np.array([err.mean()], dtype=np.float32),
        'test_err_ts':   err.astype(np.float32),
        'elapsed_time':  np.array([elapsed],    dtype=np.float32),
        **{k: np.array(v, dtype=np.float32) for k, v in history.items()}
    })
    print(f'Metrics saved → {args.out_dir}/metrics.mat')

# ── Plots ─────────────────────────────────────────────────────────────────────
if not args.no_plot:
    x_hat_phys  = denormalize(x_hat,  scale['x_lo'], scale['x_rng'])
    x_true_phys = denormalize(x_true, scale['x_lo'], scale['x_rng'])

    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)

    axes[0].plot(x_true_phys[:, 0], label='true')
    axes[0].plot(x_hat_phys[:, 0], '--', label='LRAN')
    axes[0].set_ylabel('θ (rad)'); axes[0].legend()
    axes[0].set_title("θ'' = -(g/l)sinθ + u")

    axes[1].plot(x_true_phys[:, 1], label='true')
    axes[1].plot(x_hat_phys[:, 1], '--', label='LRAN')
    axes[1].set_ylabel("θ' (rad/s)"); axes[1].legend()

    axes[2].semilogy(err)
    axes[2].set_ylabel('relative error'); axes[2].set_xlabel('time step')

    plt.tight_layout()
    plt.savefig('lran_prediction.png', dpi=150)
    plt.show()
