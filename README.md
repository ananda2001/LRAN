# LRAN : Linearly Recurrent Autoencoder for Control Systems

LRAN learns an embedding of a nonlinear control system's state space in which the dynamics are **exactly linear time invariant (LTI)**:

```
z_{k+1} = A z_k + B u_k
```

A neural encoder maps states `x` to a latent vector `z`; a decoder maps back to original state space. The matrices `A` and `B` are learned jointly with the networks. This makes the latent space amenable to linear control methods (LQR, MPC, etc.).

---

## Files

| File | Purpose |
|---|---|
| `model.py` | Encoder, Decoder, and LRAN model with `rollout()` |
| `train.py` | Training loop (reconstruction + forward + linearity losses) |
| `read_dataset.py` | Data loading, normalization, windowing |
| `driver.py` | Main script: load data, train, evaluate, save |
| `create_script.py` | Generates SLURM job scripts for a hyperparameter sweep |
| `check_results.py` | Aggregates hyperparameter sweep results and saves best model parameters |

---

## Bringing Your Own Data

Your data should be a `.mat` file with two variables:

- **`X`**: state snapshots, shape `(n_traj, T, n_x)` or `(T, n_x)` for a single trajectory
- **`U`**: control inputs, shape `(n_traj, T-1, n_u)` or `(T-1, n_u)`

`T` is the number of state snapshots per trajectory; there are `T-1` control steps between them. LRAN normalizes all features to `[-1, 1]` internally for easier training (can be disabled).

If the`.mat` file uses different variable names, pass `--x_key` and `--u_key`.

---

## Quick Start

**Built-in pendulum (for testing):**
```bash
python driver.py --dataset pendulum --epochs 500 --n_z 8
```

**Your own dataset:**
```bash
python driver.py --dataset /path/to/data.mat --x_key X --u_key U \
    --n_z 16 --alpha 4 --steps 8 --epochs 500 --out_dir results/run1
```

Key arguments:

| Argument and Description |
|---|
| `--n_z` Latent (Koopman) dimension |
| `--alpha` Width multiplier (hidden layer width = 16 * alpha) |
| `--steps` Multi-step prediction horizon during training |
| `--epochs` Training epochs |
| `--lr` Learning rate |
| `--gamma_id` Weight on reconstruction loss |
| `--gamma_fwd` Weight on decoded forward-prediction loss |
| `--gamma_lin` Weight on latent linearity loss |
| `--gamma_eig` Weight on eigenvalue stability penalty (0 = off) |
| `--out_dir` Directory to save `model.pt` and `metrics.mat` |
| `--no_plot` Suppress figures (use on HPC) |

---

## Hyperparameter Sweep (SLURM / OSC)

1. **Edit `create_script.py`**: set `DATASET`, `GRID`, `SEEDS`, `FIXED`, and SLURM settings (`ACCOUNT`, `CONDA`, etc.).
2. **Generate job scripts:**
   ```bash
   python create_script.py
   ```
   This writes one `.sh` file per hyperparameter-seed combination and a `submit_batch1.sh` (or more if needed).
3. **Submit on the cluster:**
   ```bash
   sbatch submit_batch1.sh
   ```
   You will receive one email when the submission batch starts/ends/fails. Individual jobs run silently.
4. **Aggregate results:**
   ```bash
   python check_results.py
   ```
   Outputs:
   - `lran_sweep_results.txt`: all combinations ranked by mean test error
   - `best_results/avg_metrics.mat`: loss and error curves averaged across seeds for the best combo
   - `best_results/seed{i}.mat`: per-seed `A`, `B`, encoder/decoder weights, and test errors

---

## Outputs

Each run with `--out_dir` produces:

- **`model.pt`**: full checkpoint (state dict, args, normalization scale)
- **`metrics.mat`**: training loss curves (`loss`, `loss_id`, `loss_fwd`, `loss_lin`, `loss_eig`), test error time series (`test_err_ts`), mean test error (`test_err_mean`), and wall time

The `.mat` files are loadable in MATLAB and Python (`scipy.io.loadmat`).

---

## Training Losses

LRAN is trained with three losses applied over a window of `K+1` consecutive states `{x_t, ..., x_{t+K}}` and controls `{u_t, ..., u_{t+K-1}}`:

- **L_id**: encode then decode each state (autoencoder reconstruction)
- **L_fwd**: roll out the latent dynamics from `z_t` and decode; compare to true states
- **L_lin**: roll out the latent dynamics from `z_t`; compare to encoded true states

Optionally, `L_eig` penalizes eigenvalues of `A` outside the unit circle to encourage stability.
