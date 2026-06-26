"""
LRAN hyperparameter sweep — job script generator for OSC (SLURM).

EDIT THIS FILE to configure your sweep:
  DATASET, GRID, SEEDS, FIXED

Then run:
    python create_script.py       # generates .sh files + submit_batch*.sh
    bash submit_batch1.sh         # on the cluster login node

check_results.py imports GRID, SEEDS, and result_dir from here automatically.
"""

import itertools
import textwrap

# Dataset
DATASET = 'pendulum'    # 'pendulum'  OR  path to a .mat file
# X_KEY  = 'X'         # uncomment if using a .mat file
# U_KEY  = 'U'

# Hyperparameter grid (all combinations are swept
GRID = {
    'n_z':        [8, 12, 16],
    'alpha':      [2, 4],
    'steps':      [8, 16],
    'lr':         [1e-3, 1e-2],
    'wd':         [1e-4, 1e-3],
    'batch_size': [64, 128],
    'gamma_id':   [1.0],           # expand to e.g. [0.5, 1.0] if desired
    'gamma_fwd':  [1.0, 2.0],
    'gamma_lin':  [1.0, 2.0],
    'gamma_eig':  [0.0, 1.0],
}
SEEDS = list(range(10))

# Fixed hyperparameters (held constant across all jobs)
FIXED = {
    'epochs':     500,
    'gradclip':   0.05,
    'n_traj':     100,
    'traj_len':   200,
    'train_frac': 0.8,
}

# SLURM configuration 
ACCOUNT        = 'PAS2709'
EMAIL          = 'chakrabarti.44@osu.edu'
CONDA          = 'demo'
JOB_TIME       = '2:00:00'
NODES          = 1
TASKS_PER_NODE = 2
MEM            = '16G'
JOBS_PER_BATCH = 990


#  Helpers (module-level so check_results.py can import them)

def _fmt(v):
    """Format a value for use in directory names and shell args."""
    if isinstance(v, float) and v != 0 and (abs(v) < 1 or abs(v) >= 1e4):
        return f'{v:.0e}'
    return str(v)


def all_combos():
    keys = list(GRID.keys())
    return [dict(zip(keys, vals)) for vals in itertools.product(*GRID.values())]


def _tag(p, seed):
    return (f'nz{p["n_z"]}_a{p["alpha"]}_K{p["steps"]}'
            f'_lr{_fmt(p["lr"])}_wd{_fmt(p["wd"])}_bs{p["batch_size"]}'
            f'_gid{p["gamma_id"]}_gfwd{p["gamma_fwd"]}_glin{p["gamma_lin"]}'
            f'_geig{p["gamma_eig"]}_seed{seed}')


def result_dir(p, seed):
    return f'results_lran_{_tag(p, seed)}'


def _slurm_header(job_name):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={job_name}
        #SBATCH --time={JOB_TIME}
        #SBATCH --nodes={NODES} --ntasks-per-node={TASKS_PER_NODE} --mem={MEM}
        #SBATCH --account={ACCOUNT}

        source ~/.bashrc
        conda activate {CONDA}
        cd $SLURM_SUBMIT_DIR

        """)


def _submit_header(batch_name):
    return textwrap.dedent(f"""\
        #!/bin/bash
        #SBATCH --job-name={batch_name}
        #SBATCH --time=0:30:00
        #SBATCH --nodes=1 --ntasks-per-node=1 --mem=2G
        #SBATCH --account={ACCOUNT}
        #SBATCH --mail-type=BEGIN,END,FAIL
        #SBATCH --mail-user={EMAIL}

        cd $SLURM_SUBMIT_DIR

        """)


def _fixed_arg_str():
    return ' '.join(f'--{k} {v}' for k, v in FIXED.items())


# Job generation 

if __name__ == '__main__':
    job_files = []

    for p in all_combos():
        for seed in SEEDS:
            out   = result_dir(p, seed)
            name  = f'lran_{_tag(p, seed)}'
            fname = f'{name}.sh'

            sweep_args = (
                f'--dataset {DATASET} '
                f'--n_z {p["n_z"]} --alpha {p["alpha"]} --steps {p["steps"]} '
                f'--lr {_fmt(p["lr"])} --wd {_fmt(p["wd"])} --batch_size {p["batch_size"]} '
                f'--gamma_id {p["gamma_id"]} --gamma_fwd {p["gamma_fwd"]} '
                f'--gamma_lin {p["gamma_lin"]} --gamma_eig {p["gamma_eig"]} '
                f'--seed {seed} --out_dir {out} --no_plot'
            )

            with open(fname, 'w') as fh:
                fh.write(_slurm_header(name))
                fh.write(f'python -u driver.py {sweep_args} {_fixed_arg_str()}\n')

            job_files.append(fname)

    n_batches = (len(job_files) + JOBS_PER_BATCH - 1) // JOBS_PER_BATCH
    for b in range(n_batches):
        batch = job_files[b * JOBS_PER_BATCH : (b + 1) * JOBS_PER_BATCH]
        bname = f'submit_batch{b + 1}'
        with open(f'{bname}.sh', 'w') as fh:
            fh.write(_submit_header(bname))
            for jf in batch:
                fh.write(f'dos2unix {jf}\n')
                fh.write(f'sbatch {jf}\n')

    print(f'Created {len(job_files)} job scripts-> {n_batches} submission batch(es).')
    print('On the cluster: sbatch submit_batch1.sh')
