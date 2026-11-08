#!/bin/bash
#SBATCH --job-name=rec_mappo
#SBATCH --output=%x_%j.out
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=0-11

module purge
source /zfsstore/user/s2631415/SADRL/.venv/bin/activate
unset LD_LIBRARY_PATH
export PYTHONPATH=/zfsstore/user/s2631415/SocialJax:$PYTHONPATH
cd /zfsstore/user/s2631415/SADRL

# Prevent JAX from locking the entire GPU memory on startup
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

nvidia-smi --query-gpu=name,memory.total --format=csv
MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

# Fix 1: Space-separated array
environments=(pd_arena clean_up)

# Fix 2: Modulo 2 for a 2-element array (0 or 1)
env=${environments[$((SLURM_ARRAY_TASK_ID % 2))]}

# Divides by 2 to increment the seed every 2 tasks
SEED=$((30 + SLURM_ARRAY_TASK_ID / 2))

# Fix 3: Pass your YAML config so it doesn't default to IPPO
python main.py --config experiments/RecurrentMAPPO.yaml --env $env --seed $SEED
