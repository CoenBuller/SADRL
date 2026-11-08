#!/bin/bash
#SBATCH --job-name=mappo_vs_rnn
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --array=0-5   # 2 algos (MAPPO, RecurrentMAPPO) x 3 seeds (30, 31, 32)

set -e
mkdir -p slurm_logs

module purge
source /zfsstore/user/s2631415/SADRL/.venv/bin/activate
unset LD_LIBRARY_PATH
export PYTHONPATH=/zfsstore/user/s2631415/SocialJax:$PYTHONPATH
cd /zfsstore/user/s2631415/SADRL

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

ALGOS=(MAPPO RecurrentMAPPO)
ALGO=${ALGOS[$((SLURM_ARRAY_TASK_ID % 2))]}
SEED=$((30 + SLURM_ARRAY_TASK_ID / 2))

python main.py --config experiments/recurrent.yaml --algo $ALGO --env clean_up --seed $SEED