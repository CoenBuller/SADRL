#!/bin/bash
#SBATCH --job-name=mappo_vs_rnn
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=gpu-l4-24g
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=23G
#SBATCH --array=0-5   # 6 tasks: 3 variants x 2 seeds (30, 31)
# NOTE: run `mkdir -p slurm_logs` BEFORE sbatch; Slurm opens the log files before this script starts.

set -e
mkdir -p slurm_logs

module purge
source /zfsstore/user/s2631415/SADRL/.venv/bin/activate
unset LD_LIBRARY_PATH
export PYTHONPATH=/zfsstore/user/s2631415/SocialJax:$PYTHONPATH
cd /zfsstore/user/s2631415/SADRL

export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

# variant = "<ALGO> <PREDICT_POS>"
VARIANTS=("MAPPO false" "RecurrentMAPPO false" "RecurrentMAPPO true")
read -r ALGO POS <<< "${VARIANTS[$((SLURM_ARRAY_TASK_ID % 3))]}"
SEED=$((30 + SLURM_ARRAY_TASK_ID / 3))

echo "Running Algorithm: $ALGO | predict_pos: $POS | Environment: clean_up | Seed: $SEED"
python main.py --config experiments/recurrent.yaml --algo "$ALGO" --predict_pos "$POS" --env clean_up --seed "$SEED"
