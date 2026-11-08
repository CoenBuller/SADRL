#!/bin/bash
#SBATCH --job-name=cleanup_ablations
#SBATCH --output=slurm_logs/%x_%A_%a.out
#SBATCH --error=slurm_logs/%x_%A_%a.err
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:a100:1
#SBATCH --time=12:00:00
#SBATCH --mem=32G
#SBATCH --array=0-7

# Exit immediately if a command exits with a non-zero status
set -e

# Create log directory
mkdir -p slurm_logs

# Load environment
module purge
source /zfsstore/user/s2631415/SADRL/.venv/bin/activate
unset LD_LIBRARY_PATH
export PYTHONPATH=/zfsstore/user/s2631415/SocialJax:$PYTHONPATH
cd /zfsstore/user/s2631415/SADRL

# Prevent JAX from locking the entire GPU memory on startup
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONUNBUFFERED=1

echo "=================================================="
echo "Starting Ablation Sweep on node: $(hostname)"
echo "CUDA visible devices: $CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,memory.total --format=csv
echo "=================================================="

# 4 Algorithms
ALGOS=(MAPPO RecurrentMAPPO COMA RecurrentCOMA)

# Calculate which algorithm and seed to use for this specific array task
ALGO_INDEX=$((SLURM_ARRAY_TASK_ID % 4))
ALGO=${ALGOS[$ALGO_INDEX]}

# Calculate seed: every 4 tasks, the seed increments by 1
# Tasks 0,1,2,3 -> Seed 30
# Tasks 4,5,6,7 -> Seed 31
SEED=$((30 + SLURM_ARRAY_TASK_ID / 4))

echo "Running Configuration:"
echo "  Algorithm:   $ALGO"
echo "  Environment: clean_up"
echo "  Seed:        $SEED"
echo "=================================================="

# Run the training script with the YAML config and CLI overrides
python main.py --config experiment_coma.yaml --algo $ALGO --env clean_up --seed $SEED

echo "=================================================="
echo "Task completed successfully!"
echo "=================================================="