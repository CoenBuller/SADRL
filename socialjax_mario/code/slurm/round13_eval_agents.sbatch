#!/bin/bash
# Round 13 (2026-11-05), L.1: per-agent-id behaviour of the saved MAPPO grid and CC stage-2 actors (evaluation only).
# Submit via: ./alice.sh submit slurm/round13_eval_agents.sbatch
#SBATCH --job-name=sj-r13-eval-agents
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
module load "${PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}"
source "$HOME/${VENV_DIR:-socialjax/venv}/bin/activate"
cd "$HOME/socialjax"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu   # cpu is needed for jax.debug.callback
export WANDB_DIR="$PWD" WANDB_CACHE_DIR="$PWD/.cache/wandb" WANDB_CONFIG_DIR="$PWD/.config/wandb"
echo "Job $SLURM_JOB_ID on $(hostname) | started $(date -Is)"

python -u research/mappo_grid/eval_agents.py 10
echo "Finished $(date -Is)"
