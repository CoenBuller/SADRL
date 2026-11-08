#!/bin/bash
# Round 15 (2026-11-08), T.1: swap evaluation. Round-15 actors and imported-env references, each evaluated in both environments (self_channel_fix False and True), 20 episodes each. Approved by Mario (round 15, P.1).
# Submit via: ./alice.sh submit slurm/round15_eval_swap.sbatch
#SBATCH --job-name=sj-r15-eval-swap
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
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

python -u research/mappo_grid/eval_agents.py 20 --swap
echo "Finished $(date -Is)"
