#!/bin/bash
# SocialJax training job: runs `python -u algorithms/train.py <args>` inside ~/socialjax.
# Submit via:  ./alice.sh submit [sbatch options] slurm/train.sbatch --algo IPPO --env coins [HYDRA_OVERRIDES...]
# Partitions (see `sinfo -s`): gpu-short (4h), gpu-l4-24g, gpu-2080ti-11g, gpu-mig-40g, gpu-a100-80g (7 days)
#SBATCH --job-name=sj-train
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail

# PYTHON_MODULE and VENV_DIR are exported by alice.sh; the defaults are for submitting by hand.
module load "${PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}"
source "$HOME/${VENV_DIR:-socialjax/venv}/bin/activate"

cd "$HOME/socialjax"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# Allocate GPU memory on demand instead of grabbing 75% up front, so the log's peak-memory
# line reflects what the run actually needs.
export XLA_PYTHON_CLIENT_PREALLOCATE=false
# Keep wandb's files inside this project, not in ~/.cache or ~/.config.
export WANDB_DIR="$PWD" WANDB_CACHE_DIR="$PWD/.cache/wandb" WANDB_CONFIG_DIR="$PWD/.config/wandb"
# The W&B key is stored per project on ALICE (never in git, never in ~/.netrc).
if [ -f .secrets/wandb_api_key ]; then export WANDB_API_KEY="$(cat .secrets/wandb_api_key)"; fi

echo "Job $SLURM_JOB_ID on $(hostname) | partition $SLURM_JOB_PARTITION | started $(date -Is)"
echo "Args: $*"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python -u algorithms/train.py "$@"

echo "Finished $(date -Is)"
sstat -j "$SLURM_JOB_ID.batch" --format=MaxRSS,AveCPU 2>/dev/null || true
nvidia-smi --query-gpu=memory.used --format=csv,noheader
