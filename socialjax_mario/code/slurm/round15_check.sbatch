#!/bin/bash
# Round 15 (2026-11-06): tests and 2-update smoke runs before the fixed-environment gate runs.
# Submit via: ./alice.sh submit slurm/round15_check.sbatch
#SBATCH --job-name=sj-r15-check
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:50:00
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

echo "=== 1. self channel: every agent sees itself in all 7 envs; self_channel_fix=False gives back the imported observation"
python -u tests/test_self_channel.py
echo "=== 2. EV gate: algebra, state, configs, 2-update CC loo with and without the gate"
python -u tests/test_round15_gate.py
echo "=== 3. smoke via train.py (W&B offline, 2 updates), fixed environment"
python -u algorithms/train.py --algo MAPPO --env cleanup CNN_SIZE=small FC_SIZE=full NUM_ENVS=2 NUM_STEPS=100 \
    NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline ENV_KWARGS.self_channel_fix=True REWARD=r15_smoke_mappo
python -u algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_NORM=team CREDIT_GATE=ev CNN_SIZE=small FC_SIZE=full \
    NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline \
    ENV_KWARGS.self_channel_fix=True REWARD=r15_smoke_loo_gate
echo "ALL CHECKS PASSED $(date -Is)"
