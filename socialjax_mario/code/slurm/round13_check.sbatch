#!/bin/bash
# Round 13 (2026-11-05): tests and 2-update smoke runs before the depth and CC-MAPPO runs.
# Submit via: ./alice.sh submit slurm/round13_check.sbatch
#SBATCH --job-name=sj-r13-check
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

echo "=== 1. new test (GPU): CNN_LAYERS, CC networks, new logging"
python -u tests/test_round13_depth_cc.py
echo "=== 2 (skipped in the rerun: passed in job 5454097). CPU: MAPPO small/small parameters identical to the pre-2026-10-20 reference after the logging change"
# JAX_PLATFORMS=cpu python -u tests/test_mappo_shared_setup.py  # passed in 5454097
echo "=== 3. smoke via train.py (W&B offline, 2 updates)"
python -u algorithms/train.py --algo MAPPO --env cleanup CNN_SIZE=small FC_SIZE=full CNN_LAYERS=3 NUM_ENVS=2 NUM_STEPS=100 \
    NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline REWARD=r13_smoke_depth3
for mode in qret loo; do
  python -u algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=$mode CREDIT_NORM=team CNN_SIZE=small FC_SIZE=full \
      NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline REWARD=r13_smoke_cc_$mode
done
echo "ALL CHECKS PASSED $(date -Is)"
