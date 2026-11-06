#!/bin/bash
# Round 10 (2026-11-03): one short job before the MAPPO network grid. Tests, a GIF smoke run (W&B offline), and a
# speed probe for each CNN_SIZE x FC_SIZE x rollout setting. Submit via: ./alice.sh submit slurm/mappo_grid_check.sbatch
#SBATCH --job-name=sj-mappo-grid-check
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:30:00
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
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== 1a. grid test on GPU: records how far GPU results drift (not a pass/fail check)"
python -u tests/test_mappo_net_grid.py || echo "GPU grid test: not identical (see the diffs above)"
echo "=== 1b. tests on CPU (deterministic): parameters must be identical"
for t in tests/test_mappo_net_grid.py tests/test_mappo_shared_setup.py tests/test_mappo_loss_logging.py tests/test_mappo_dead_actor_logging.py; do
    echo "--- $t"; JAX_PLATFORMS=cpu python -u "$t"
done

echo "=== 2. GIF smoke (W&B offline, 2 updates, small CNN + full FC)"
python -u algorithms/train.py --algo MAPPO --env cleanup CNN_SIZE=small FC_SIZE=full NUM_ENVS=2 NUM_STEPS=100 \
    NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline REWARD=grid_smoke
ls -la evaluation/clean_up/*grid_smoke* 2>/dev/null || ls -la evaluation/clean_up | tail -3

echo "=== 3. speed probes (LR 5e-4, clip 0.2/7)"
for net in "small small" "small full" "full small" "full full"; do
    set -- $net
    python -u research/mappo_grid/speed_probe.py CNN_SIZE=$1 FC_SIZE=$2 LR=0.0005 NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 TOTAL_TIMESTEPS=1.2e6 || echo "SPEED FAILED $*"
    python -u research/mappo_grid/speed_probe.py CNN_SIZE=$1 FC_SIZE=$2 LR=0.0005 NUM_ENVS=32 NUM_STEPS=1000 NUM_MINIBATCHES=500 TOTAL_TIMESTEPS=1.28e6 || echo "SPEED FAILED $*"
done
echo "Finished $(date -Is)"
nvidia-smi --query-gpu=memory.used --format=csv,noheader
