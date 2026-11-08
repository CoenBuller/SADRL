#!/bin/bash
# Round 16b (2026-11-08): tests and smoke runs before the single masked critic runs (CREDIT_MODE=loo_single). Testing only on ALICE.
# Submit via: ./alice.sh submit slurm/round16b_check.sbatch
#SBATCH --job-name=sj-r16b-check
#SBATCH --partition=gpu-short
#SBATCH --gres=gpu:l4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out

set -eo pipefail
module load "${PYTHON_MODULE:-Python/3.10.8-GCCcore-12.2.0}"
source "$HOME/${VENV_DIR:-socialjax/venv}/bin/activate"
cd "$HOME/socialjax"
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cuda,cpu
export WANDB_DIR="$PWD" WANDB_CACHE_DIR="$PWD/.cache/wandb" WANDB_CONFIG_DIR="$PWD/.config/wandb"
echo "Job $SLURM_JOB_ID on $(hostname) | started $(date -Is)"

echo "=== 1. single masked critic: slots, masks (V(s) ignores all actions, V(s, a^-i) ignores a^i only), credit, 2-update training with and without the agent id"
python -u tests/test_round16_single_critic.py
echo "=== 2. regression: the other credit modes still train and log (coma, qret, loo, local), agent-id tests (fast part)"
python -u tests/test_cc_logging.py
python -u tests/test_round16_agent_id.py --fast
echo "=== 3. smoke via train.py (W&B offline, 2 updates, end-of-run evaluation and GIF): imported env, and fixed env + agent id"
python -u algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=loo_single CREDIT_NORM=team CNN_SIZE=small FC_SIZE=full \
    NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline \
    ENV_KWARGS.self_channel_fix=False REWARD=r16b_smoke_single_oldenv
python -u algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=loo_single CREDIT_NORM=team CNN_SIZE=small FC_SIZE=full AGENT_ID=True \
    NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline \
    ENV_KWARGS.self_channel_fix=True REWARD=r16b_smoke_single_aid
echo "ALL CHECKS PASSED $(date -Is)"
