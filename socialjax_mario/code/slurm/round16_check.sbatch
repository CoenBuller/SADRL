#!/bin/bash
# Round 16 (2026-11-08): tests and 2-update smoke runs before the agent-id runs (A.1). Testing only on ALICE (Mario).
# Submit via: ./alice.sh submit slurm/round16_check.sbatch
#SBATCH --job-name=sj-r16-check
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

echo "=== 1. agent id: helper, networks, actor, AGENT_ID=False bit-identical, 2-update MAPPO and CC loo, eval_agents"
JAX_PLATFORMS=cuda,cpu python -u tests/test_round16_agent_id.py
echo "=== 2. regression: round-13 network classes and parameter counts, MAPPO depth and CC logging"
python -u tests/test_round13_depth_cc.py
echo "=== 3. smoke via train.py (W&B offline, 2 updates, end-of-run evaluation and GIF), fixed environment, AGENT_ID=True"
python -u algorithms/train.py --algo MAPPO --env cleanup CNN_SIZE=small FC_SIZE=full AGENT_ID=True NUM_ENVS=2 NUM_STEPS=100 \
    NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline ENV_KWARGS.self_channel_fix=True REWARD=r16_smoke_mappo_aid
python -u algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_NORM=team CNN_SIZE=small FC_SIZE=full AGENT_ID=True \
    NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 TOTAL_TIMESTEPS=400 GIF_NUM_FRAMES=20 WANDB_MODE=offline \
    ENV_KWARGS.self_channel_fix=True REWARD=r16_smoke_loo_aid
echo "ALL CHECKS PASSED $(date -Is)"
