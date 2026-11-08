#!/usr/bin/env bash
# Batch 2026-11-08 (branch round15-envfix-gate): IPPO common reward, 4 envs x 100 steps, on the FIXED environment
# (ENV_KWARGS.self_channel_fix=True), seeds 30-32, 1e8. Same rollout as the round-15 MAPPO / CC runs (4 x 100,
# 4 minibatches of 700 samples, 2 epochs, LR 5e-4), and the same command as round 9's IPPO 4 x 100 (r4dxsl97,
# qaz1or7y, fny0nddj; imported env, 30M only) except the environment and the 1e8 budget.
# IPPO 4 x 100 ran 1 h 44 min per 3e7 on gpu-short, so 1e8 needs about 6 h: L4 partition.
# Requested by Mario in chat, 2026-11-08. REWARD names start with FIX_.
set -euo pipefail
cd "$(dirname "$0")/.."

sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  sub --job-name=sj-fix-ippo-4x100-s$seed --partition=gpu-l4-24g --time=10:00:00 slurm/train.sbatch --algo IPPO --env cleanup \
      NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 TOTAL_TIMESTEPS=1e8 +ENV_KWARGS.self_channel_fix=True \
      SEED=$seed REWARD=FIX_IPPO_common_4x100
done
./alice.sh status
