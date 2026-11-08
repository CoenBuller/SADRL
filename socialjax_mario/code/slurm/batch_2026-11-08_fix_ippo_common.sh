#!/usr/bin/env bash
# Batch 2026-11-08 (branch round15-envfix-gate): IPPO common reward on the FIXED environment
# (ENV_KWARGS.self_channel_fix=True), seeds 30-32, 1e8. Same command as the earlier IPPO common runs on the imported
# environment (0y756h7t, v0y2reh1, e7j7u2co: 32 envs x 1000 steps, 137 / 1054 / 824), so the only change is the
# environment. Question: does IPPO (separate critics, no centralised value) also stop near 900 on the fixed env?
# REWARD names start with FIX_. Submission requested by Mario in chat, 2026-11-08.
set -euo pipefail
cd "$(dirname "$0")/.."

sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  sub --job-name=sj-fix-ippo-common-s$seed --time=04:00:00 slurm/train.sbatch --algo IPPO --env cleanup \
      NUM_ENVS=32 TOTAL_TIMESTEPS=1e8 +ENV_KWARGS.self_channel_fix=True SEED=$seed REWARD=FIX_IPPO_common
done
./alice.sh status
