#!/usr/bin/env bash
# Batch 2026-11-07 (branch round15-envfix-gate): CC-MAPPO loo + EV gate on the IMPORTED environment
# (ENV_KWARGS.self_channel_fix=False, the self-channel off-by-one), seeds 30-32, 1e8.
# Purpose: compare the gate with the round-13 CC loo runs (CC_loo_team, 3214 +- 4, 4 of 6 CC loo runs in rounds 12-13
# never got the first reward) on the same environment, because on the fixed environment every arm plateaus near 900
# and the start-up failures vanished in all arms, so the gate's effect could not be isolated there.
# Same setup as the FIX_CC_loo_gate runs except the environment flag. REWARD names start with OLDENV_.
# Submission approved by Mario in chat, 2026-11-07.
set -euo pipefail
cd "$(dirname "$0")/.."

COMMON="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 ENV_KWARGS.self_channel_fix=False"
sub() { ./alice.sh submit ${DEP:-} "$@" 2>&1 | tail -1; }

for seed in 30 31 32; do
  sub --job-name=sj-old-gate-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
      --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_NORM=team CREDIT_GATE=ev GATE_EFULL=0.5 GATE_BETA=0.9 $COMMON \
      SEED=$seed REWARD=OLDENV_CC_loo_gate_csff_4x100
done
./alice.sh status
