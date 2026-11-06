#!/usr/bin/env bash
# Batch 2026-10-21 (branch dir4-cc-mappo), workbook round 8: the one-change MAPPO screen. Clean Up only.
# Run from the repo root on the laptop:  bash slurm/batch_2026-10-21_round8.sh smoke|screen
# MAPPO common failed on the shared setup (round 8). Each screen run starts from the shared setup and puts back ONE
# old setting. MAPPO common, seed 30, 30M steps (the old run was near 1000 at 15M), clip 0.2/7, gpu-short.
set -euo pipefail
cd "$(dirname "$0")/.."

# Shared setup, one key per setting, so each run overrides it without repeating a key on the command line.
shared() {  # shared [KEY=VALUE ...]: the shared setup with these settings replaced
  local -A cfg=([NETWORK_SIZE]=full [NUM_ENVS]=32 [NUM_STEPS]=1000 [NUM_MINIBATCHES]=500 [LR]=0.0005 [UPDATE_EPOCHS]=2 [SCALE_CLIP_EPS]=True)
  local kv; for kv in "$@"; do cfg[${kv%%=*}]=${kv#*=}; done
  local k out=""; for k in "${!cfg[@]}"; do out+="$k=${cfg[$k]} "; done; echo "$out"
}
SHARED="$(shared)"
SCREEN="TOTAL_TIMESTEPS=3e7 SEED=30"
SHORT="--time=04:00:00"
sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

smoke() {
  # The logging fix's equivalence test, and 3 full-size updates on gpu-short (does the shared setup fit the 2080 Ti?).
  sub --job-name=sj-test-losslog --time=00:30:00 --mem=16G slurm/python.sbatch tests/test_mappo_loss_logging.py
  sub --job-name=sj-full-mappo-short --time=00:45:00 slurm/train.sbatch --algo MAPPO --env cleanup $SHARED TOTAL_TIMESTEPS=96000 WANDB_MODE=disabled GIF_NUM_FRAMES=3
}

screen() {
  # Approved by Mario 2026-10-21 (round 8, M.1 and M.4).
  sub --job-name=sj-mappo-screen-m0 $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $SHARED $SCREEN REWARD=MAPPO_screen_M0_shared
  sub --job-name=sj-mappo-screen-ma $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $(shared NETWORK_SIZE=small) $SCREEN REWARD=MAPPO_screen_Ma_small_nets
  sub --job-name=sj-mappo-screen-mb $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $(shared LR=0.001) $SCREEN REWARD=MAPPO_screen_Mb_lr1e-3
  sub --job-name=sj-mappo-screen-mc $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $(shared NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4) $SCREEN REWARD=MAPPO_screen_Mc_old_rollout
}

epochs() {
  # Approved by Mario 2026-10-21 (M.4 note): does the number of epochs matter? 4 and 8 instead of 2.
  for e in 4 8; do
    sub --job-name=sj-mappo-screen-md-e$e $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $(shared UPDATE_EPOCHS=$e) $SCREEN REWARD=MAPPO_screen_Md_epochs$e
  done
}

clip() {
  # Approved by Mario 2026-10-21, clip size Claude's choice: unscaled 0.3 and 0.5 (0.2 unscaled is cs9pnxl6, ~0 at 30M).
  # CLIP_EPS clips both the policy ratio and the value change per update.
  for c in 0.3 0.5; do
    sub --job-name=sj-mappo-screen-me-c$c $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $(shared SCALE_CLIP_EPS=False CLIP_EPS=$c) $SCREEN REWARD=MAPPO_screen_Me_clip$c
  done
}

ippo_seeds() {
  # Approved by Mario 2026-10-21 (R.2): seeds 31 and 32 for IPPO team + IA (B5, 0uxekzlc) and blend + IA (B6, 6uj9o7aw),
  # same commands as the 2026-10-19 batch.
  for seed in 31 32; do
    sub --job-name=sj-ippo-cleanup-teamia-b05h-s$seed $SHORT slurm/train.sbatch --algo IPPO --env cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=0.5 ENV_KWARGS.inequity_aversion_observe_traces=False REWARD=team_ia_b05_hidden NUM_ENVS=32 TOTAL_TIMESTEPS=1e8 SEED=$seed
    sub --job-name=sj-ippo-cleanup-blend-ia-s$seed $SHORT slurm/train.sbatch --algo IPPO --env dual_cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.4378 DUAL_IND_REWARD=ia DUAL_MODE=blend REWARD=blend_ia NUM_ENVS=32 TOTAL_TIMESTEPS=1e8 SEED=$seed
  done
}

rollout200() {
  # Approved by Mario 2026-10-21 (chat): a shared setup with 5x more updates for BOTH algorithms. 32 envs x 200 steps,
  # 100 minibatches (same 448 agent-steps per minibatch, same ~3.1M gradient steps per 100M), 2 and 4 epochs.
  # Common reward, seed 30, 30M steps like the screen.
  for e in 2 4; do
    sub --job-name=sj-mappo-r200-e$e $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $(shared NUM_STEPS=200 NUM_MINIBATCHES=100 UPDATE_EPOCHS=$e) $SCREEN REWARD=MAPPO_common_r200_e$e
    sub --job-name=sj-ippo-r200-e$e $SHORT slurm/train.sbatch --algo IPPO --env cleanup NUM_ENVS=32 NUM_STEPS=200 NUM_MINIBATCHES=100 UPDATE_EPOCHS=$e $SCREEN REWARD=common_r200_e$e
  done
}

case "${1:-}" in
  smoke) smoke ;;
  screen) screen ;;
  epochs) epochs ;;
  clip) clip ;;
  ippo_seeds) ippo_seeds ;;
  rollout200) rollout200 ;;
  *) echo "usage: $0 smoke|screen|epochs|clip|ippo_seeds|rollout200"; exit 1 ;;
esac
./alice.sh status
