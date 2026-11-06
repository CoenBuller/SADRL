#!/usr/bin/env bash
# Batch 2026-10-21 (branch dir4-cc-mappo), workbook round 9. Clean Up only.
# Run from the repo root on the laptop:  bash slurm/batch_2026-10-21_round9.sh <stage>
# The MAPPO screen found that only M-c's rollout learns (m3248dz4: 2561 at 30M). Round 9, S.2: MAPPO and CC-MAPPO use
# M-c's setup from now on: 4 envs x 100 steps, 4 minibatches x 2 epochs, full nets, LR 5e-4, clip 0.2/7.
set -euo pipefail
cd "$(dirname "$0")/.."

MC="NETWORK_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 UPDATE_EPOCHS=2 LR=0.0005 SCALE_CLIP_EPS=True"
SHORT="--time=04:00:00"
L4_12H="--partition=gpu-l4-24g --time=12:00:00"
sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

ippo4x100() {
  # Approved by Mario 2026-10-21 (round 9, I.1/I.2): IPPO common on M-c's rollout, seeds 30-32, 30M, gpu-short.
  # IPPO's own config otherwise (2 epochs, LR 5e-4, clip 0.2 unscaled).
  for seed in 30 31 32; do
    sub --job-name=sj-ippo-4x100-s$seed $SHORT slurm/train.sbatch --algo IPPO --env cleanup NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 TOTAL_TIMESTEPS=3e7 SEED=$seed REWARD=common_4x100
  done
}

cctest() {
  # Round 9, C.2: the CC logging test (commit 1944a24), then a speed run per credit mode and one for plain MAPPO on the
  # stage-2 partition, so each mode's time per 100M is known before stage 2. 2M steps each, logged to W&B.
  sub --job-name=sj-test-cclog --time=00:45:00 --mem=16G slurm/python.sbatch tests/test_cc_logging.py
  sub --job-name=sj-speed-mappo --partition=gpu-l4-24g --time=00:40:00 slurm/train.sbatch --algo MAPPO --env cleanup $MC TOTAL_TIMESTEPS=2e6 SEED=30 REWARD=speed_MAPPO_mc GIF_NUM_FRAMES=3
  for mode in coma qret loo local; do
    sub --job-name=sj-speed-cc-$mode --partition=gpu-l4-24g --time=00:40:00 slurm/train.sbatch --algo MAPPO --env cc_cleanup $MC CREDIT_MODE=$mode TOTAL_TIMESTEPS=2e6 SEED=30 REWARD=speed_CC_$mode GIF_NUM_FRAMES=3
  done
}

deadtest() {
  # Equivalence test of the dead-actor logging (parameters identical to the pre-fix copy after 2 updates).
  sub --job-name=sj-test-deadlog --time=00:30:00 --mem=16G slurm/python.sbatch tests/test_mappo_dead_actor_logging.py
}

deadcheck() {
  # Approved by Mario 2026-10-21 (round 9, D.2): M-0 (shared setup) and M-c with the dead-actor logging, seed 30, 5M.
  local shared="NETWORK_SIZE=full NUM_ENVS=32 NUM_STEPS=1000 NUM_MINIBATCHES=500 UPDATE_EPOCHS=2 LR=0.0005 SCALE_CLIP_EPS=True"
  sub --job-name=sj-dead-m0 $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $shared LOG_ACTOR_HEALTH=True TOTAL_TIMESTEPS=5e6 SEED=30 REWARD=MAPPO_dead_M0
  sub --job-name=sj-dead-mc $SHORT slurm/train.sbatch --algo MAPPO --env cleanup $MC LOG_ACTOR_HEALTH=True TOTAL_TIMESTEPS=5e6 SEED=30 REWARD=MAPPO_dead_Mc
}

cccheck() {
  # CC theory T.4 (2026-10-21): CREDIT_NORM=team. Same 2M speed-run setup as cctest, to see that coma and local no longer
  # collapse (entropy, clean_fraction) before stage 2.
  for mode in coma qret loo local; do
    sub --job-name=sj-cccheck-$mode --partition=gpu-l4-24g --time=00:40:00 slurm/train.sbatch --algo MAPPO --env cc_cleanup $MC CREDIT_MODE=$mode CREDIT_NORM=team TOTAL_TIMESTEPS=2e6 SEED=30 REWARD=cccheck_${mode}_teamnorm GIF_NUM_FRAMES=3
  done
}

stage2_qret_loo() {
  # Approved by Mario 2026-10-21 (chat, after the cccheck runs): coma and local still collapse with CREDIT_NORM=team,
  # so stage 2 runs only qret and loo (with the round-7 normalisation, CREDIT_NORM=self) and the MAPPO common baseline.
  # 9 jobs, 90M, M-c's setup, gpu-l4-24g, 12 h. coma and local wait for a warm-up designed with Mario.
  for seed in 30 31 32; do
    sub --job-name=sj-mappo-mc-common-s$seed $L4_12H slurm/train.sbatch --algo MAPPO --env cleanup $MC TOTAL_TIMESTEPS=9e7 SEED=$seed REWARD=MAPPO_common_mc
  done
  for mode in qret loo; do
    for seed in 30 31 32; do
      sub --job-name=sj-cc-$mode-s$seed $L4_12H slurm/train.sbatch --algo MAPPO --env cc_cleanup $MC CREDIT_MODE=$mode CREDIT_NORM=self TOTAL_TIMESTEPS=9e7 SEED=$seed REWARD=CC_${mode}_mc
    done
  done
}

stage2() {
  # Approved by Mario 2026-10-21 (round 9, C.1-C.3): CC-MAPPO 4 modes x seeds 30-32 and MAPPO common seeds 30-32 as the
  # baseline, on M-c's setup, gpu-l4-24g, 12 h. 90M steps (Mario, chat): the speed runs give 10.1-11.8 h per 100M for CC.
  for seed in 30 31 32; do
    sub --job-name=sj-mappo-mc-common-s$seed $L4_12H slurm/train.sbatch --algo MAPPO --env cleanup $MC TOTAL_TIMESTEPS=9e7 SEED=$seed REWARD=MAPPO_common_mc
  done
  for mode in coma qret loo local; do
    for seed in 30 31 32; do
      sub --job-name=sj-cc-$mode-s$seed $L4_12H slurm/train.sbatch --algo MAPPO --env cc_cleanup $MC CREDIT_MODE=$mode CREDIT_NORM=team TOTAL_TIMESTEPS=9e7 SEED=$seed REWARD=CC_${mode}_mc
    done
  done
}

case "${1:-}" in
  ippo4x100) ippo4x100 ;;
  cctest) cctest ;;
  deadtest) deadtest ;;
  deadcheck) deadcheck ;;
  cccheck) cccheck ;;
  stage2) stage2 ;;
  stage2_qret_loo) stage2_qret_loo ;;
  *) echo "usage: $0 ippo4x100|cctest|deadtest|deadcheck|cccheck|stage2|stage2_qret_loo"; exit 1 ;;
esac
./alice.sh status
