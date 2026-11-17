#!/usr/bin/env bash
# Batch 2026-11-17 (branch round17-envs): round 17, E.1-E.5. MAPPO vs CC-MAPPO loo_single (one masked critic) beyond
# Clean Up, on mushrooms, gift, pd_arena and harvest_common_open (harvest last, E.1), fixed environment, common reward.
#   1. tests/test_round17_envs.py (envs unchanged in training reward; Clean Up trainers bit-identical; 2 updates per new
#      env) and a 2-update smoke run through train.py per env and arm. Every run below waits for all of them (afterok).
#   2. MAPPO probes (E.5), 3e7 steps, seed 30: SocialJax's own per-env config (small/small networks, their LR /
#      NUM_ENVS / NUM_STEPS) against our setup. The better MAPPO decides the setup of the main runs (Mario, 17 Nov).
#   3. Main runs (submitted separately, after the probes: ./slurm/batch_2026-11-17_round17_envs.sh main), our Clean Up setup (E.2): small CNN + full FC, 4 x 100, LR 5e-4, 2 epochs, 1e8, seeds 30-32 (E.3).
# Team return = N x returned_episode_original_returns, with N = num_agents of the env (5, 5, 4, 7).
set -euo pipefail
cd "$(dirname "$0")/.."

OURS="CNN_SIZE=small FC_SIZE=full NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4 LR=0.0005 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=1e8 ENV_KWARGS.self_channel_fix=True"
SMOKE="CNN_SIZE=small FC_SIZE=full NUM_ENVS=2 NUM_STEPS=100 NUM_MINIBATCHES=2 UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=400 WANDB_MODE=disabled GIF_NUM_FRAMES=3"
# SocialJax's mappo_cnn_<env>.yaml on top of mappo_base (LR 0.001, 2 epochs, scaled clip, SocialJax's small networks)
declare -A SJCFG=(
  [mushrooms]="NETWORK_SIZE=small LR=0.001 NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4"
  [gift]="NETWORK_SIZE=small LR=0.001 NUM_ENVS=4 NUM_STEPS=100 NUM_MINIBATCHES=4"
  [pd_arena]="NETWORK_SIZE=small LR=0.001 NUM_ENVS=4 NUM_STEPS=1000 NUM_MINIBATCHES=4"
  [harvest_open]="NETWORK_SIZE=small LR=0.0005 NUM_ENVS=256 NUM_STEPS=100 NUM_MINIBATCHES=4"
)
ENVS="mushrooms gift pd_arena harvest_open"
PART="${1:-probes}"  # probes: parts 1-2; main: part 3 only; retry: see below
jid() {
  local id; id=$(./alice.sh submit "$@" 2>&1 | sed -n 's/^Submitted job \([0-9]*\).*/\1/p')
  [ -n "$id" ] || { echo "submission failed: $*" >&2; exit 1; }
  echo "$id"
}

if [ "$PART" = probes ]; then
DEPS=$(jid --job-name=sj-test-r17 --time=01:30:00 --mem=32G --cpus-per-task=4 slurm/python.sbatch tests/test_round17_envs.py)
for e in $ENVS; do
  DEPS="$DEPS:$(jid --job-name=sj-smoke-$e --time=00:30:00 slurm/train.sbatch --algo MAPPO --env sadrl_$e $SMOKE)"
  DEPS="$DEPS:$(jid --job-name=sj-smoke-cc-$e --time=00:30:00 slurm/train.sbatch --algo MAPPO --env cc_$e $SMOKE)"
done
AFTER="--dependency=afterok:$DEPS"
echo "tests and smoke runs: $DEPS"

fi

# retry (17 Nov, 19:25): the test job (5645736) and 6 smoke runs passed, but sj-smoke-cc-gift and sj-smoke-pd_arena hung
# without output on node863 (A100) until the 30-min limit, so the probes were never released. Rerun those two smoke runs
# away from node863, then the probes after them.
if [ "$PART" = retry ]; then
X="--exclude=node863"
DEPS="$(jid $X --job-name=sj-smoke-cc-gift --time=00:30:00 slurm/train.sbatch --algo MAPPO --env cc_gift $SMOKE)"
DEPS="$DEPS:$(jid $X --job-name=sj-smoke-pd_arena --time=00:30:00 slurm/train.sbatch --algo MAPPO --env sadrl_pd_arena $SMOKE)"
AFTER="--dependency=afterok:$DEPS $X"
echo "smoke reruns: $DEPS"
fi

if [ "$PART" = probes ] || [ "$PART" = retry ]; then
for e in $ENVS; do
  jid $AFTER --job-name=sj-sjcfg-$e --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env sadrl_$e ${SJCFG[$e]} UPDATE_EPOCHS=2 SCALE_CLIP_EPS=True TOTAL_TIMESTEPS=3e7 SEED=30 \
      REWARD=R17_MAPPO_sjcfg_3e7 > /dev/null
  jid $AFTER --job-name=sj-ours-$e --partition=gpu-short --time=03:30:00 slurm/train.sbatch \
      --algo MAPPO --env sadrl_$e ${OURS/TOTAL_TIMESTEPS=1e8/TOTAL_TIMESTEPS=3e7} SEED=30 \
      REWARD=R17_MAPPO_ours_3e7 > /dev/null
done
fi

if [ "$PART" = main ]; then
for e in $ENVS; do  # harvest last
  for seed in 30 31 32; do
    jid --job-name=sj-r17-mappo-$e-s$seed --partition=gpu-l4-24g --time=12:00:00 slurm/train.sbatch \
        --algo MAPPO --env sadrl_$e $OURS SEED=$seed REWARD=R17_MAPPO_csff_4x100 > /dev/null
    jid --job-name=sj-r17-cc-$e-s$seed --partition=gpu-l4-24g --time=16:00:00 slurm/train.sbatch \
        --algo MAPPO --env cc_$e $OURS CREDIT_MODE=loo_single CREDIT_NORM=team CREDIT_GATE=none SEED=$seed \
        REWARD=R17_CC_loo_single_csff_4x100 > /dev/null
  done
done
fi
./alice.sh status
