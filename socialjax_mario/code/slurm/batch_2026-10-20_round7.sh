#!/usr/bin/env bash
# Batch 2026-10-20 (branch dir4-cc-mappo), workbook round 7. Clean Up only.
# Run from the repo root on the laptop:  bash slurm/batch_2026-10-20_round7.sh stage1
# Each stage needs Mario's OK before it is submitted. Order (round 7, D.4): hyperparameter comparison, diagnostic,
# CC-MAPPO, everything else. Seeds: at most 3 for the main comparison, 1 for the small IPPO studies (D.1).
set -euo pipefail
cd "$(dirname "$0")/.."

# Shared IPPO/MAPPO setup (round 7, A.1-A.2): IPPO's torso, 32 envs x 1000 steps, 500 minibatches x 2 epochs, LR 5e-4, 1e8.
# IPPO already runs with these settings; MAPPO gets them on the command line.
SHARED="NETWORK_SIZE=full NUM_ENVS=32 NUM_STEPS=1000 NUM_MINIBATCHES=500 LR=0.0005 TOTAL_TIMESTEPS=1e8"
# PPO clip for MAPPO, both compared (A.3): 0.2 / 7 (SocialJax) and 0.2.
CLIP_SCALED="SCALE_CLIP_EPS=True"
CLIP_02="SCALE_CLIP_EPS=False"
# MAPPO has no reward group: the team-IA env flags go in as +ENV_KWARGS (as in the 2026-10-19 batch).
MIA="+ENV_KWARGS.inequity_aversion=True +ENV_KWARGS.enable_smooth_rewards=True +ENV_KWARGS.inequity_aversion_team_reward=True +ENV_KWARGS.inequity_aversion_normalize_trace=True +ENV_KWARGS.inequity_aversion_trace_view=spatial +ENV_KWARGS.inequity_aversion_alpha=0.0"
HIDDEN_B05="+ENV_KWARGS.inequity_aversion_observe_traces=False +ENV_KWARGS.inequity_aversion_beta=0.5"
# MAPPO at 32 x 1000 needs ~6-7 GB after the world-state fix; keep it off the 11 GB 2080 Ti.
MAPPO_PART="--partition=gpu-l4-24g,gpu-mig-40g --time=12:00:00"
SHORT="--time=04:00:00"

sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

stage1() {
  # Approved by Mario 2026-10-20, trimmed to 8 jobs to save ALICE time. Main clip = 0.2/7 (A.3), 0.2 as a 1-seed check.
  # Hyperparameter comparison: IPPO common vs MAPPO common on the shared setup, 3 seeds.
  # IPPO common seed 30 exists (0y756h7t, same settings).
  for seed in 31 32; do
    sub --job-name=sj-ippo-cleanup-common-s$seed $SHORT slurm/train.sbatch --algo IPPO --env cleanup NUM_ENVS=32 TOTAL_TIMESTEPS=1e8 SEED=$seed  # same command as 0y756h7t
  done
  for seed in 30 31 32; do
    sub --job-name=sj-mappo-cleanup-common-shared-clip7-s$seed $MAPPO_PART slurm/train.sbatch --algo MAPPO --env cleanup $SHARED $CLIP_SCALED REWARD=MAPPO_common_shared_clip7 SEED=$seed
  done
  sub --job-name=sj-mappo-cleanup-common-shared-clip2-s30 $MAPPO_PART slurm/train.sbatch --algo MAPPO --env cleanup $SHARED $CLIP_02 REWARD=MAPPO_common_shared_clip02 SEED=30
  # Diagnostic (round 6, R.2): MAPPO 1+3, beta 0.5 hidden, on the shared setup, 2 seeds.
  for seed in 30 31; do
    sub --job-name=sj-mappo-cleanup-dualia-shared-clip7-s$seed $MAPPO_PART slurm/train.sbatch --algo MAPPO --env dual_cleanup $SHARED $CLIP_SCALED $MIA $HIDDEN_B05 DUAL_IND_REWARD=ia REWARD=MAPPO_dual_ia_b05h_shared_clip7 SEED=$seed
  done
}

stage2() {
  # CC-MAPPO, 4 credit modes x 3 seeds, on the shared setup. CLIP is chosen from stage 1 (A.3 note):
  #   CLIP="SCALE_CLIP_EPS=False" bash slurm/batch_2026-10-20_round7.sh stage2
  : "${CLIP:?set CLIP to SCALE_CLIP_EPS=True or SCALE_CLIP_EPS=False}"
  for mode in coma qret loo local; do
    for seed in 30 31 32; do
      sub --job-name=sj-mappo-cleanup-cc-$mode-s$seed $MAPPO_PART slurm/train.sbatch --algo MAPPO --env cc_cleanup $SHARED $CLIP CREDIT_MODE=$mode REWARD=CC_$mode SEED=$seed
    done
  done
}

case "${1:-}" in
  stage1) stage1 ;;
  stage2) stage2 ;;
  *) echo "usage: $0 stage1|stage2"; exit 1 ;;
esac
./alice.sh status
