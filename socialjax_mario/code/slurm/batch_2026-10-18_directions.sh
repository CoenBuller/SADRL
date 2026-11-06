#!/usr/bin/env bash
# Overnight batch 2026-10-18 (branch dir3-dual-critic, which contains dir1-team-ia): directions 1, 3 and 1+3.
# Run from the repo root on the laptop: bash slurm/batch_2026-10-18_directions.sh
# All runs are seed 30 on gpu-short. Coins IPPO uses 256 envs (L4 when the trace channel is on); Clean Up IPPO uses 32 envs.
set -euo pipefail
cd "$(dirname "$0")/.."

# IA under the team reward: guilt only (round 2, 1.A); normalised traces.
#   Coins:    beta = 1.0  (same penalty per raw coin as the working individual run 3kqci1ie: beta 0.5 on 2*r_i)
#   Clean Up: beta = 0.05 / (1 - 0.99*0.975) = 1.4378, exactly Hughes et al. (2018)'s advantageous IA (alpha 0, beta 0.05)
BLIND="ENV_KWARGS.inequity_aversion_observe_traces=False"
MIA="+ENV_KWARGS.inequity_aversion=True +ENV_KWARGS.enable_smooth_rewards=True +ENV_KWARGS.inequity_aversion_team_reward=True +ENV_KWARGS.inequity_aversion_normalize_trace=True +ENV_KWARGS.inequity_aversion_observe_traces=True +ENV_KWARGS.inequity_aversion_trace_view=spatial +ENV_KWARGS.inequity_aversion_alpha=0.0"
L4="--gres=gpu:l4:1"

sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

# --- Coins (1e8 steps) ---
sub --job-name=sj-mappo-coins-common      --time=01:00:00 slurm/train.sbatch --algo MAPPO --env coins NUM_ENVS=64 TOTAL_TIMESTEPS=1e8 REWARD=MAPPO_common
sub --job-name=sj-ippo-coins-teamia  $L4  --time=02:00:00 slurm/train.sbatch --algo IPPO --env coins reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.0 TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-coins-teamia-blind --time=02:00:00 slurm/train.sbatch --algo IPPO --env coins reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.0 $BLIND REWARD=team_ia_blind TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-coins-dual         --time=02:00:00 slurm/train.sbatch --algo IPPO --env dual_coins TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-coins-dual-ia $L4  --time=02:00:00 slurm/train.sbatch --algo IPPO --env dual_coins reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.0 DUAL_IND_REWARD=ia REWARD=dual_ia TOTAL_TIMESTEPS=1e8
sub --job-name=sj-mappo-coins-dual        --time=01:00:00 slurm/train.sbatch --algo MAPPO --env dual_coins NUM_ENVS=64 TOTAL_TIMESTEPS=1e8

# --- Clean Up (IPPO 1e8 at 32 envs, about 2.6 h; MAPPO 4 envs x 100 steps runs about 5.4k steps/s, so 7e7 to finish in 4 h) ---
sub --job-name=sj-ippo-cleanup-common       --time=04:00:00 slurm/train.sbatch --algo IPPO --env cleanup NUM_ENVS=32 TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-cleanup-teamia       --time=04:00:00 slurm/train.sbatch --algo IPPO --env cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.4378 NUM_ENVS=32 TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-cleanup-teamia-blind --time=04:00:00 slurm/train.sbatch --algo IPPO --env cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.4378 $BLIND REWARD=team_ia_blind NUM_ENVS=32 TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-cleanup-dual         --time=04:00:00 slurm/train.sbatch --algo IPPO --env dual_cleanup NUM_ENVS=32 TOTAL_TIMESTEPS=1e8
sub --job-name=sj-ippo-cleanup-dual-ia      --time=04:00:00 slurm/train.sbatch --algo IPPO --env dual_cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.4378 DUAL_IND_REWARD=ia REWARD=dual_ia NUM_ENVS=32 TOTAL_TIMESTEPS=1e8
sub --job-name=sj-mappo-cleanup-teamia      --time=04:00:00 slurm/train.sbatch --algo MAPPO --env cleanup $MIA +ENV_KWARGS.inequity_aversion_beta=1.4378 REWARD=MAPPO_team_ia TOTAL_TIMESTEPS=7e7

./alice.sh status
