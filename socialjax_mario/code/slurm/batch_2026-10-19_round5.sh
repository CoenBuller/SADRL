#!/usr/bin/env bash
# Batch 2026-10-19 (branch dir3-dual-critic), approved by Mario in workbook round 5 (T5.1, T5.2, G.1). Clean Up only, seed 30.
# Run from the repo root on the laptop: bash slurm/batch_2026-10-19_round5.sh
#   MAPPO (B1a-B4): 1.5e8 steps at about 4.9k steps/s, about 8.5 h, so the 7-day gpu-2080ti-11g partition with a 12 h limit.
#   IPPO (B5-B7): 1e8 steps at 32 envs, about 2.6 h, gpu-short.
set -euo pipefail
cd "$(dirname "$0")/.."

# MAPPO has no reward group: the team-IA env flags go in as +ENV_KWARGS (guilt only, normalised traces, spatial view).
MIA="+ENV_KWARGS.inequity_aversion=True +ENV_KWARGS.enable_smooth_rewards=True +ENV_KWARGS.inequity_aversion_team_reward=True +ENV_KWARGS.inequity_aversion_normalize_trace=True +ENV_KWARGS.inequity_aversion_trace_view=spatial +ENV_KWARGS.inequity_aversion_alpha=0.0"
HIDDEN_B05="+ENV_KWARGS.inequity_aversion_observe_traces=False +ENV_KWARGS.inequity_aversion_beta=0.5"
VISIBLE_HUGHES="+ENV_KWARGS.inequity_aversion_observe_traces=True +ENV_KWARGS.inequity_aversion_beta=1.4378"
LONG="--partition=gpu-2080ti-11g --time=12:00:00"
SHORT="--time=04:00:00"

sub() { ./alice.sh submit "$@" 2>&1 | tail -1; }

# --- MAPPO, 1.5e8 ---
sub --job-name=sj-mappo-cleanup-dualia-b05h $LONG slurm/train.sbatch --algo MAPPO --env dual_cleanup $MIA $HIDDEN_B05 DUAL_IND_REWARD=ia REWARD=MAPPO_dual_ia_b05_hidden TOTAL_TIMESTEPS=1.5e8      # B1a: 1+3, round-3 settings
sub --job-name=sj-mappo-cleanup-dualia-hug  $LONG slurm/train.sbatch --algo MAPPO --env dual_cleanup $MIA $VISIBLE_HUGHES DUAL_IND_REWARD=ia REWARD=MAPPO_dual_ia TOTAL_TIMESTEPS=1.5e8               # B1b: 1+3, as IPPO 1+3 (rc30i6i1)
sub --job-name=sj-mappo-cleanup-teamia-b05h $LONG slurm/train.sbatch --algo MAPPO --env cleanup $MIA $HIDDEN_B05 REWARD=MAPPO_team_ia_b05_hidden TOTAL_TIMESTEPS=1.5e8                             # B2: dir 1, longer (T5.2: beta 0.5 hidden)
sub --job-name=sj-mappo-cleanup-common      $LONG slurm/train.sbatch --algo MAPPO --env cleanup REWARD=MAPPO_common TOTAL_TIMESTEPS=1.5e8                                                        # B3: same-length baseline
sub --job-name=sj-mappo-cleanup-dual        $LONG slurm/train.sbatch --algo MAPPO --env dual_cleanup REWARD=MAPPO_dual TOTAL_TIMESTEPS=1.5e8                                                     # B4: dir 3 alone, world-state critic

# --- IPPO, 1e8 at 32 envs ---
sub --job-name=sj-ippo-cleanup-teamia-b05h  $SHORT slurm/train.sbatch --algo IPPO --env cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=0.5 ENV_KWARGS.inequity_aversion_observe_traces=False REWARD=team_ia_b05_hidden NUM_ENVS=32 TOTAL_TIMESTEPS=1e8   # B5
sub --job-name=sj-ippo-cleanup-teamia-b05v  $SHORT slurm/train.sbatch --algo IPPO --env cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=0.5 REWARD=team_ia_b05 NUM_ENVS=32 TOTAL_TIMESTEPS=1e8                                                      # B5b: xuogxbcy with beta 0.5
sub --job-name=sj-ippo-cleanup-blend-ia     $SHORT slurm/train.sbatch --algo IPPO --env dual_cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.4378 DUAL_IND_REWARD=ia DUAL_MODE=blend REWARD=blend_ia NUM_ENVS=32 TOTAL_TIMESTEPS=1e8              # B6: control for 1+3
sub --job-name=sj-ippo-cleanup-dualia-gT    $SHORT slurm/train.sbatch --algo IPPO --env dual_cleanup reward=team_ia ENV_KWARGS.inequity_aversion_beta=1.4378 DUAL_IND_REWARD=ia DUAL_GAMMA_TEAM=0.995 REWARD=dual_ia_gT0995 NUM_ENVS=32 TOTAL_TIMESTEPS=1e8  # B7: 1+3 with gamma_T 0.995 (G.1)

./alice.sh status
