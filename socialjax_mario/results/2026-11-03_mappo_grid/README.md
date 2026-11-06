# MAPPO network × rollout grid (round 10), Clean Up

MAPPO common reward, LR 5e-4, 2 epochs, clip 0.2/7, seeds 30–32, 1e8 env steps each, ALICE `gpu-l4-24g`, jobs 5409910–5409937 (3–5 Nov 2026).
Networks: CNN = conv layers (small: 1 × 16 @ 3×3; full: 32 @ 5×5, 32 @ 3×3, 32 @ 3×3), FC = both dense layers (small: 16, 16; full: 64, 64); same variant for actor and critic.
Rollouts: 4 envs × 100 steps (4 minibatches of 700 samples) and 32 envs × 1000 steps (500 minibatches of 448).

Team return = 7 × `returned_episode_original_returns` (checked: equals `returned_episode_returns` within 0.1%).
Curves: W&B history binned onto 500 points over env steps, running mean of 9 points, then mean ± std across seeds (as in the round 6 and 8 workbooks).

## Files
- `team_return.png/.pdf`, `explained_variance.png/.pdf`, `entropy.png/.pdf`: learning curves, mean ± std over seeds, one panel per rollout.
- `final_team_return.png/.pdf`: last 5% of training, mean ± std with the 3 seeds as dots.
- `runs.csv`: every run with W&B id, final/peak return, steps to a team return of 2000, explained variance, entropy, approx KL, clip fraction and the sorted per-agent clean / reward ranks.
- End-of-training GIFs (24, ≈ 2.5 MB each, not in git): `results/mappo_grid/gifs/<setting>_seed<seed>_<wandb id>.gif`.

Regenerate everything with `~/miniconda3/envs/SocialJax/bin/python research/mappo_grid/grid_summary.py` (download + cache), then `research/mappo_grid/grid_plots.py --gifs`.

## Result
| Network | Rollout | Team return, last 5% (mean ± std) | Seeds 30 / 31 / 32 | W&B ids |
|---|---|---|---|---|
| small CNN / small FC | 4 × 100 | 2757 ± 65 | 2818 / 2765 / 2689 | `ai5istof`, `v9iynann`, `hkmrour5` |
| small CNN / full FC | 4 × 100 | 2905 ± 50 | 2962 / 2870 / 2882 | `8xpn1ipu`, `5vs2xiwc`, `xkymgicp` |
| full CNN / small FC | 4 × 100 | 1871 ± 1621 | 8 / 2963 / 2642 | `sg7ulhxz`, `ylw71i6z`, `abho86j4` |
| full CNN / full FC | 4 × 100 | 2801 ± 30 | 2767 / 2810 / 2827 | `nydyrdaa`, `fyhimpw1`, `o8cumo8l` |
| small CNN / small FC | 32 × 1000 | 107 ± 93 | 180 / 3 / 140 | `o16remkt`, `dlwv459a`, `hq9fy407` |
| small CNN / full FC | 32 × 1000 | 85 ± 145 | 252 / 1 / 1 | `fr47gd8o`, `fpk5ucx3`, `a4nv9lgt` |
| full CNN / small FC | 32 × 1000 | 28 ± 48 | 0 / 84 / 0 | `hxbvqpq6`, `cdkdie82`, `znjct1g6` |
| full CNN / full FC | 32 × 1000 | 0 ± 0 | 0 / 0 / 0 | `ug0pp3sr`, `hvzs50a9`, `9l7nvkx1` |

Reading (round 11): 4 × 100 learns in 11 of 12 runs, 32 × 1000 in none. Among networks the differences are small; small CNN + full FC is best and fastest (team return 2000 at 12–19M steps) and became the MAPPO network from round 11 on (R.2). Full CNN + small FC failed on seed 30 (left as the finding, R.3).
