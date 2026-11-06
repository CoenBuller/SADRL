# Round 13 runs: CNN depth, CC-MAPPO against MAPPO, rollout 2 × 2 (Clean Up)

All runs: Clean Up, 7 agents, common reward, 1e8 environment steps, seeds 30–32, small CNN + full FC
(the best MAPPO network of the round-10 grid, `../2026-11-03_mappo_grid/`). Team return = 7 ×
`returned_episode_original_returns`, mean of the last 5% of training. Std over seeds uses ddof = 1.
Made by `research/mappo_grid/round13_plots.py`.

## Final results

| Setting | Team return, last 5% (mean ± std) | Seeds 30 / 31 / 32 | Steps to 2000 | W&B ids |
|---|---|---|---|---|
| MAPPO, 4 × 100 (grid reference) | 2905 ± 50 | 2962 / 2870 / 2882 | 19M, 12M, 13M | `8xpn1ipu`, `5vs2xiwc`, `xkymgicp` |
| MAPPO, CNN depth 2 | 2933 ± 118 | 2796 / 2995 / 3006 | 11M, 9M, 9M | `bdjqfi46`, `fxybxll6`, `n0ms1n9q` |
| MAPPO, CNN depth 3 | 2885 ± 89 | 2849 / 2986 / 2820 | 14M, 7M, 12M | `m9mk7tgh`, `ovosrn65`, `1npv0b30` |
| CC-MAPPO loo | **3214 ± 4** | 3209 / 3216 / 3217 | 11M, 9M, 7M | `0h5xf6ob`, `fgu60a5c`, `lgvtxgcc` |
| CC-MAPPO qret | 50 ± 83 | 3 / 0 / 146 | never | `xfe3mwmn`, `l6vjh59r`, `bvoffi14` |
| MAPPO 32 × 100, 700-sample minibatches | 2205 ± 90 | 2221 / 2285 / 2108 | 79M, 51M, 83M | `jbd5ffqy`, `2t2x14mf`, `yy5w0jmn` |
| MAPPO 4 × 1000, 700-sample minibatches | 1145 ± 81 | 1232 / 1071 / 1131 | never | `rnesd5b7`, `l5gplxvu`, `puqxqnph` |
| MAPPO 32 × 1000, 700-sample minibatches | 216 ± 69 | 256 / 136 / 256 | never | `m1xihiyv`, `ol0innsa`, `dqi4wodc` |

CC-MAPPO: `CREDIT_NORM=team` (credit divided by the std of MAPPO's advantage), shared actor without
agent id. Per-seed numbers, including the spread over the last 20% and the mean return over training,
are in `runs.csv`.

Plots: `team_return_cc_and_depth.png`, `rollout_2x2.png` (against environment steps and against
update number), `final_team_return.png`, `cc_early_phase.png`.

## Reading

**CNN depth.** Depth 2 and 3 end where depth 1 does; the differences are inside the seed spread.
Depth 2 reaches 2000 a little earlier (9–11M against 12–19M), on 3 seeds.

**Rollout 2 × 2.** With 1e8 steps, fewer, larger updates learn less: 4 × 100 (250,000 updates) >
32 × 100 (31,250) > 4 × 1000 (25,000) > 32 × 1000 (3,125). Against update number (right panel of
`rollout_2x2.png`) the larger rollouts are, if anything, a little better per update; they simply get
far fewer updates in the same number of steps.

**CC-MAPPO is all or nothing.** Over both rounds there are 12 CC runs:

| | October 2026 (`CREDIT_NORM=self`, full nets, 90M) | November 2026 (`CREDIT_NORM=team`, small CNN + full FC, 100M) | Total |
|---|---|---|---|
| qret | 2 of 3 learn (2797, 3038, ≈1) | 0 of 3 | 2 of 6 |
| loo | 1 of 3 learn (≈0, 2806, ≈6) | 3 of 3 | 4 of 6 |
| MAPPO common | 3 of 3 (2761, 2742, 2555) | 3 of 3 | 6 of 6 |

- When a CC run learns, it ends at or above MAPPO every time (October: 2797–3038 against MAPPO's
  2555–2761; November: 3209–3217 against 2870–2962), with almost no spread between seeds.
- About half the CC runs never get the first apples. The split between qret and loo flips between
  the two rounds and is not significant with 3 seeds per arm (3 of 3 against 0 of 3: Fisher p = 0.1),
  so it doesn't show that one mode is better.
- `cc_early_phase.png` shows the difference before any reward: MAPPO's `approx_kl` stays about
  10–100× lower than CC's, and its entropy stays near uniform (2.2) until the first apples. In the
  CC runs that never learn, the entropy swings between 2.2 and 0.1–0.5 while the return stays at 0.
  The credit is only weakly correlated with MAPPO's advantage (0.1–0.2) in this phase, against
  0.8–0.94 once the agents get reward.
- Working hypothesis, to discuss: before the first reward the per-agent credit is noise, and the
  normalisation scales it up to the same size as a real signal, so the shared actor moves on noise
  and loses the random exploration that finds the first apples. Not tested yet.

## GIFs

One end-of-training episode per setting (seed 31) is in `../gifs/`. All runs' GIFs are kept on the
laptop (`results/round13/gifs/`, `results/mappo_grid/gifs/`).
