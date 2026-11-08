# Round 15 runs: the fixed environment, the EV gate, the update study (Clean Up)

Three studies, run 6–8 Nov 2026 on ALICE (L4), branch `round15-envfix-gate`. All runs: Clean Up, 7 agents,
common reward, small CNN + full FC, shared actor without agent id, LR 5e-4 annealed over the run, 2 epochs,
4 minibatches (700 samples each). Team return = 7 × `returned_episode_original_returns`, mean of the last 5% of
training. Std over seeds uses ddof = 1. Made by `research/mappo_grid/round15_plots.py` (`--download` refreshes the
W&B cache in `results/round15/`).

**Two environments, never mixed.** *Fixed* = `ENV_KWARGS.self_channel_fix=True` (upstream's self-channel fix,
`[SADRL fix 2026-11-06]`, REWARD names `FIX_*`). *Imported* = `self_channel_fix=False`, the environment of every run
before 6 Nov (REWARD names `OLDENV_*`, plus the round-10/13 runs).

## Final results

| Setting [REWARD] | Team return, last 5% (mean ± std) | Per seed | First reward (team ≥ 100) | Steps to 2000 | W&B ids |
|---|---|---|---|---|---|
| **Fixed environment, seeds 30–35, 1e8** | | | | | |
| MAPPO [FIX_MAPPO_csff_4x100] | 1221 ± 799 | 903 / 891 / 895 / 895 / 893 / 2852 | 0.9M, 3.7M, 0.9M, 1.2M, 0.9M, 1.0M | never ×5, 13.5M | `zgjyurte`, `zujgb5lk`, `b3dx3e9s`, `qq4c9wys`, `0pos93v7`, `d7k301o4` |
| CC-MAPPO loo [FIX_CC_loo_team_csff_4x100] | 910 ± 16 | 933 / 894 / 900 / 924 / 896 / 916 | 2.1M, 4.0M, 4.7M, 0.7M, 1.2M, 1.8M | never | `xkwio5fh`, `779r1v8x`, `ukmhjx2i`, `rvb1f87n`, `tfz0pdsg`, `dpz7mhfg` |
| CC-MAPPO loo + EV gate [FIX_CC_loo_gate_csff_4x100] | 918 ± 21 | 932 / 931 / 932 / 893 / 931 / 889 | 0.7M, 2.9M, 0.5M, 3.3M, 1.2M, 1.3M | never | `id7xb0x5`, `y1m2van9`, `m372cfta`, `lnopfz44`, `vhpui0oz`, `ole8oakm` |
| **Imported environment, seeds 30–32** | | | | | |
| CC-MAPPO loo, 1e8 (round 13) [CC_loo_team_csff_4x100] | 3214 ± 4 | 3209 / 3216 / 3217 | 3.6M, 2.6M, 0.9M | 11.4M, 8.7M, 7.0M | `0h5xf6ob`, `fgu60a5c`, `lgvtxgcc` |
| CC-MAPPO loo + EV gate, 1e8 [OLDENV_CC_loo_gate_csff_4x100] | 3164 ± 23 | 3191 / 3149 / 3152 | 1.4M, 2.4M, 0.6M | 7.4M, 11.6M, 6.8M | `boy5swnq`, `x5r49s2z`, `6kvhl0uf` |
| MAPPO 4 × 100, 1e8 (round-10 grid) | 2905 ± 50 | 2962 / 2870 / 2882 | 5.5M, 1.2M, 3.7M | 19.0M, 12.0M, 12.6M | `8xpn1ipu`, `5vs2xiwc`, `xkymgicp` |
| MAPPO 32 × 100, 1e8 (round 13) | 2205 ± 90 | 2221 / 2285 / 2108 | 11.4M, 9.8M, 19.5M | 78.7M, 51.0M, 82.9M | `jbd5ffqy`, `2t2x14mf`, `yy5w0jmn` |
| MAPPO 32 × 100, **3e8** [OLDENV_MAPPO_upd3e8_32x100_mb700] | 2871 ± 76 | 2919 / 2783 / 2909 | 11.3M, 17.8M, 11.0M | 49.4M, 76.9M, 63.4M | `ltikobjp`, `zqe33141`, `verc8kxh` |
| MAPPO 4 × 1000, 1e8 (round 13) | 1145 ± 81 | 1232 / 1071 / 1131 | 6.2M, 12.5M, 3.5M | never | `rnesd5b7`, `l5gplxvu`, `puqxqnph` |
| MAPPO 4 × 1000, **3e8** [OLDENV_MAPPO_upd3e8_4x1000_mb700] | 1397 ± 22 | 1378 / 1392 / 1422 | 5.3M, 5.4M, 4.9M | never | `na9fmdlx`, `c4trl9r2`, `ir6gsm7c` |

Per-seed numbers, with the spread over the last 20%, are in `runs.csv`.

Plots: `fixed_env_team_return.png`, `fixed_env_roles.png`, `gate.png`, `update_study.png`, `final_team_return.png`.

## 1. The fixed environment: a trap near 900

- **The start-up problem is gone in every arm.** All 18 runs get their first reward within 0.5–4.7M steps, CC
  included (on the imported environment, 6 of 12 CC runs in rounds 12–13 never did).
- **17 of 18 runs then stop near 900** (889–933) and stay there for the remaining 90M steps
  (`fixed_env_team_return.png`). Only MAPPO seed 35 escapes, reaching 2000 at 13.5M and ending at 2852,
  close to MAPPO on the imported environment (2905).
- **What the trap looks like** (`fixed_env_roles.png`, MAPPO, last 5%, rank 0 = the agent that did most in that
  episode): stuck runs have one main cleaner (155 ± 60 waste cells per episode; 261 in seed 32) and a weak second
  one (60), and everybody else harvests a little (282 → 12 apples by rank). Seed 35 has two full-time cleaners
  (261 and 204) and five harvesters (647 → 453 apples), the same division of labour as MAPPO on the imported
  environment (266 and 200 cells; 663 → 446 apples). So the 900 runs lack a stable **second cleaner**: too little
  waste is cleared, few apples grow, and every harvester's return is low.
- CC runs log per-agent-id values, not per-episode ranks, so the same breakdown is not available for them.

**Two hypotheses (neither tested).**
- *H1, seeing yourself makes you a harvester.* On the imported environment agents 0–5 saw their own body as
  "another agent" and only agent 6 saw itself. Agent 6 was the full-time harvester in every round-13 loo run, and in
  the round-13 evaluation (`../2026-11-05_eval_agents/table.txt`) it cleaned least in most learned runs (4–38 clean
  actions against 30–150 for the others). On the fixed environment every agent sees itself, and the team ends up
  with about one cleaner. So the bugged view may have pushed agents towards cleaning; the mechanism is unknown.
- *H2, symmetry.* All 7 agents share one actor without agent id, so a split into roles has to come from position
  and chance; seed 35 found the 2-cleaner split, the others did not. Against H2 alone: on the imported environment
  agents 0–5 were just as identical, and the 2-cleaner split appeared in every learned run, reassigned from episode
  to episode.
- Cheap test of H1, no training: evaluate saved actors in the other environment (fixed-env actors with
  `self_channel_fix=False`, imported-env actors with `True`) and compare cleaning per agent. Direct test of H2:
  agent id in the actor's input (idea I-1) on the fixed environment. All 27 round-15 actors are saved on ALICE.

## 2. The EV gate does not help (imported environment)

- The gate (`CREDIT_GATE=ev`, A = (1 − w) Â + w c, w = clip(ē / 0.5, 0, 1), ē running mean of the credit critic's
  explained variance with β = 0.9 per update) ends at **3164 ± 23**, against **3214 ± 4** for plain loo in round 13
  (same setup, same seeds). Both learn on 3 of 3 seeds; steps to 2000 are 6.8–11.6M against 7.0–11.4M.
- The round-13 loo runs never failed to start, so there was no failure for the gate to remove in this setup. In
  the round-12/13 tally loo learned on 4 of 6 runs; the 2 failures were October runs, with full nets and
  `CREDIT_NORM=self`. (The comment in `slurm/batch_2026-11-07_oldenv_gate.sh` says "4 of 6 never got the first
  reward"; that is wrong, 4 of 6 did.)
- **What the gate does** (`gate.png`, right): w flickers between 0 and 1 for the first 1–4M steps, because ē
  forgets in about 10 updates (≈ 4,000 environment steps) and the batch explained variance is noisy before rewards
  arrive. It stays ≥ 0.9 after 0.5–9.4M steps on the imported environment (one fixed-env run: 19M). Late in
  training it still dips (to 0.64) in 10–33% of updates on the imported environment, mixing MAPPO's advantage back
  in, which may explain the 50-point lower final return.
- On the fixed environment the gate changes nothing either (918 ± 21 against 910 ± 16 for loo).

## 3. Update study (imported environment)

At 3e8 steps, 32 × 100 keeps learning (2205 at 1e8 → **2871** at 3e8, close to 4 × 100 at 1e8) and 4 × 1000 barely
moves (1145 → **1397**). Against the number of PPO updates (`update_study.png`, right; mean over seeds at matched
update counts, ±3% window):

| PPO updates | 4 × 100, 1e8 | 32 × 100, 1e8 | 32 × 100, 3e8 | 4 × 1000, 1e8 | 4 × 1000, 3e8 |
|---|---|---|---|---|---|
| 10k | 390 | 925 | 986 | 968 | 860 |
| 25k | 1409 | 1977 | 2076 | 1120 | 1148 |
| 50k | 2165 | – | 2589 | – | 1325 |
| 75k | 2449 | – | 2753 | – | 1393 |
| 90k | 2573 | – | 2857 | – | – |

- **Per update, 32 × 100 is the best shape**: ahead of 4 × 100 at every matched count. More parallel environments
  per update give a better gradient.
- **4 × 1000 stalls at about 1100–1400** however many updates it gets. Four 1000-step rollouts per update are four
  long, correlated trajectories: more updates do not fix that.
- Round 13's reading ("the number of updates is what counts; per update the curves almost coincide") holds only up
  to about 10k updates. Beyond that the rollout shape matters: many environments help, long rollouts hurt.
- Caveat: the LR anneals to 0 over each run's total, so at a given update the 3e8 runs have a higher LR than the
  1e8 runs. The 1e8 and 3e8 curves of the same shape still nearly coincide (dashed and solid lines).

## GIFs

Each run logs an end-of-training episode GIF to W&B (`Episode GIF`).

## 4. Swap evaluation (round 15, T.1): the view of oneself changes the behaviour

ALICE job 5501025 (`slurm/round15_eval_swap.sbatch`, `research/mappo_grid/eval_agents.py 20 --swap`), 8 Nov 2026,
13 min. Every saved actor plays 20 episodes (sampled actions) in **both** environments, without retraining. Team
return = apples eaten by all 7 agents, mean over episodes. "Clean actions" counts firing the cleaning beam, not waste
removed. Raw output: `results/round15/eval_swap.json` and the job log (laptop, not in git).

| Actors (trained on) | n | Team return, imported env | Team return, fixed env | 2nd cleaner's clean actions, imported / fixed |
|---|---|---|---|---|
| MAPPO, stuck runs (fixed) | 5 | **1526** (1162–1721) | 895 (889–909) | 104 / 57 |
| MAPPO seed 35, escaped (fixed) | 1 | 2790 | 3009 | 219 / 257 |
| CC loo (fixed) | 6 | **1546** (1043–1843) | 910 (891–935) | 99 / 43 |
| CC loo + gate (fixed) | 6 | **1666** (1217–2002) | 915 (891–928) | 124 / 201 * |
| MAPPO grid 4 × 100 (imported) | 3 | 2908 (2855–2936) | **2179** (1321–2693) | 199 / 143 |
| CC loo, round 13 (imported) | 3 | 3211 (3203–3218) | **2495** (2119–3059) | 203 / 117 |
| CC loo + gate (imported) | 3 | 3176 (3158–3194) | **1529** (485–2744) | 208 / 110 |

\* Gate seed 35 fires the beam about 750 times per agent per episode in the fixed environment while scoring 891, which
inflates this mean; without it the fixed-env value is in line with the other stuck actors.

Reading:
- **Every stuck fixed-env actor (17 of 17) scores more in the imported environment**: 1043–2002 instead of ≈ 900,
  with the same weights. When agents 0–5 see their own body as "another agent", the second cleaner cleans about twice
  as much.
- **Every imported-env actor loses in the fixed environment** (9 of 9; 485–3059 instead of 2855–3218). Its second
  cleaner cleans less. Agent 6, whose own view is identical in both environments, cleans more there (about 5 → 50–60
  clean actions in the CC actors), apparently making up for the others.
- **The one policy with a firm 2-cleaner split (MAPPO seed 35) is unaffected** (2790 against 3009).
- So H1 is supported: what an agent sees of itself shifts it between harvesting and cleaning, and the bugged view
  pushed agents towards cleaning. The swap does not reach the trained level in either direction, so the roles are also
  partly learned (H2 is not ruled out). Mechanism still unknown; one guess is that an "other agent" on one's own cell
  reads as a crowded orchard. Caveat: 20 episodes per actor and environment; actors in an environment they were not
  trained on.
