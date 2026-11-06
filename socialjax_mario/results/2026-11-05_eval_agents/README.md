# Per-agent-id evaluation of saved actors (round 13, L.1: is there a lazy agent?)

ALICE job 5454135 (`slurm/round13_eval_agents.sbatch`, `research/mappo_grid/eval_agents.py`), 5 Nov 2026. Each saved final actor plays 10 episodes of 1000 steps (sampled actions). Per agent id (mean over the 10 episodes): apples eaten, clean actions, stay actions. "Idle" = an agent whose apples and clean actions are both below 5% of that episode's per-agent mean. This is only meaningful in runs that learned; in the failed runs (team ≈ 0) the apple mean is 0, so nobody counts as idle.

Checkpoints: the round-10 grid at 4 × 100 (12), stage-2 MAPPO common `_mc` (3), stage-2 CC `qret_mc` s30/s31 and `loo_mc` s31/s32. qret s32 and loo s30 crashed and saved no actor.

Result (`table.txt`, `eval_agents.json`):
- **No idle agent in any of the 15 runs that learned**, MAPPO or CC: 0.00 idle agents per episode everywhere.
- **Roles are not tied to agent ids.** Averaged over episodes, every agent eats (≈ 180–520 apples) and cleans (≈ 7–160). The within-episode split (2 cleaners, 5 harvesters, seen in the training logs) is reassigned from episode to episode, presumably by spawn position: the shared actor has no agent id (idea I-1).
- In several runs one agent id cleans much less than the others on average (e.g. 7–39 against 50–160). That's a mild, consistent preference, not idleness.
