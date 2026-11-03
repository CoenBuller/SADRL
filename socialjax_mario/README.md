# Mario's SocialJax work (patches)

This folder shares Mario's SocialJax changes and research code without vendoring SocialJax,
following the same convention as the `reputation` branch. Each patch is **standalone**. Apply exactly
one of them to a clean checkout of [cooperativex/SocialJax](https://github.com/cooperativex/SocialJax)
at the commit in `BASE` (`9df972d`, "fix agents numbers", 2026-08-21).

The patches are generated from a private working fork by an export script. This folder is
overwritten at every export, so please don't edit it by hand.

## Patches

| Patch | Contents |
|---|---|
| `01-main.patch` | Inequity-aversion (IA) reward shaping fixed to match Hughes et al. (2018), and IA enabled in gift, territory, coop_mining and harvest_open. Opt-in normalised and observable IA traces, raw-return logging for IPPO. Adaptive-IA demo in `research/adaptive_ia/`. Tests, plus SLURM scripts for the ALICE cluster (`alice.sh`, `slurm/`). |
| `02-dir1-team-ia.patch` | Direction 1: team reward plus an IA penalty on raw individual traces, a spatial trace channel, `agent_rewards` in `info` (`algorithms/IPPO/config/reward/team_ia.yaml`). |
| `03-dir3-dual-critic.patch` | Direction 1, plus Direction 3: IPPO/MAPPO with a team critic and an individual critic (`research/dual_critic/`), `DUAL_MODE=blend`, `DUAL_GAMMA_TEAM`. |
| `04-dir4-cc-mappo.patch` | Directions 1 and 3, plus counterfactual-credit MAPPO (`research/cc_mappo/`, credit modes coma/qret/loo/local, `CREDIT_NORM=team`). Also MAPPO Clean Up fixes: world state stored once per env step, PPO diagnostics logging, dead-actor check. |

| `05-mappo-net-grid.patch` | Everything in 04, plus the MAPPO network grid (2026-11-03): `CNN_SIZE` / `FC_SIZE` (`small` = SocialJax MAPPO sizes, `full` = IPPO torso) for actor and critic, extra diagnostics (critic explained variance, actor/critic gradient norms, per-agent reward and clean-action ranks, end-of-training GIF in W&B), `tests/test_mappo_net_grid.py`, and the 24-run batch script `slurm/batch_2026-11-03_round10_grid.sh`. |
Patches 02-05 branched off before the last updates to 01, so each one is complete on its own and
they are not meant to be stacked.

## Usage

```bash
git clone https://github.com/cooperativex/SocialJax.git socialjax
cd socialjax
git checkout "$(cat ../socialjax_mario/BASE)"
git apply ../socialjax_mario/patches/04-dir4-cc-mappo.patch   # or 01 / 02 / 03 / 05
python tests/test_inequity_aversion_fix.py                   # tests run as plain scripts
```

Runs are logged to W&B (entity `magalinimario-leiden-university`, project `socialjax`). On ALICE,
`alice.sh` expects an ssh host alias `alice` and keeps everything under `~/socialjax`. The W&B key
is read from `.secrets/wandb_api_key`, which is never committed.
