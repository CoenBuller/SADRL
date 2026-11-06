# Mario's SocialJax work (patches)

This folder shares Mario's SocialJax changes and research code without vendoring SocialJax,
following the same convention as the `reputation` branch. Each patch is **standalone**. Apply exactly
one of them to a clean checkout of [cooperativex/SocialJax](https://github.com/cooperativex/SocialJax)
at the commit in `BASE` (`9df972d`, "fix agents numbers", 2026-08-21).

The patches are generated from a private working fork by an export script. This folder is
overwritten at every export, so please don't edit it by hand.

## Browse the code

`code/` holds the same changes as plain files, so they can be read on GitHub without applying
anything: every file that `06-round13-depth-cc.patch` adds or changes, at the same path as in
SocialJax. The rest of SocialJax is unchanged, so `code/` alone doesn't run; to run it, apply a patch
(see Usage). Good places to start:

| Path | What it is |
|---|---|
| `code/research/cc_mappo/` | Counterfactual-credit MAPPO: credit modes (`credit.py`), networks, training loop |
| `code/algorithms/MAPPO/mappo_cnn_cleanup.py` | MAPPO on Clean Up with our fixes, network options and diagnostics |
| `code/algorithms/MAPPO/mappo_cnn_cc_cleanup.py` | Entry point for CC-MAPPO on Clean Up |
| `code/research/dual_critic/` | IPPO/MAPPO with a team critic and an individual critic |
| `code/research/mappo_grid/` | Run summaries, plots and the per-agent evaluation of saved actors |
| `code/socialjax/environments/` | Inequity-aversion fix and IA support in the environments |
| `code/tests/` | Tests (run as plain scripts) |
| `code/slurm/`, `code/alice.sh` | Batch scripts for the ALICE cluster |

## Patches

| Patch | Contents |
|---|---|
| `01-main.patch` | Inequity-aversion (IA) reward shaping fixed to match Hughes et al. (2018), and IA enabled in gift, territory, coop_mining and harvest_open. Opt-in normalised and observable IA traces, raw-return logging for IPPO. Adaptive-IA demo in `research/adaptive_ia/`. Tests, plus SLURM scripts for the ALICE cluster (`alice.sh`, `slurm/`). Since 2026-11-05 `main` also includes everything in 05 (merged), so 01 now equals 05 plus the rollout 2 × 2 batch and the grid scripts. |
| `02-dir1-team-ia.patch` | Direction 1: team reward plus an IA penalty on raw individual traces, a spatial trace channel, `agent_rewards` in `info` (`algorithms/IPPO/config/reward/team_ia.yaml`). |
| `03-dir3-dual-critic.patch` | Direction 1, plus Direction 3: IPPO/MAPPO with a team critic and an individual critic (`research/dual_critic/`), `DUAL_MODE=blend`, `DUAL_GAMMA_TEAM`. |
| `04-dir4-cc-mappo.patch` | Directions 1 and 3, plus counterfactual-credit MAPPO (`research/cc_mappo/`, credit modes coma/qret/loo/local, `CREDIT_NORM=team`). Also MAPPO Clean Up fixes: world state stored once per env step, PPO diagnostics logging, dead-actor check. |
| `05-mappo-net-grid.patch` | Everything in 04, plus the MAPPO network grid (2026-11-03): `CNN_SIZE` / `FC_SIZE` (`small` = SocialJax MAPPO sizes, `full` = IPPO torso) for actor and critic, extra diagnostics (critic explained variance, actor/critic gradient norms, per-agent reward and clean-action ranks, end-of-training GIF in W&B), `tests/test_mappo_net_grid.py`, and the 24-run batch script `slurm/batch_2026-11-03_round10_grid.sh`. |
| `06-round13-depth-cc.patch` | Everything in 05 (now also in 01), plus (2026-11-05): `CNN_LAYERS` for the small MAPPO CNN (n × 16 @ 3×3), CC-MAPPO actor/critic chosen by `CNN_SIZE` / `FC_SIZE`, per-agent-id reward and clean logging (MAPPO and CC), credit diagnostics (`advantage_std`, `credit_adv_corr`, `credit_critic_explained_var`), the rollout 2 × 2 and depth/CC batch scripts, the grid summary/plot scripts and a per-agent evaluation of saved actors (`research/mappo_grid/`). Test: `tests/test_round13_depth_cc.py`. |

Patches 02-05 branched off before the last updates to 01, so each one is complete on its own and
they are not meant to be stacked.

## Usage

```bash
git clone https://github.com/cooperativex/SocialJax.git socialjax
cd socialjax
git checkout "$(cat ../socialjax_mario/BASE)"
git apply ../socialjax_mario/patches/04-dir4-cc-mappo.patch   # or 01 / 02 / 03 / 05 / 06
python tests/test_inequity_aversion_fix.py                   # tests run as plain scripts
```

Runs are logged to W&B (entity `magalinimario-leiden-university`, project `socialjax`). On ALICE,
`alice.sh` expects an ssh host alias `alice` and keeps everything under `~/socialjax`. The W&B key
is read from `.secrets/wandb_api_key`, which is never committed.
