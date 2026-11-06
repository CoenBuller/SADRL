# Mario's SocialJax work

This folder shares Mario's SocialJax code and results. It is generated from a private working fork by
an export script and overwritten at every export, so please don't edit it by hand.

- `code/` is the **complete code**: [cooperativex/SocialJax](https://github.com/cooperativex/SocialJax)
  (commit `9df972d`, 2026-08-21, Apache 2.0 licence, see `code/LICENSE`) with all of Mario's changes and
  research code. It runs as is; nothing to apply. It is the newest research branch, which contains all
  earlier ones.
- `results/` holds the results of the runs on Clean Up.

Until 2026-11-06 this folder held `.patch` files instead; they are replaced by `code/`.

## Run it

```bash
cd socialjax_mario/code
pip install -r requirements.txt            # Python 3.10
export PYTHONPATH=$PWD:$PYTHONPATH
python tests/test_self_channel.py          # tests run as plain scripts
python algorithms/train.py --algo MAPPO --env cleanup NUM_ENVS=2 NUM_STEPS=100 TOTAL_TIMESTEPS=400 WANDB_MODE=disabled
python algorithms/train.py --algo MAPPO --env cc_cleanup CREDIT_MODE=loo CREDIT_GATE=ev \
    NUM_ENVS=2 NUM_STEPS=100 TOTAL_TIMESTEPS=400 WANDB_MODE=disabled
```

The settings of every real run are in the batch scripts under `code/slurm/batch_*.sh`.

## Where things are

| Path | What it is |
|---|---|
| `code/research/cc_mappo/` | Counterfactual-credit MAPPO: credit modes (`credit.py`: coma, qret, loo, local, the explained-variance gate), networks, training loop |
| `code/algorithms/MAPPO/mappo_cnn_cleanup.py` | MAPPO on Clean Up with our fixes (world state stored once per step), network options (`CNN_SIZE`, `FC_SIZE`, `CNN_LAYERS`) and diagnostics |
| `code/algorithms/MAPPO/mappo_cnn_cc_cleanup.py` | Entry point for CC-MAPPO on Clean Up (`--env cc_cleanup`) |
| `code/research/dual_critic/` | IPPO/MAPPO with a team critic and an individual critic |
| `code/research/mappo_grid/` | Run summaries, plots and the per-agent evaluation of saved actors |
| `code/research/adaptive_ia/` | Adaptive inequity-aversion demo |
| `code/socialjax/environments/` | The environments, with the inequity-aversion fix and the self-channel fix (below) |
| `code/tests/` | Tests (run as plain scripts) |
| `code/slurm/`, `code/alice.sh` | Batch scripts for the ALICE cluster |

**Self-channel fix (2026-11-06).** The SocialJax version we started from has an off-by-one in the
observation's "self" channel: only the last agent sees itself, the others see their own body as another
agent. Upstream fixed it on 25 Sep 2026 (`fb4ac0f`). Here it is an environment option,
`self_channel_fix`, default `True`; `False` gives back the old observation exactly. **Every run before
2026-11-06 used the old observation**; runs on the fixed one have names starting with `FIX_`.

## Results

`results/` holds one folder per experiment (Clean Up, seeds 30–32, 1e8 steps), each with a README
(setup, table with W&B run ids, reading), the plots and a CSV per run:

| Folder | What it is |
|---|---|
| `results/2026-11-06_round13/` | **Latest.** CNN depth, CC-MAPPO (qret, loo) against MAPPO, the rollout 2 × 2, and why CC-MAPPO sometimes never learns |
| `results/2026-11-05_eval_agents/` | Per-agent evaluation of the saved actors: is there a lazy agent? (No.) |
| `results/2026-11-03_mappo_grid/` | MAPPO network (CNN / FC size) × rollout (4 × 100, 32 × 1000) grid |
| `results/gifs/` | One end-of-training episode per setting (seed 31) |

Runs are logged to W&B (entity `magalinimario-leiden-university`, project `socialjax`). On ALICE,
`alice.sh` expects an ssh host alias `alice` and keeps everything under `~/socialjax`. The W&B key is
read from `.secrets/wandb_api_key`, which is never committed.
