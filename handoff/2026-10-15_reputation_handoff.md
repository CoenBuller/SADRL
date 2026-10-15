# Handoff: 2026-10-15 (afternoon), reputation channel

## RESUME HERE (state when the session ended)
- **A reputation observation channel is implemented, tested and committed** on branch `reputation` (commit `38a7ada`), and pushed to `origin`. It is not merged into `main`.
- It is **off by default**. Turn it on for any algorithm and env with `REPUTATION=True` (optionally with `REPUTATION_LAMBDA`, default 0.99):
  `python algorithms/train.py --algo IPPO --env pd_arena REPUTATION=True REPUTATION_LAMBDA=0.95`
- So far only tiny smoke runs have been done. **No full training run and no cluster jobs yet.** Whether reputation changes behaviour is still open.
- **Next topics (Mario decides):**
  - (a) Full runs on ALICE with and without `REPUTATION=True` (same seed) on one env, e.g. pd_arena or coins, under common and individual reward.
  - (b) Pick λ: 0.99 is about a 100-step memory; episodes are 1000 steps.
  - (c) Consider normalising the channel. rep is unbounded and grows with reward scale (×N in most envs), while the other channels are 0/1.
  - (d) Merge `reputation` into `main` once the runs look right.

This is a separate session from the morning one (`2026-10-15_handoff.md`, Coins IA). Nothing from that thread was touched.

## 1. Definition
- `R_i(t) = λ·R_i(t−1) + r_i(t)`, reset to 0 at every new episode (`done["__all__"]`).
- `rep_i(t) = R_i(t) − mean_{j≠i} R_j(t)`.
- **Channel:** a map holding `rep_j` at agent j's cell and 0 elsewhere. It is cut out and rotated exactly like each agent's own 11×11 view, then appended as the **last** channel, which makes the obs float32 with C+1 channels.
- **`r_i` is the individual reward taken before the common/shaped conversion,** as Mario required, so reputation works under `shared_rewards=True` too. Its scale is what the agent would get in individual-reward mode, which in most envs is the raw reward ×num_agents.

## 2. What changed (all on branch `reputation`)
| File | Change |
|---|---|
| `socialjax/environments/*/*.py` (all 9 envs) | Every reward mode now writes `info["individual_rewards"]` (per agent, before aggregation). The existing `original_rewards` key couldn't be used: in gift, mushrooms, coop_mining and lb_foraging it already holds the shared value in shared mode. |
| `socialjax/wrappers/reputation.py` (new) | `ReputationWrapper`: state `ReputationState(env_state, discounted_returns)`. It adds the channel, sets `info["reputation"]`, reports `observation_space()` with C+1, and passes `render()` through. coop_mining uses a centred view and is handled. |
| `socialjax/registration.py` | `make(env_id, reputation=False, reputation_lambda=0.99, **kw)` wraps when asked. |
| `algorithms/*/config/*_base.yaml` (6 families) | `REPUTATION: False`, `REPUTATION_LAMBDA: 0.99`. |
| `algorithms/train.py` | `_apply_reputation(cfg)` copies the flags into `ENV_KWARGS`. Every algorithm builds its env via `socialjax.make(**ENV_KWARGS)`, so no per-algorithm edits were needed. |
| `tests/test_reputation.py` (new) | Standalone script, no pytest, like the other tests. |

## 3. Verification
- **Run the tests with:** `PYTHONPATH=$PWD JAX_PLATFORMS=cpu python tests/test_reputation.py`. All 8 envs pass: coin_game, harvest_common_open, clean_up, coop_mining, territory_open, pd_arena, mushrooms, gift. Each is tested with shared and individual rewards, 30 random steps, and all 4 orientations.
  - The formula and reset are checked against a hand-computed R.
  - The obs shape matches `observation_space()`.
  - **Alignment:** the env's grid is run through the wrapper's slice-and-rotate, and the result must match exactly the cells where the env's own obs is non-zero. This checks offset and rotation independently of how each env encodes agents.
- **Smoke runs:** on CPU, `NUM_ENVS=2`, a few steps, `WANDB_MODE=disabled`. All finished with exit 0:
  - IPPO, pd_arena, with reputation
  - IPPO, pd_arena, without reputation
  - MAPPO, pd_arena, with reputation
  - SVO, coop_mining, with reputation
  - VDN, pd_arena, with reputation
- **Not tested:** IRAT and TRANSFER (same `make` path, so they should work), and the harvest_closed / harvest_partnership configs.

## 4. Gotchas found along the way
- **pd_arena's declared `observation_space()`** hard-codes channels for 4 agents. With `num_agents=2` the real obs has fewer channels. This bug predates our work; the configs use 4 agents. The test pins pd_arena to 4 agents.
- **territory's grid** contains marker ids of about 1000. Its one-hot drops them, so they never appear in obs. This is harmless and noted in the test.
- **lb_foraging** has no `OBS_SIZE`/`agent_locs`-style egocentric view. The wrapper raises `NotImplementedError` there. No algorithm config uses it.
- **W&B:** run tags are unchanged, but `REPUTATION` and `REPUTATION_LAMBDA` are in the run config, so you can filter on them.
- **VDN's `SAVE_PATH` (`checkpoints/vdn`)** is written on every run, including smoke runs. The smoke-run checkpoint was deleted.
- **Chinese comments:** a few pre-existing upstream comments exist, e.g. `pd_arena.py` "新的reward计算" = "new reward calculation" and "原来的reward计算" = "original reward calculation". They are harmless and were not changed.

## 5. Other repo
> Update: the reputation commit is now also in SADRL_github as `patches/0001-Add-opt-in-reputation-observation-channel.patch` (see README). This file is a copy of the handoff from the SocialJax fork (`401dbc0`), so paths above refer to the fork.
SADRL_github also has a `reputation` branch, with no code changes. It holds `coen.md`, a transcript of the planning chat. That repo has no socialjax code since the submodule was removed (`8109dd2`). All the work lives in this fork.
