"""Per-agent-id behaviour of saved MAPPO / CC-MAPPO actors on Clean Up (round 13, L.1: is there a lazy agent?).

For every checkpoint matching the patterns below, play N_EPISODES episodes of 1000 steps with the saved actor (sampled
actions, as in training) and record, per agent id: apples eaten, clean actions, stay actions. An agent is called idle in
an episode if its apples and its clean actions are both below 5% of the per-agent mean of that episode.

Run on ALICE from ~/socialjax:  python research/mappo_grid/eval_agents.py [N_EPISODES]
Writes results/mappo_grid/eval_agents.json and prints a table.

Round 15 (T.1, swap evaluation): --swap evaluates the round-15 actors and the imported-environment reference actors in
BOTH environments (self_channel_fix True and False), to see whether what an agent sees of itself changes who cleans:
    python research/mappo_grid/eval_agents.py 20 --swap
Writes results/round15/eval_swap.json. Besides the per-agent-id means it records per-episode ranks (each episode's
values sorted, then averaged), comparable with the clean_rank / reward_rank training logs.
Note: the round-13 evaluation (5 Nov) ran before self_channel_fix existed, so it used the imported environment;
without --swap the script now passes self_channel_fix=False to reproduce it.
"""
import argparse
import glob
import json
import os
import re
import sys

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
import socialjax  # noqa: E402
from algorithms.utils import load_params  # noqa: E402
from algorithms.utils.networks import add_agent_id, agent_id_dim, mappo_networks  # noqa: E402

CLEAN, STAY = 8, 6
PATTERNS = ["checkpoints/clean_up_seed3?_reward_MAPPO_grid_*_4x100.pkl",
            "checkpoints/clean_up_seed3?_reward_MAPPO_common_mc.pkl",
            "checkpoints/clean_up_seed3?_reward_CC_*_mc.pkl"]
# Round 15 swap evaluation: the 18 fixed-env actors, and imported-env references with the same network (small CNN + full FC).
SWAP_PATTERNS = ["checkpoints/clean_up_seed3?_reward_FIX_*_csff_4x100.pkl",
                 "checkpoints/clean_up_seed3?_reward_MAPPO_grid_cnn-small_fc-full_4x100.pkl",
                 "checkpoints/clean_up_seed3?_reward_CC_loo_team_csff_4x100.pkl",
                 "checkpoints/clean_up_seed3?_reward_OLDENV_CC_loo_gate_csff_4x100.pkl"]


def net_config(name):
    m = re.search(r"cnn-(small|full)_fc-(small|full)", name)
    if m:
        cfg = {"CNN_SIZE": m.group(1), "FC_SIZE": m.group(2)}
    elif "_csff_" in name:  # round-13/15 naming: small CNN + full FC
        cfg = {"CNN_SIZE": "small", "FC_SIZE": "full"}
    else:
        cfg = {"NETWORK_SIZE": "full"}
    if "_aid_" in name:  # round 16 (A.1): actor with a one-hot agent id
        cfg.update(AGENT_ID=True, ENV_KWARGS={"num_agents": 7})
    return cfg


def run(path, n_episodes, steps=1000, self_channel_fix=False):
    env = socialjax.make("clean_up", num_agents=7, num_inner_steps=steps, shared_rewards=True, cnn=True, jit=True,
                         self_channel_fix=self_channel_fix)
    cfg = net_config(path)
    actor = mappo_networks(cfg)[0](env.action_space().n, activation="relu")
    id_dim = agent_id_dim(cfg)
    params = load_params(path)
    n = env.num_agents

    def episode(key):
        key, k0 = jax.random.split(key)
        obs, state = env.reset(k0)

        def step(carry, _):
            obs, state, key = carry
            key, ka, ks = jax.random.split(key, 3)
            batch = jnp.stack([obs[a] for a in env.agents])
            if id_dim:
                batch = add_agent_id(batch, jnp.arange(n), id_dim)
            act = actor.apply(params, batch).sample(seed=ka)
            obs, state, _, _, info = env.step(ks, state, [act[i] for i in range(n)])
            return (obs, state, key), (info["agent_rewards"], act)

        _, (rew, act) = jax.lax.scan(step, (obs, state, key), None, steps)
        return rew.sum(0), (act == CLEAN).sum(0), (act == STAY).sum(0)

    apples, cleans, stays = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), n_episodes))
    apples, cleans, stays = (np.asarray(x, float) for x in (apples, cleans, stays))
    idle = (apples < 0.05 * apples.mean(1, keepdims=True)) & (cleans < 0.05 * cleans.mean(1, keepdims=True))
    rank = lambda x: (-np.sort(-x, axis=1)).mean(0).tolist()  # per-episode ranks, rank 0 = most
    return dict(apples=apples.mean(0).tolist(), cleans=cleans.mean(0).tolist(), stays=stays.mean(0).tolist(),
                team_return=float(apples.sum(1).mean()), team_return_std=float(apples.sum(1).std()),
                idle_agents_per_episode=float(idle.sum(1).mean()), idle_frac_by_agent=idle.mean(0).tolist(),
                apples_rank=rank(apples), cleans_rank=rank(cleans), n_episodes=n_episodes,
                self_channel_fix=bool(self_channel_fix))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_episodes", nargs="?", type=int, default=10)
    ap.add_argument("--swap", action="store_true", help="round-15 actors in both environments")
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--checkpoints", nargs="*", help="explicit checkpoint paths (override the patterns)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    pats = SWAP_PATTERNS if args.swap else PATTERNS
    paths = args.checkpoints or sorted(p for pat in pats for p in glob.glob(pat))
    envs = [False, True] if args.swap else [False]
    out_path = args.out or ("results/round15/eval_swap.json" if args.swap else "results/mappo_grid/eval_agents.json")
    out = {}
    for p in paths:
        for fix in envs:
            r = run(p, args.n_episodes, args.steps, self_channel_fix=fix)
            key = os.path.basename(p) + ("|fixed" if fix else "|imported")
            out[key] = r
            print(f"{key:72s} team {r['team_return']:6.0f} ± {r['team_return_std']:4.0f}  "
                  f"idle/ep {r['idle_agents_per_episode']:.2f}")
            print("   apples " + " ".join(f"{x:6.0f}" for x in r["apples"]))
            print("   cleans " + " ".join(f"{x:6.0f}" for x in r["cleans"]))
            print("   cleans by rank " + " ".join(f"{x:6.0f}" for x in r["cleans_rank"]), flush=True)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
