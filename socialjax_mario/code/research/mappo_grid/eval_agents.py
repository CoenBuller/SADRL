"""Per-agent-id behaviour of saved MAPPO / CC-MAPPO actors on Clean Up (round 13, L.1: is there a lazy agent?).

For every checkpoint matching the patterns below, play N_EPISODES episodes of 1000 steps with the saved actor (sampled
actions, as in training) and record, per agent id: apples eaten, clean actions, stay actions. An agent is called idle in
an episode if its apples and its clean actions are both below 5% of the per-agent mean of that episode.

Run on ALICE from ~/socialjax:  python research/mappo_grid/eval_agents.py [N_EPISODES]
Writes results/mappo_grid/eval_agents.json and prints a table.
"""
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
from algorithms.utils.networks import mappo_networks  # noqa: E402

CLEAN, STAY = 8, 6
PATTERNS = ["checkpoints/clean_up_seed3?_reward_MAPPO_grid_*_4x100.pkl",
            "checkpoints/clean_up_seed3?_reward_MAPPO_common_mc.pkl",
            "checkpoints/clean_up_seed3?_reward_CC_*_mc.pkl"]


def net_config(name):
    m = re.search(r"cnn-(small|full)_fc-(small|full)", name)
    return {"CNN_SIZE": m.group(1), "FC_SIZE": m.group(2)} if m else {"NETWORK_SIZE": "full"}


def run(path, n_episodes, steps=1000):
    env = socialjax.make("clean_up", num_agents=7, num_inner_steps=steps, shared_rewards=True, cnn=True, jit=True)
    actor = mappo_networks(net_config(path))[0](env.action_space().n, activation="relu")
    params = load_params(path)
    n = env.num_agents

    def episode(key):
        key, k0 = jax.random.split(key)
        obs, state = env.reset(k0)

        def step(carry, _):
            obs, state, key = carry
            key, ka, ks = jax.random.split(key, 3)
            batch = jnp.stack([obs[a] for a in env.agents])
            act = actor.apply(params, batch).sample(seed=ka)
            obs, state, _, _, info = env.step(ks, state, [act[i] for i in range(n)])
            return (obs, state, key), (info["agent_rewards"], act)

        _, (rew, act) = jax.lax.scan(step, (obs, state, key), None, steps)
        return rew.sum(0), (act == CLEAN).sum(0), (act == STAY).sum(0)

    apples, cleans, stays = jax.jit(jax.vmap(episode))(jax.random.split(jax.random.PRNGKey(0), n_episodes))
    apples, cleans, stays = (np.asarray(x, float) for x in (apples, cleans, stays))
    idle = (apples < 0.05 * apples.mean(1, keepdims=True)) & (cleans < 0.05 * cleans.mean(1, keepdims=True))
    return dict(apples=apples.mean(0).tolist(), cleans=cleans.mean(0).tolist(), stays=stays.mean(0).tolist(),
                team_return=float(apples.sum(1).mean()), idle_agents_per_episode=float(idle.sum(1).mean()),
                idle_frac_by_agent=idle.mean(0).tolist())


def main():
    n_episodes = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    paths = sorted(p for pat in PATTERNS for p in glob.glob(pat))
    out = {}
    for p in paths:
        r = run(p, n_episodes)
        out[os.path.basename(p)] = r
        print(f"{os.path.basename(p)[:-4]:60s} team {r['team_return']:6.0f}  idle/ep {r['idle_agents_per_episode']:.2f}")
        print("   apples " + " ".join(f"{x:6.0f}" for x in r["apples"]))
        print("   cleans " + " ".join(f"{x:6.0f}" for x in r["cleans"]))
        print("   stays  " + " ".join(f"{x:6.0f}" for x in r["stays"]), flush=True)
    os.makedirs("results/mappo_grid", exist_ok=True)
    with open("results/mappo_grid/eval_agents.json", "w") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    main()
