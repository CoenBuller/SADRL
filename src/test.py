"""All-in-one checks for src/environments/wrappers.py. Run: python check_wrapper.py"""
import sys
import numpy as np
import jax
import jax.numpy as jnp
from src.environments.wrappers import Environment

# ------------------------------ config ------------------------------ #
ENV_NAME, N_AGENTS, N_ENVS = "clean_up", 5, 8
OWN_POS, DECAY = (9, 5), 0.9
SAMPLE_EVERY = 5          # keep every n-th rollout step for the rotation test
MIN_COUNT = 20            # min number of observations per direction to judge it

results = []


def check(name, ok, detail=""):
    results.append(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail else ""))


def info(msg):
    print(f"[INFO] {msg}")


# ------------------------------ 1. setup / shapes ------------------------------ #
print("\n=== 1. Setup and shapes ===")
env = Environment(ENV_NAME, N_AGENTS, N_ENVS, reputation=True, decay_coef=DECAY, own_pos=OWN_POS)
env_nr = Environment(ENV_NAME, N_AGENTS, N_ENVS, reputation=False)
rng = jax.random.PRNGKey(0)
obs, (st, rep), rng = env.initialize(rng)
obs_nr, (st_nr, _), _ = env_nr.initialize(rng)
E, A, H, W, C = obs.shape
Craw = C - 1
info(f"obs {obs.shape} {obs.dtype}; no-rep obs {obs_nr.shape}")

check("rep env has one extra channel vs no-rep env", obs_nr.shape[-1] == Craw and obs_nr.shape[:4] == obs.shape[:4])
check("obs dtype is floating", jnp.issubdtype(obs.dtype, jnp.floating), str(obs.dtype))
check("rep shape is (E, A)", rep.shape == (E, A), str(rep.shape))
check("OWN_POS inside view", 0 <= OWN_POS[0] < H and 0 <= OWN_POS[1] < W, f"view {H}x{W}")
check("agent_locs has shape (E, A, 3)", st.agent_locs.shape == (E, A, 3), str(st.agent_locs.shape))

step_nr = jax.jit(env_nr.step)
o2, _, r2, d2, _ = step_nr(rng, jnp.zeros((E, A), jnp.int32), (st_nr, 0))
check("no-rep step works with (state, 0)", o2.shape[-1] == Craw and r2.shape == (E, A) and d2.shape == (E, A),
      f"obs {o2.shape}, reward {r2.shape}, done {d2.shape}")

# ------------------------------ 2. add_rep / update_reputation ------------------------------ #
print("\n=== 2. Reputation channel and update rule ===")
raw = obs[..., :-1]
rep_t = jnp.broadcast_to(jnp.arange(1, A + 1, dtype=jnp.float32), (E, A))
out = env.add_rep(raw, st, rep_t)
chan = np.array(out[..., -1])
check("own cell holds own reputation", np.allclose(chan[:, :, OWN_POS[0], OWN_POS[1]], np.array(rep_t)))
check("raw channels untouched", np.array_equal(np.array(out[..., :-1]), np.array(raw)))
small = env.add_rep(raw, st, jnp.full((E, A), 0.03))
check("small reputations survive the dtype cast", np.isclose(np.array(small[..., -1]).max(), 0.03))

reward = jnp.broadcast_to(jnp.arange(A, dtype=jnp.float32), (E, A))
rep0 = jnp.ones((E, A))
got = np.array(env.update_reputation(rep0, reward, DECAY))
exp = np.array(rep0) * DECAY + (np.array(reward) - np.array(reward).mean(1, keepdims=True)) * (1 - DECAY)
check("update_reputation matches EMA formula", np.allclose(got, exp))

# ------------------------------ 3. rollout: reset, auto-reset, data for rotation ------------------------------ #
print("\n=== 3. Rollout with random actions ===")
step = jax.jit(env.step)
action_dim = env.env.action_space(0).n
ep_len = next((int(getattr(env.env, n)) for n in ("max_steps", "num_inner_steps", "max_episode_steps", "episode_length")
               if isinstance(getattr(env.env, n, None), (int, np.integer))), 1000)
T = min(ep_len + 5, 2000)
info(f"rolling out {T} steps (episode length guess {ep_len})")

fields = [f for f in getattr(st, "__dataclass_fields__", {}) if any(s in f.lower() for s in ("step", "time", "inner"))]
carry, key = (st, rep), jax.random.PRNGKey(1)
dones, reps, counters, locs_hist, samples = [], [], [], [], []
for t in range(T):
    key, ka, ks = jax.random.split(key, 3)
    act = jax.random.randint(ka, (E, A), 0, action_dim)
    obs_t, carry, rew, done, _ = step(ks, act, carry)
    dones.append(np.array(done))
    reps.append(np.array(carry[1]))
    locs_hist.append(np.array(carry[0].agent_locs))
    counters.append({f: np.array(getattr(carry[0], f)) for f in fields})
    if t % SAMPLE_EVERY == 0:
        samples.append((np.array(obs_t[..., :-1]), np.array(carry[0].agent_locs)))
dones, reps = np.stack(dones), np.stack(reps)          # (T, E, A)

check("observations finite", all(np.isfinite(s[0]).all() for s in samples))
check("rep is zero wherever done", all(np.all(reps[t][dones[t]] == 0) for t in range(T)))
check("rep sums to ~0 across agents", np.abs(reps.sum(-1)).max() < 1e-4, f"max |sum| {np.abs(reps.sum(-1)).max():.2e}")
info(f"max |rep| seen: {np.abs(reps).max():.4f} (0 means rewards were always zero under random actions)")

any_done = dones.any(-1)                               # (T, E)
if any_done.sum() == 0:
    info(f"no done in {T} steps: cannot test auto-reset (set ep_len manually)")
else:
    run, maxrun = np.zeros(E, int), np.zeros(E, int)
    for t in range(T):
        run = np.where(any_done[t], run + 1, 0)
        maxrun = np.maximum(maxrun, run)
    check("auto-reset: done is not sticky (step_env resets on done)", maxrun.max() <= 1,
          f"longest consecutive-done run {maxrun.max()}; if FAIL use self.env.step instead of step_env")
    e0 = int(np.argmax(any_done.any(0)))
    t0 = int(np.argmax(any_done[:, e0]))
    for tt in range(max(t0 - 1, 0), min(t0 + 2, T)):
        info(f"env {e0} t={tt} done={any_done[tt, e0]} locs={locs_hist[tt][e0, :, :2].tolist()} "
             + " ".join(f"{k}={v[e0]}" for k, v in counters[tt].items()))

# ------------------------------ 4. rotation / OWN_POS against the env's own obs ------------------------------ #
print("\n=== 4. Rotation convention vs the env's own observations ===")
SYMS = [lambda a, b: (a, b), lambda a, b: (b, -a), lambda a, b: (-a, -b), lambda a, b: (-b, a),
        lambda a, b: (a, -b), lambda a, b: (-a, b), lambda a, b: (b, a), lambda a, b: (-b, -a)]
SYM_STR = ["(dr, dc)", "(dc, -dr)", "(-dr, -dc)", "(-dc, dr)", "(dr, -dc)", "(-dr, dc)", "(dc, dr)", "(-dc, -dr)"]

hits, tot = np.zeros((4, 8, Craw)), np.zeros((4, 8))
base_num, base_den = np.zeros(Craw), 0
for raw_s, locs_s in samples:
    active = raw_s > 0
    base_num += active.sum(axis=(0, 1, 2, 3))
    base_den += E * A * H * W
    locs_l = locs_s.tolist()
    for e in range(E):
        for i in range(A):
            d = int(locs_l[e][i][2])
            if not 0 <= d < 4:
                continue
            for j in range(A):
                if j == i:
                    continue
                dr = locs_l[e][j][0] - locs_l[e][i][0]
                dc = locs_l[e][j][1] - locs_l[e][i][1]
                for s, f in enumerate(SYMS):
                    er, ec = f(dr, dc)
                    r, c = er + OWN_POS[0], ec + OWN_POS[1]
                    if 0 <= r < H and 0 <= c < W:
                        hits[d, s] += active[e, i, r, c]
                        tot[d, s] += 1

base = base_num / base_den
rate = hits / np.maximum(tot, 1)[..., None]
score = (rate - base) / (1 - base + 1e-9)               # lift over each channel's base rate
valid = tot >= MIN_COUNT
if not valid.any():
    info("too few agent-in-view observations; increase N_ENVS or rollout length")
else:
    masked = np.where(valid[..., None], score, -np.inf)
    k_star = int(np.argmax(masked.max(axis=(0, 1))))
    info(f"channel most aligned with other agents: {k_star} (base rate {base[k_star]:.3f})")
    suggestion = []
    for d in range(4):
        if tot[d, d] < MIN_COUNT:
            info(f"direction {d}: only {int(tot[d, d])} observations, skipped")
            suggestion.append(d)
            continue
        best = int(np.argmax(np.where(valid[d], score[d, :, k_star], -np.inf)))
        suggestion.append(best)
        ok = best == d and score[d, d, k_star] > 0.8
        check(f"direction {d} rotation correct", ok,
              f"wrapper {SYM_STR[d]} score {score[d, d, k_star]:.2f}; best {SYM_STR[best]} score {score[d, best, k_star]:.2f}; n={int(tot[d, d])}")
    if not all(s == d for d, s in enumerate(suggestion)):
        er = ", ".join(SYM_STR[s].strip("()").split(", ")[0] for s in suggestion)
        ec = ", ".join(SYM_STR[s].strip("()").split(", ")[1] for s in suggestion)
        info(f"suggested fix: er = select(.., [{er}]); ec = select(.., [{ec}])")
    top = np.argsort(-masked[np.arange(4), np.arange(4)].max(0))[:3]
    info(f"top channels under wrapper mapping: {top.tolist()}; if no channel scores near 1, agents may be encoded "
         "differently or OWN_POS is off")


# ------------------------------ summary ------------------------------ #
print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)