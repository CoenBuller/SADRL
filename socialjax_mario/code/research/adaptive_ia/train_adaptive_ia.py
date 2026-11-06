"""Demo: a PPO controller that sets inequity-aversion (alpha, beta) during IPPO training.

Design (b) from handoff/2026-10-10_1600_handoff.md. One controller iteration =
B inner IPPO runs trained from scratch in parallel (vmapped). Every K inner updates
("chunk") the controller observes each run's training statistics and picks that run's
alpha and beta for the next chunk. Its reward is the run's collective raw return
(sum over agents, unshaped), so the controller's return is the area under the
inner learning curve.

Modes:
  --mode controller                  learn alpha/beta with PPO (default)
  --mode fixed --alpha A --beta B    constant alpha/beta (0,0 = plain IPPO individual)
  --mode fixed --reward common       IPPO common reward, same budget (reference)

Run from the repo root: python -u research/adaptive_ia/train_adaptive_ia.py [flags]
"""
import argparse
import json
import os
import sys
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import flax.linen as nn
from flax.training.train_state import TrainState

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import socialjax  # noqa: E402
from algorithms.utils import ActorCritic, unbatchify  # noqa: E402
from ia_wrapper import InequityAversionWrapper  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=["controller", "fixed"], default="controller")
    p.add_argument("--reward", choices=["ia", "common"], default="ia")
    p.add_argument("--alpha", type=float, default=0.0, help="fixed mode only")
    p.add_argument("--beta", type=float, default=0.0, help="fixed mode only")
    p.add_argument("--env", default="coin_game")
    p.add_argument("--num_agents", type=int, default=2)
    # inner IPPO (paper hyperparameters except NUM_ENVS / minibatches / no LR annealing)
    p.add_argument("--num_runs", type=int, default=8, help="B: parallel inner runs")
    p.add_argument("--num_envs", type=int, default=16, help="E: envs per inner run")
    p.add_argument("--num_steps", type=int, default=1000, help="rollout length = 1 episode")
    p.add_argument("--updates_per_chunk", type=int, default=20, help="K")
    p.add_argument("--num_chunks", type=int, default=10, help="T: controller steps per inner run")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--update_epochs", type=int, default=2)
    p.add_argument("--num_minibatches", type=int, default=32)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    p.add_argument("--clip_eps", type=float, default=0.2)
    p.add_argument("--ent_coef", type=float, default=0.01)
    p.add_argument("--vf_coef", type=float, default=0.5)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--ia_gamma_lambda", type=float, default=0.99 * 0.975)
    # controller
    p.add_argument("--iterations", type=int, default=12, help="controller iterations")
    p.add_argument("--alpha_max", type=float, default=5.0)
    p.add_argument("--beta_max", type=float, default=5.0)
    p.add_argument("--ctrl_lr", type=float, default=3e-3)
    p.add_argument("--ctrl_epochs", type=int, default=8)
    p.add_argument("--ctrl_gamma", type=float, default=0.99)
    p.add_argument("--ctrl_gae_lambda", type=float, default=0.95)
    p.add_argument("--ctrl_clip_eps", type=float, default=0.2)
    p.add_argument("--ctrl_ent_coef", type=float, default=0.0)
    p.add_argument("--ctrl_init_log_std", type=float, default=0.0)
    p.add_argument("--return_scale", type=float, default=50.0, help="divides returns in controller obs/reward")
    # bookkeeping
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--wandb_mode", default=os.environ.get("WANDB_MODE", "online"))
    p.add_argument("--project", default="socialjax")
    p.add_argument("--name", default=None)
    p.add_argument("--out_dir", default="outputs/adaptive_ia")
    return p.parse_args()


# ----------------------------------------------------------------------------- inner IPPO

class RunState(NamedTuple):
    train_state: TrainState
    env_state: object
    obs: jnp.ndarray
    rng: jnp.ndarray


def make_inner(cfg):
    base = socialjax.make(cfg.env, num_agents=cfg.num_agents, num_inner_steps=cfg.num_steps,
                          shared_rewards=False, inequity_aversion=False, cnn=True, jit=True)
    env = InequityAversionWrapper(base, gamma_lambda=cfg.ia_gamma_lambda, reward_mode=cfg.reward)
    n, E = env.num_agents, cfg.num_envs
    n_actors = n * E
    obs_shape = env.observation_space()[0].shape
    network = ActorCritic(env.action_space().n, activation="relu")
    tx = optax.chain(optax.clip_by_global_norm(cfg.max_grad_norm), optax.adam(cfg.lr, eps=1e-5))
    batch_size = n_actors * cfg.num_steps
    assert batch_size % cfg.num_minibatches == 0

    def init_run(rng):
        rng, k_net, k_env = jax.random.split(rng, 3)
        params = network.init(k_net, jnp.zeros((1, *obs_shape)))
        ts = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
        obs, env_state = jax.vmap(env.reset)(jax.random.split(k_env, E))
        return RunState(ts, env_state, obs, rng)

    def update(run, alpha, beta):
        """One IPPO update (rollout of num_steps + PPO epochs) with the given alpha/beta."""

        def env_step(carry, _):
            ts, env_state, obs, rng = carry
            rng, k_act, k_step = jax.random.split(rng, 3)
            obs_batch = jnp.transpose(obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
            pi, value = network.apply(ts.params, obs_batch)
            action = pi.sample(seed=k_act)
            log_prob = pi.log_prob(action)
            env_act = [v for v in unbatchify(action, env.agents, E, n).values()]
            obs2, env_state, reward, done, info = jax.vmap(
                env.step, in_axes=(0, 0, 0, None, None)
            )(jax.random.split(k_step, E), env_state, env_act, alpha, beta)
            tr = dict(
                obs=obs_batch, action=action, value=value, log_prob=log_prob,
                reward=reward.T.reshape(-1),                      # agent-major, like obs_batch
                done=jnp.tile(done.astype(jnp.float32), n),
                ret_raw=info["ret_raw"], ret_own=info["ret_own"], ia_penalty=info["ia_penalty"],
            )
            return (ts, env_state, obs2, rng), tr

        (ts, env_state, obs, rng), traj = jax.lax.scan(
            env_step, (run.train_state, run.env_state, run.obs, run.rng), None, cfg.num_steps
        )
        last_obs = jnp.transpose(obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
        _, last_val = network.apply(ts.params, last_obs)

        def gae_step(carry, t):
            gae, next_value = carry
            delta = t["reward"] + cfg.gamma * next_value * (1 - t["done"]) - t["value"]
            gae = delta + cfg.gamma * cfg.gae_lambda * (1 - t["done"]) * gae
            return (gae, t["value"]), gae

        _, adv = jax.lax.scan(gae_step, (jnp.zeros_like(last_val), last_val),
                              {k: traj[k] for k in ("reward", "done", "value")}, reverse=True, unroll=16)
        targets = adv + traj["value"]

        def loss_fn(params, b):
            pi, value = network.apply(params, b["obs"])
            log_prob = pi.log_prob(b["action"])
            v_clip = b["value"] + (value - b["value"]).clip(-cfg.clip_eps, cfg.clip_eps)
            v_loss = 0.5 * jnp.maximum(jnp.square(value - b["targets"]),
                                       jnp.square(v_clip - b["targets"])).mean()
            ratio = jnp.exp(log_prob - b["log_prob"])
            gae = (b["adv"] - b["adv"].mean()) / (b["adv"].std() + 1e-8)
            pg = -jnp.minimum(ratio * gae, jnp.clip(ratio, 1 - cfg.clip_eps, 1 + cfg.clip_eps) * gae).mean()
            ent = pi.entropy().mean()
            return pg + cfg.vf_coef * v_loss - cfg.ent_coef * ent, (v_loss, pg, ent)

        def epoch(carry, _):
            ts, rng = carry
            rng, k = jax.random.split(rng)
            batch = {"obs": traj["obs"], "action": traj["action"], "value": traj["value"],
                     "log_prob": traj["log_prob"], "adv": adv, "targets": targets}
            batch = jax.tree.map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
            perm = jax.random.permutation(k, batch_size)
            mbs = jax.tree.map(lambda x: jnp.take(x, perm, axis=0).reshape(
                (cfg.num_minibatches, -1) + x.shape[1:]), batch)

            def mb_step(ts, b):
                grads, aux = jax.grad(loss_fn, has_aux=True)(ts.params, b)
                return ts.apply_gradients(grads=grads), aux

            ts, aux = jax.lax.scan(mb_step, ts, mbs)
            return (ts, rng), aux

        (ts, rng), _ = jax.lax.scan(epoch, (ts, rng), None, cfg.update_epochs)
        # each rollout spans exactly one episode, so the last step holds that episode's stats
        stats = {
            "collective_return": traj["ret_raw"][-1].sum(-1).mean(),        # sum over agents
            "own_coins": traj["ret_own"][-1].sum(-1).mean(),
            "inequality": jnp.abs(traj["ret_raw"][-1][:, 0] - traj["ret_raw"][-1][:, 1]).mean()
            if n == 2 else traj["ret_raw"][-1].std(-1).mean(),
            "ia_penalty": traj["ia_penalty"].sum(0).mean(),                  # per agent per episode
        }
        return RunState(ts, env_state, obs, rng), stats

    def chunk(run, alpha, beta):
        def body(run, _):
            return update(run, alpha, beta)
        return jax.lax.scan(body, run, None, cfg.updates_per_chunk)   # stats: [K]

    return env, init_run, chunk


# ----------------------------------------------------------------------------- controller

N_FEAT = 7


class Controller(nn.Module):
    init_log_std: float = 0.0

    @nn.compact
    def __call__(self, x):
        h = nn.tanh(nn.Dense(64)(x))
        h = nn.tanh(nn.Dense(64)(h))
        mean = nn.Dense(2, kernel_init=nn.initializers.orthogonal(0.01))(h)
        log_std = self.param("log_std", lambda _: jnp.full((2,), self.init_log_std))
        v = nn.tanh(nn.Dense(64)(x))
        v = nn.Dense(1)(nn.tanh(nn.Dense(64)(v)))[..., 0]
        return mean, log_std, v


def gaussian_logp(a, mean, log_std):
    return (-0.5 * jnp.square((a - mean) / jnp.exp(log_std)) - log_std - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def main():
    cfg = parse_args()
    import wandb

    steps_per_update = cfg.num_envs * cfg.num_steps
    steps_per_run = steps_per_update * cfg.updates_per_chunk * cfg.num_chunks
    tag = (f"ctrl" if cfg.mode == "controller" else
           ("common" if cfg.reward == "common" else f"fixed_a{cfg.alpha:g}_b{cfg.beta:g}"))
    name = cfg.name or f"adaptive_ia_{cfg.env}_{tag}"
    print(f"{name}: {cfg.num_runs} runs x {steps_per_run:.3g} env steps per iteration; "
          f"{cfg.iterations if cfg.mode == 'controller' else 1} iteration(s)", flush=True)
    wandb.init(project=cfg.project, entity=os.environ.get("WANDB_ENTITY") or None, mode=cfg.wandb_mode,
               name=name, group="adaptive_ia_demo", tags=["adaptive_ia", cfg.mode, cfg.reward],
               config=vars(cfg) | {"steps_per_run": steps_per_run})

    env, init_run, chunk = make_inner(cfg)
    ctrl = Controller(cfg.ctrl_init_log_std)
    rng = jax.random.PRNGKey(cfg.seed)
    rng, k = jax.random.split(rng)
    ctrl_state = TrainState.create(
        apply_fn=ctrl.apply, params=ctrl.init(k, jnp.zeros((1, N_FEAT))),
        tx=optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.ctrl_lr)))
    s = cfg.return_scale

    def features(t, prev, prev2, alpha, beta):
        return jnp.stack([
            jnp.full_like(prev["collective_return"], t / cfg.num_chunks),
            prev["collective_return"] / s,
            (prev["collective_return"] - prev2["collective_return"]) / s,
            prev["own_coins"] / s,
            prev["inequality"] / s,
            alpha / cfg.alpha_max,
            beta / cfg.beta_max,
        ], axis=-1)

    def episode(ctrl_params, rng, deterministic):
        """Train B inner runs from scratch; the controller (or fixed values) sets alpha/beta."""
        B = cfg.num_runs
        rng, k = jax.random.split(rng)
        runs = jax.vmap(init_run)(jax.random.split(k, B))
        zero = {"collective_return": jnp.zeros(B), "own_coins": jnp.zeros(B),
                "inequality": jnp.zeros(B), "ia_penalty": jnp.zeros(B)}

        def step(carry, t):
            runs, prev, prev2, alpha, beta, rng = carry
            rng, k = jax.random.split(rng)
            x = features(t, prev, prev2, alpha, beta)
            mean, log_std, value = ctrl.apply(ctrl_params, x)
            raw = jnp.where(deterministic, mean, mean + jnp.exp(log_std) * jax.random.normal(k, mean.shape))
            logp = gaussian_logp(raw, mean, log_std)
            if cfg.mode == "controller":
                alpha = cfg.alpha_max * jax.nn.sigmoid(raw[:, 0])
                beta = cfg.beta_max * jax.nn.sigmoid(raw[:, 1])
            else:
                alpha = jnp.full(B, cfg.alpha)
                beta = jnp.full(B, cfg.beta)
            runs, st = jax.vmap(chunk)(runs, alpha, beta)            # st: [B, K]
            cur = jax.tree.map(lambda v: v.mean(axis=1), st)
            reward = cur["collective_return"] / s
            out = dict(x=x, raw=raw, logp=logp, value=value, reward=reward,
                       alpha=alpha, beta=beta, curve=st)
            return (runs, cur, prev, alpha, beta, rng), out

        init = (runs, zero, zero, jnp.zeros(B), jnp.zeros(B), rng)
        _, traj = jax.lax.scan(step, init, jnp.arange(cfg.num_chunks))
        return traj                                                   # leaves: [T, B, ...]

    episode_jit = jax.jit(episode, static_argnums=2)

    @jax.jit
    def ctrl_update(ctrl_state, traj):
        def gae_step(gae, t):
            r, v, v_next, last = t
            delta = r + cfg.ctrl_gamma * v_next * (1 - last) - v
            gae = delta + cfg.ctrl_gamma * cfg.ctrl_gae_lambda * (1 - last) * gae
            return gae, gae
        T = cfg.num_chunks
        v_next = jnp.concatenate([traj["value"][1:], jnp.zeros_like(traj["value"][:1])], 0)
        last = jnp.zeros((T, 1)).at[-1].set(1.0) * jnp.ones_like(traj["value"])
        _, adv = jax.lax.scan(gae_step, jnp.zeros_like(traj["value"][0]),
                              (traj["reward"], traj["value"], v_next, last), reverse=True)
        targets = adv + traj["value"]
        adv_n = (adv - adv.mean()) / (adv.std() + 1e-8)

        def loss_fn(params):
            mean, log_std, value = ctrl.apply(params, traj["x"])
            logp = gaussian_logp(traj["raw"], mean, log_std)
            ratio = jnp.exp(logp - traj["logp"])
            pg = -jnp.minimum(ratio * adv_n, jnp.clip(ratio, 1 - cfg.ctrl_clip_eps, 1 + cfg.ctrl_clip_eps) * adv_n).mean()
            v_loss = 0.5 * jnp.square(value - targets).mean()
            ent = (log_std + 0.5 * jnp.log(2 * jnp.pi * jnp.e)).sum()
            return pg + 0.5 * v_loss - cfg.ctrl_ent_coef * ent, (pg, v_loss, ent)

        def ep(state, _):
            grads, aux = jax.grad(loss_fn, has_aux=True)(state.params)
            return state.apply_gradients(grads=grads), aux

        ctrl_state, aux = jax.lax.scan(ep, ctrl_state, None, cfg.ctrl_epochs)
        return ctrl_state, jax.tree.map(lambda a: a[-1], aux)

    os.makedirs(cfg.out_dir, exist_ok=True)
    history = []
    n_iters = cfg.iterations if cfg.mode == "controller" else 1
    for it in range(n_iters + (1 if cfg.mode == "controller" else 0)):
        final_eval = cfg.mode == "controller" and it == n_iters
        rng, k = jax.random.split(rng)
        t0 = time.time()
        traj = jax.device_get(episode_jit(ctrl_state.params, k, final_eval))
        dt = time.time() - t0
        log = {"iteration": it, "iter_seconds": dt,
               "env_steps_per_s": cfg.num_runs * steps_per_run / dt}
        if cfg.mode == "controller" and not final_eval:
            ctrl_state, (pg, v_loss, ent) = ctrl_update(ctrl_state, jax.tree.map(jnp.asarray, traj))
            log.update(ctrl_pg_loss=float(pg), ctrl_v_loss=float(v_loss), ctrl_entropy=float(ent))
        curve = traj["curve"]["collective_return"]                     # [T, B, K]
        log.update({
            "collective_return_final": float(curve[-1, :, -1].mean()),
            "collective_return_auc": float(curve.mean()),
            "own_coins_final": float(traj["curve"]["own_coins"][-1, :, -1].mean()),
            "inequality_final": float(traj["curve"]["inequality"][-1, :, -1].mean()),
        })
        for t in range(cfg.num_chunks):
            log[f"alpha/chunk{t}"] = float(traj["alpha"][t].mean())
            log[f"beta/chunk{t}"] = float(traj["beta"][t].mean())
        prefix = "eval/" if final_eval else ""
        wandb.log({prefix + k: v for k, v in log.items()})
        print(json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in log.items()
                          if "/" not in k}), flush=True)
        history.append({"iteration": it, "final_eval": final_eval,
                        "alpha": traj["alpha"].tolist(), "beta": traj["beta"].tolist(),
                        "curve": {k: v.tolist() for k, v in traj["curve"].items()}})

    # Learning curve of the last episode (mean over runs), on an env-step axis, for plotting.
    last = history[-1]["curve"]
    cr = np.array(last["collective_return"]).transpose(1, 0, 2).reshape(cfg.num_runs, -1)
    table = wandb.Table(columns=["env_step", "collective_return_mean", "collective_return_std"])
    for u in range(cr.shape[1]):
        table.add_data((u + 1) * steps_per_update, float(cr[:, u].mean()), float(cr[:, u].std()))
    wandb.log({"final_curve": table})
    path = os.path.join(cfg.out_dir, f"{name}_seed{cfg.seed}.json")
    with open(path, "w") as f:
        json.dump({"config": vars(cfg), "steps_per_update": steps_per_update, "history": history}, f)
    print("saved", path, flush=True)
    wandb.finish()


if __name__ == "__main__":
    main()
