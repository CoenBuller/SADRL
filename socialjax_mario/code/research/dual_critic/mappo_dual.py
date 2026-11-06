"""Direction 3 (2026-10-18): MAPPO with a central critic that has a team head and a per-agent head.

Copy of algorithms/MAPPO/mappo_cnn_coins.py, changed in the same places as ippo_dual.py:
  - DualSmallCritic(world state) -> v_team (scalar: "is this joint state good?") and v_ind (one value per agent:
    "given that, how much of it is agent i's"); the per-agent head sees v_team through stop_gradient.
  - Team stream r_T = env team reward; individual stream r_I = info["agent_rewards"] (or IA-shaped, DUAL_IND_REWARD=ia).
  - Actor advantage A = (1 - w) * norm(A_T) + w * norm(A_I) with w = DUAL_W.
The runner_state layout is unchanged, so algorithms.MAPPO._runner saves and evaluates the actor as before.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import wandb

import socialjax
from socialjax.wrappers.baselines import MAPPOWorldStateWrapper, SVOLogWrapper
from algorithms.utils import SmallActor, Actor as FullActor, batchify_dict, batchify_numpy, unbatchify
from research.dual_critic.networks import DualSmallCritic, DualCritic
from research.dual_critic.ippo_dual import _gae


class DualMAPPOTransition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    v_team: jnp.ndarray
    v_ind: jnp.ndarray
    r_team: jnp.ndarray
    r_ind: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    world_state: jnp.ndarray
    agent_id: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    assert config["ENV_KWARGS"].get("shared_rewards", False), "the team stream needs shared_rewards=True"
    ind_mode = config.get("DUAL_IND_REWARD", "agent")
    assert ind_mode in ("agent", "ia"), ind_mode
    if ind_mode == "ia":
        assert config["ENV_KWARGS"].get("inequity_aversion_team_reward", False), "DUAL_IND_REWARD=ia needs the env in team-IA mode"
    w = float(config.get("DUAL_W", 0.5))
    scale_keys = list(config.get("METRIC_SCALE_KEYS", []))
    n_agents = env.num_agents

    config["NUM_ACTORS"] = n_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_EPS"] / n_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]
    env = MAPPOWorldStateWrapper(env)
    env = SVOLogWrapper(env, replace_info=False)
    obs_shape = env.observation_space()[0].shape
    agent_id = jnp.arange(config["NUM_ACTORS"]) // config["NUM_ENVS"]  # actors are agent-major

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def env_world_states(obs):
        return jnp.transpose(obs, (0, 2, 3, 1, 4)).reshape(config["NUM_ENVS"], *obs_shape[:-1], -1)

    def world_states(obs):
        ws = env_world_states(obs)
        return jnp.tile(ws[None], (n_agents, 1, 1, 1, 1)).reshape(-1, *ws.shape[1:])

    # 2026-10-20: NETWORK_SIZE "small" (default) or "full" (IPPO's torso), as in algorithms/MAPPO/mappo_cnn_cleanup.py
    full = config.get("NETWORK_SIZE", "small") == "full"
    Actor, Critic = (FullActor, DualCritic) if full else (SmallActor, DualSmallCritic)

    def train(rng):
        actor_network = Actor(env.action_space().n, activation=config["ACTIVATION"])
        critic_network = Critic(n_agents, activation=config["ACTIVATION"])
        rng, _ra, _rc = jax.random.split(rng, 3)
        actor_params = actor_network.init(_ra, jnp.zeros((1, *obs_shape)))
        critic_params = critic_network.init(_rc, jnp.zeros((1, *obs_shape[:-1], obs_shape[-1] * n_agents)))
        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        make_tx = lambda: optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=lr, eps=1e-5))
        actor_state = TrainState.create(apply_fn=actor_network.apply, params=actor_params, tx=make_tx())
        critic_state = TrainState.create(apply_fn=critic_network.apply, params=critic_params, tx=make_tx())

        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))

        def critic_values(params, ws, ids):
            v_team, v_ind = critic_network.apply(params, ws)
            return v_team, jnp.take_along_axis(v_ind, ids[:, None], axis=1)[:, 0]

        def _update_step(update_runner_state, unused):
            runner_state, update_steps = update_runner_state

            def _env_step(runner_state, step_idx):
                train_states, env_state, last_obs, last_done, rng = runner_state
                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                pi = actor_network.apply(train_states[0].params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = [v for v in unbatchify(action, env.agents, config["NUM_ENVS"], n_agents).values()]
                ws = world_states(last_obs)
                v_team, v_ind = critic_values(train_states[1].params, ws, agent_id)

                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(_rng, config["NUM_ENVS"]), env_state, env_act)

                to_actor = lambda x: jnp.transpose(x, (1, 0)).reshape(config["NUM_ACTORS"])
                r_env = batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                r_agent = to_actor(info["agent_rewards"])
                if ind_mode == "ia":
                    p = to_actor(info["ia_penalty"])
                    r_team, r_ind = r_env + p, r_agent - p
                else:
                    r_team, r_ind = r_env, r_agent
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze()
                # 2026-10-20: the world state is stored once per env step; the transition keeps its row index
                # (actor k is env k % NUM_ENVS), which cuts the world-state memory by n_agents.
                ws_index = step_idx * config["NUM_ENVS"] + jnp.arange(config["NUM_ACTORS"]) % config["NUM_ENVS"]
                transition = DualMAPPOTransition(
                    jnp.tile(done["__all__"], n_agents), last_done, action, v_team, v_ind, r_team, r_ind,
                    log_prob, obs_batch, ws_index, agent_id, info)
                return (train_states, env_state, obsv, done_batch, rng), (transition, ws[: config["NUM_ENVS"]])

            runner_state, (traj, ws_buffer) = jax.lax.scan(_env_step, runner_state, jnp.arange(config["NUM_STEPS"]))
            ws_buffer = ws_buffer.reshape(-1, *ws_buffer.shape[2:])
            train_states, env_state, last_obs, last_done, rng = runner_state
            last_vt, last_vi = critic_values(train_states[1].params, world_states(last_obs), agent_id)
            adv_t, tgt_t = _gae(traj.done, traj.v_team, traj.r_team, last_vt, config["GAMMA"], config["GAE_LAMBDA"])
            adv_i, tgt_i = _gae(traj.done, traj.v_ind, traj.r_ind, last_vi, config["GAMMA"], config["GAE_LAMBDA"])

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch):
                    actor_state, critic_state = train_states
                    traj, adv_t, adv_i, tgt_t, tgt_i = batch

                    def _actor_loss_fn(params):
                        pi = actor_network.apply(params, traj.obs)
                        log_prob = pi.log_prob(traj.action)
                        norm = lambda a: (a - a.mean()) / (a.std() + 1e-8)
                        gae = (1 - w) * norm(adv_t) + w * norm(adv_i)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                        entropy = pi.entropy().mean()
                        return loss_actor - config["ENT_COEF"] * entropy, entropy

                    def _critic_loss_fn(params):
                        v_team, v_ind = critic_values(params, jnp.take(ws_buffer, traj.world_state, axis=0), traj.agent_id)

                        def vloss(v, v_old, tgt):
                            v_clip = v_old + (v - v_old).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            return 0.5 * jnp.maximum(jnp.square(v - tgt), jnp.square(v_clip - tgt)).mean()
                        return config["VF_COEF"] * (vloss(v_team, traj.v_team, tgt_t) + vloss(v_ind, traj.v_ind, tgt_i))

                    (a_loss, _), a_grads = jax.value_and_grad(_actor_loss_fn, has_aux=True)(actor_state.params)
                    c_loss, c_grads = jax.value_and_grad(_critic_loss_fn)(critic_state.params)
                    return (actor_state.apply_gradients(grads=a_grads), critic_state.apply_gradients(grads=c_grads)), a_loss + c_loss

                train_states, traj, adv_t, adv_i, tgt_t, tgt_i, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                perm = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]),
                                               (traj, adv_t, adv_i, tgt_t, tgt_i))
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), batch)
                train_states, loss = jax.lax.scan(_update_minbatch, train_states, minibatches)
                return (train_states, traj, adv_t, adv_i, tgt_t, tgt_i, rng), loss

            update_state = (train_states, traj, adv_t, adv_i, tgt_t, tgt_i, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_states, rng = update_state[0], update_state[-1]

            update_steps = update_steps + 1
            metric = jax.tree.map(lambda x: x.mean(), traj.info)
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["value_team"] = traj.v_team.mean()
            metric["value_ind"] = traj.v_ind.mean()
            metric["adv_team_ind_corr"] = jnp.corrcoef(adv_t.reshape(-1), adv_i.reshape(-1))[0, 1]
            for k in scale_keys:
                metric[k] = metric[k] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(lambda m: wandb.log(m), metric)
            return ((train_states, env_state, last_obs, last_done, rng), update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = ((actor_state, critic_state), env_state, obsv,
                        jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng)
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train
