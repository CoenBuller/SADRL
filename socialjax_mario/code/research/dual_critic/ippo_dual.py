"""Direction 3 (2026-10-18): IPPO with a team critic and an individual critic.

Copy of algorithms/IPPO/ippo_cnn_coins.py (parameter sharing only), changed in four places:
  1. DualActorCritic gives v_team and v_ind.
  2. Two reward streams per step:
       team:       r_T = the env's team reward R (shared_rewards=True)
       individual: r_I = info["agent_rewards"] (DUAL_IND_REWARD=agent), or the IA-shaped
                   r_i - IA penalty (DUAL_IND_REWARD=ia, env in team-IA mode: directions 1+3)
  3. GAE per stream; each head regresses its own return.
  4. Actor advantage: A = (1 - w) * norm(A_T) + w * norm(A_I), w = DUAL_W, each normalised per minibatch.
A fixed w on its own is close to training one critic on the blended reward; the two heads make the values
separately visible and are what a w that changes during training would need.

DUAL_MODE=blend (2026-10-19, control for 1+3): one critic on the blended reward (1 - w) * r_T + w * r_I.
DUAL_GAMMA_TEAM (2026-10-19): discount of the team stream only, e.g. 0.995 for a longer horizon; null = GAMMA.
The team head regresses the blended return, the individual head is not trained, and the actor uses norm(A_blend).
"""
import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import wandb

import socialjax
from socialjax.wrappers.baselines import SVOLogWrapper
from algorithms.utils import batchify, batchify_dict, unbatchify
from research.dual_critic.networks import DualActorCritic
from typing import NamedTuple


class DualTransition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    v_team: jnp.ndarray
    v_ind: jnp.ndarray
    r_team: jnp.ndarray
    r_ind: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def _gae(dones, values, rewards, last_val, gamma, lam):
    def step(carry, x):
        gae, next_value = carry
        done, value, reward = x
        delta = reward + gamma * next_value * (1 - done) - value
        gae = delta + gamma * lam * (1 - done) * gae
        return (gae, value), gae
    _, adv = jax.lax.scan(step, (jnp.zeros_like(last_val), last_val), (dones, values, rewards), reverse=True, unroll=16)
    return adv, adv + values


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    assert config["PARAMETER_SHARING"], "dual critic is implemented for parameter sharing only"
    assert config["ENV_KWARGS"].get("shared_rewards", False), "the team stream needs shared_rewards=True"
    ind_mode = config.get("DUAL_IND_REWARD", "agent")
    assert ind_mode in ("agent", "ia"), ind_mode
    if ind_mode == "ia":
        assert config["ENV_KWARGS"].get("inequity_aversion_team_reward", False), "DUAL_IND_REWARD=ia needs the env in team-IA mode"
    w = float(config.get("DUAL_W", 0.5))
    blend = config.get("DUAL_MODE", "dual") == "blend"
    gamma_team = float(config.get("DUAL_GAMMA_TEAM") or config["GAMMA"])  # 2026-10-19: optional longer horizon for the team stream
    assert config.get("DUAL_MODE", "dual") in ("dual", "blend"), config.get("DUAL_MODE")
    scale_keys = list(config.get("METRIC_SCALE_KEYS", []))

    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    env = SVOLogWrapper(env, replace_info=False)
    obs_shape = env.observation_space()[0].shape

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def train(rng):
        network = DualActorCritic(env.action_space().n, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        params = network.init(_rng, jnp.zeros((1, *obs_shape)))
        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        tx = optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=lr, eps=1e-5))
        train_state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)

        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))

        def _update_step(runner_state, unused):
            def _env_step(runner_state, unused):
                train_state, env_state, last_obs, update_step, rng = runner_state
                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                pi, v_team, v_ind = network.apply(train_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = [v for v in unbatchify(action, env.agents, config["NUM_ENVS"], env.num_agents).values()]

                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(_rng, config["NUM_ENVS"]), env_state, env_act)

                # Per-agent infos are (NUM_ENVS, N); flatten agent-major like obs_batch.
                to_actor = lambda x: jnp.transpose(x, (1, 0)).reshape(config["NUM_ACTORS"])
                r_env = batchify(reward, env.agents, config["NUM_ACTORS"]).squeeze()
                r_agent = to_actor(info["agent_rewards"])
                if ind_mode == "ia":
                    p = to_actor(info["ia_penalty"])
                    r_team, r_ind = r_env + p, r_agent - p
                else:
                    r_team, r_ind = r_env, r_agent
                if blend:
                    r_team = (1 - w) * r_team + w * r_ind
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                transition = DualTransition(
                    batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze(),
                    action, v_team, v_ind, r_team, r_ind, log_prob, obs_batch, info)
                return (train_state, env_state, obsv, update_step, rng), transition

            runner_state, traj = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])
            train_state, env_state, last_obs, update_step, rng = runner_state
            _, last_vt, last_vi = network.apply(
                train_state.params, jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape))
            adv_t, tgt_t = _gae(traj.done, traj.v_team, traj.r_team, last_vt, gamma_team, config["GAE_LAMBDA"])
            adv_i, tgt_i = _gae(traj.done, traj.v_ind, traj.r_ind, last_vi, config["GAMMA"], config["GAE_LAMBDA"])

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch):
                    traj, adv_t, adv_i, tgt_t, tgt_i = batch

                    def _loss_fn(params):
                        pi, v_team, v_ind = network.apply(params, traj.obs)
                        log_prob = pi.log_prob(traj.action)

                        def vloss(v, v_old, tgt):
                            v_clip = v_old + (v - v_old).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                            return 0.5 * jnp.maximum(jnp.square(v - tgt), jnp.square(v_clip - tgt)).mean()
                        norm = lambda a: (a - a.mean()) / (a.std() + 1e-8)
                        if blend:
                            value_loss = vloss(v_team, traj.v_team, tgt_t)
                            gae = norm(adv_t)
                        else:
                            value_loss = vloss(v_team, traj.v_team, tgt_t) + vloss(v_ind, traj.v_ind, tgt_i)
                            gae = (1 - w) * norm(adv_t) + w * norm(adv_i)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                        entropy = pi.entropy().mean()
                        total = loss_actor + config["VF_COEF"] * value_loss - config["ENT_COEF"] * entropy
                        return total, (value_loss, loss_actor, entropy)

                    (total, aux), grads = jax.value_and_grad(_loss_fn, has_aux=True)(train_state.params)
                    return train_state.apply_gradients(grads=grads), total

                train_state, traj, adv_t, adv_i, tgt_t, tgt_i, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                perm = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]),
                                               (traj, adv_t, adv_i, tgt_t, tgt_i))
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), batch)
                train_state, loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                return (train_state, traj, adv_t, adv_i, tgt_t, tgt_i, rng), loss

            update_state = (train_state, traj, adv_t, adv_i, tgt_t, tgt_i, rng)
            update_state, _ = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state, rng = update_state[0], update_state[-1]

            update_step = update_step + 1
            metric = jax.tree.map(lambda x: x.mean(), traj.info)
            metric["update_step"] = update_step
            metric["env_step"] = update_step * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["value_team"] = traj.v_team.mean()
            metric["value_ind"] = traj.v_ind.mean()
            metric["adv_team_ind_corr"] = jnp.corrcoef(adv_t.reshape(-1), adv_i.reshape(-1))[0, 1]
            for k in scale_keys:
                metric[k] = metric[k] * config["ENV_KWARGS"]["num_inner_steps"]
            jax.debug.callback(lambda m: wandb.log(m), metric)
            return (train_state, env_state, last_obs, update_step, rng), metric

        rng, _rng = jax.random.split(rng)
        runner_state, metric = jax.lax.scan(
            _update_step, (train_state, env_state, obsv, 0, _rng), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state, "metrics": metric}

    return train
