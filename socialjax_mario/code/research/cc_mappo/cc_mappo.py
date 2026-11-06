"""CC-MAPPO (2026-10-20, workbook rounds 6-7): MAPPO whose actor is trained on a per-agent credit A^i_t.

Copy of algorithms/MAPPO/mappo_cnn_cleanup.py (with its 2026-10-20 world-state fix), changed in these places:
  - V_psi(s) is MAPPO's critic, unchanged; it gives the advantage hat A_t and the lambda-return G_t (GAE).
  - A credit critic (research/cc_mappo/networks.py, IPPO's torso, own Adam) gives each agent its own credit:
      CREDIT_MODE=coma : Q(s, a^-i, i)[a^i] - sum_a' pi_old(a') Q[a']          target of Q[a^i]: G_t
      CREDIT_MODE=qret : G_t - sum_a' pi_old(a') Q(s, a^-i, i)[a']             target of Q[a^i]: G_t
      CREDIT_MODE=loo  : G_t - V(s, a^-i, i)                                    target of V:      G_t
      CREDIT_MODE=local: U(o^i, V-bar, i)[a^i] - sum_a' pi_old(a') U[a']        target of U[a^i]: hat A_t
  - The credit critic is evaluated during the rollout with the same parameters as V_psi's values, and both are
    updated afterwards on the same minibatches. Credit is used from the first update (round 6, C.3), normalised with one
    mean over the minibatch and scaled by the std of hat A_t (CREDIT_NORM=team, CC theory workbook T.4, 2026-10-21; was its own
    std, round 7 B.4, CREDIT_NORM=self).
The runner_state layout is unchanged, so algorithms.MAPPO._runner saves and evaluates the actor as before (round 14 adds the
gate state as a third element of the outer scan carry; out["runner_state"][0] is still the runner state).
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
from flax.training.train_state import TrainState
import wandb

import socialjax
from socialjax.wrappers.baselines import MAPPOWorldStateWrapper, SVOLogWrapper
from algorithms.utils.networks import mappo_networks
from algorithms.utils import batchify_dict, batchify_numpy, unbatchify
from research.dual_critic.ippo_dual import _gae
from research.cc_mappo.credit import CREDIT_MODES, CREDIT_NORMS, relative_others, baseline, credit, critic_prediction, normalise_credit
from research.cc_mappo.credit import CREDIT_GATES, explained_variance, ev_smooth, gate_weight, gated_credit  # round 14 gate
from research.cc_mappo.networks import CentralCreditCritic, LocalCreditCritic

CLEAN_ACTION, STAY_ACTION = 8, 6  # socialjax/environments/cleanup/clean_up.py Actions


class CCTransition(NamedTuple):
    global_done: jnp.ndarray
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    probs: jnp.ndarray       # pi_old(.|o^i), for the counterfactual baseline
    cq: jnp.ndarray          # credit critic output at rollout time: Q or U (9), or V(s, a^-i) (1)
    obs: jnp.ndarray
    world_state: jnp.ndarray  # row index into the per-env world-state and joint-action buffers
    agent_id: jnp.ndarray
    agent_reward: jnp.ndarray
    info: jnp.ndarray


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    mode = config["CREDIT_MODE"]
    assert mode in CREDIT_MODES, mode
    norm = config.get("CREDIT_NORM", "team")
    assert norm in CREDIT_NORMS, norm
    gate = config.get("CREDIT_GATE", "none")  # round 14 (B.2, B.4): "ev" mixes hat A and the credit by the credit critic's EV
    assert gate in CREDIT_GATES, gate
    assert gate == "none" or mode in ("qret", "loo"), "the EV gate is a mixed baseline: only for G-based credits (qret, loo)"
    gate_beta, gate_efull = float(config.get("GATE_BETA", 0.9)), float(config.get("GATE_EFULL", 0.5))
    assert config["ENV_KWARGS"].get("shared_rewards", False), "CC-MAPPO is defined on the team reward"
    scale_keys = list(config.get("METRIC_SCALE_KEYS", []))
    n_agents = env.num_agents
    n_actions = env.action_space().n

    config["NUM_ACTORS"] = n_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    config["MINIBATCH_SIZE"] = config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    config["CLIP_EPS"] = config["CLIP_EPS"] / n_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]
    env = MAPPOWorldStateWrapper(env)
    env = SVOLogWrapper(env, replace_info=False)
    obs_shape = env.observation_space()[0].shape
    agent_id = jnp.arange(config["NUM_ACTORS"]) // config["NUM_ENVS"]  # actors are agent-major
    env_id = jnp.arange(config["NUM_ACTORS"]) % config["NUM_ENVS"]

    # [SADRL fix 2026-11-05] actor and critic from mappo_networks (CNN_SIZE / FC_SIZE / CNN_LAYERS, falling back to
    # NETWORK_SIZE as before), so CC-MAPPO can use the same networks as the MAPPO grid (round 12, C.3).
    Actor, Critic = mappo_networks(config)

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * frac

    def env_world_states(obs):
        return jnp.transpose(obs, (0, 2, 3, 1, 4)).reshape(config["NUM_ENVS"], *obs_shape[:-1], -1)

    def train(rng):
        actor_network = Actor(n_actions, activation=config["ACTIVATION"])
        critic_network = Critic(activation=config["ACTIVATION"])
        if mode == "local":
            cc_network = LocalCreditCritic(n_agents, n_actions, activation=config["ACTIVATION"])
        else:
            cc_network = CentralCreditCritic(n_agents, n_actions, 1 if mode == "loo" else n_actions,
                                             activation=config["ACTIVATION"])

        def cc_apply(params, obs, world_state, joint, agent, v_bar):
            if mode == "local":
                return cc_network.apply(params, obs, v_bar, agent)
            return cc_network.apply(params, world_state, relative_others(joint, agent), agent)

        rng, _ra, _rc, _rq = jax.random.split(rng, 4)
        ws_shape = (*obs_shape[:-1], obs_shape[-1] * n_agents)
        actor_params = actor_network.init(_ra, jnp.zeros((1, *obs_shape)))
        critic_params = critic_network.init(_rc, jnp.zeros((1, *ws_shape)))
        cc_params = cc_network.init(_rq, *(
            (jnp.zeros((1, *obs_shape)), jnp.zeros((1,)), jnp.zeros((1,), jnp.int32)) if mode == "local" else
            (jnp.zeros((1, *ws_shape)), jnp.zeros((1, n_agents - 1), jnp.int32), jnp.zeros((1,), jnp.int32))))
        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        make_tx = lambda: optax.chain(optax.clip_by_global_norm(config["MAX_GRAD_NORM"]), optax.adam(learning_rate=lr, eps=1e-5))
        actor_state = TrainState.create(apply_fn=actor_network.apply, params=actor_params, tx=make_tx())
        critic_state = TrainState.create(apply_fn=critic_network.apply, params=critic_params, tx=make_tx())
        cc_state = TrainState.create(apply_fn=cc_network.apply, params=cc_params, tx=make_tx())

        rng, _rng = jax.random.split(rng)
        obsv, env_state = jax.vmap(env.reset)(jax.random.split(_rng, config["NUM_ENVS"]))

        def _update_step(update_runner_state, unused):
            runner_state, update_steps, gate_ev = update_runner_state

            def _env_step(runner_state, step_idx):
                train_states, env_state, last_obs, last_done, rng = runner_state
                actor_state, critic_state, cc_state = train_states
                rng, _rng = jax.random.split(rng)
                obs_batch = jnp.transpose(last_obs, (1, 0, 2, 3, 4)).reshape(-1, *obs_shape)
                pi = actor_network.apply(actor_state.params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = [v for v in unbatchify(action, env.agents, config["NUM_ENVS"], n_agents).values()]

                ws_env = env_world_states(last_obs)
                ws = jnp.tile(ws_env[None], (n_agents, 1, 1, 1, 1)).reshape(-1, *ws_shape)
                value = critic_network.apply(critic_state.params, ws).reshape(config["NUM_ACTORS"])
                joint_env = action.reshape(n_agents, config["NUM_ENVS"]).T  # (NUM_ENVS, N)
                cq = cc_apply(cc_state.params, obs_batch, ws, joint_env[env_id], agent_id, value)

                rng, _rng = jax.random.split(rng)
                obsv, env_state, reward, done, info = jax.vmap(env.step)(
                    jax.random.split(_rng, config["NUM_ENVS"]), env_state, env_act)
                agent_reward = jnp.transpose(info["agent_rewards"], (1, 0)).reshape(config["NUM_ACTORS"])
                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze()
                ws_index = step_idx * config["NUM_ENVS"] + env_id
                transition = CCTransition(
                    jnp.tile(done["__all__"], n_agents), last_done, action, value,
                    batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze(), log_prob, pi.probs, cq,
                    obs_batch, ws_index, agent_id, agent_reward, info)
                return (train_states, env_state, obsv, done_batch, rng), (transition, ws_env, joint_env)

            runner_state, (traj, ws_buffer, joint_buffer) = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"]))
            ws_buffer = ws_buffer.reshape(-1, *ws_shape)            # (NUM_STEPS * NUM_ENVS, H, W, C * N)
            joint_buffer = joint_buffer.reshape(-1, n_agents)       # (NUM_STEPS * NUM_ENVS, N)
            train_states, env_state, last_obs, last_done, rng = runner_state

            last_ws = jnp.tile(env_world_states(last_obs)[None], (n_agents, 1, 1, 1, 1)).reshape(-1, *ws_shape)
            last_val = critic_network.apply(train_states[1].params, last_ws).reshape(config["NUM_ACTORS"])
            advantages, targets = _gae(traj.done, traj.value, traj.reward, last_val, config["GAMMA"], config["GAE_LAMBDA"])
            credits = credit(mode, traj.cq, traj.probs, traj.action, targets)
            if gate == "ev":  # round 14: w = 0 (plain MAPPO) until the credit critic explains the return, then the credit
                gate_ev = ev_smooth(gate_ev, explained_variance(targets, critic_prediction(mode, traj.cq, traj.action)), gate_beta)
                gate_w = gate_weight(gate_ev, gate_efull)
                actor_credits = gated_credit(advantages, credits, gate_w)
            else:
                actor_credits = credits

            def _update_epoch(update_state, unused):
                def _update_minbatch(train_states, batch):
                    actor_state, critic_state, cc_state = train_states
                    traj, advantages, targets, credits = batch

                    def _actor_loss_fn(params):
                        pi = actor_network.apply(params, traj.obs)
                        log_prob = pi.log_prob(traj.action)
                        gae = normalise_credit(credits, advantages, norm)
                        ratio = jnp.exp(log_prob - traj.log_prob)
                        loss_actor = -jnp.minimum(
                            ratio * gae,
                            jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1) - (log_prob - traj.log_prob)).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        return loss_actor - config["ENT_COEF"] * entropy, (entropy, approx_kl, clip_frac)

                    def _critic_loss_fn(params):
                        value = critic_network.apply(params, jnp.take(ws_buffer, traj.world_state, axis=0))
                        v_clip = traj.value + (value - traj.value).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        return config["VF_COEF"] * 0.5 * jnp.maximum(
                            jnp.square(value - targets), jnp.square(v_clip - targets)).mean()

                    def _cc_loss_fn(params):
                        q = cc_apply(params, traj.obs, jnp.take(ws_buffer, traj.world_state, axis=0),
                                     jnp.take(joint_buffer, traj.world_state, axis=0), traj.agent_id, traj.value)
                        target = advantages if mode == "local" else targets
                        return config["VF_COEF"] * 0.5 * jnp.square(critic_prediction(mode, q, traj.action) - target).mean()

                    (a_loss, (entropy, approx_kl, clip_frac)), a_grads = jax.value_and_grad(_actor_loss_fn, has_aux=True)(actor_state.params)
                    c_loss, c_grads = jax.value_and_grad(_critic_loss_fn)(critic_state.params)
                    q_loss, q_grads = jax.value_and_grad(_cc_loss_fn)(cc_state.params)
                    train_states = (actor_state.apply_gradients(grads=a_grads),
                                    critic_state.apply_gradients(grads=c_grads),
                                    cc_state.apply_gradients(grads=q_grads))
                    return train_states, {"actor_loss": a_loss, "value_loss": c_loss, "credit_critic_loss": q_loss,
                                          "entropy": entropy, "approx_kl": approx_kl, "clip_frac": clip_frac}

                train_states, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                perm = jax.random.permutation(_rng, batch_size)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]),
                                               (traj, advantages, targets, actor_credits))
                batch = jax.tree_util.tree_map(lambda x: jnp.take(x, perm, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])), batch)
                train_states, losses = jax.lax.scan(_update_minbatch, train_states, minibatches)
                return (train_states, rng), losses

            (train_states, rng), losses = jax.lax.scan(_update_epoch, (train_states, rng), None, config["UPDATE_EPOCHS"])

            update_steps = update_steps + 1
            metric = jax.tree.map(lambda x: x.mean(), traj.info)
            metric.update(jax.tree.map(lambda x: x.mean(), losses))
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]
            for k in scale_keys:
                metric[k] = metric[k] * config["ENV_KWARGS"]["num_inner_steps"]
            metric.update(credit_metrics(traj, advantages, credits, targets))
            metric["target_mean"] = targets.mean()
            if gate == "ev":
                metric["gate_w"] = gate_w
                metric["gate_ev_smooth"] = gate_ev
            jax.debug.callback(lambda m: wandb.log(m), metric)
            return ((train_states, env_state, last_obs, last_done, rng), update_steps, gate_ev), metric

        def credit_metrics(traj, advantages, credits, targets):
            """Round 6, section 7: efficiency gap, credit by behaviour, credit per agent per episode, consistency."""
            T, E = config["NUM_STEPS"], config["NUM_ENVS"]
            by_agent = credits.reshape(T, n_agents, E)
            gap = by_agent.sum(axis=1) - advantages.reshape(T, n_agents, E)[:, 0, :]
            masked = lambda m: jnp.sum(credits * m) / jnp.maximum(jnp.sum(m), 1.0)
            m = {
                "credit_mean": credits.mean(), "credit_std": credits.std(),
                "efficiency_gap_mean": gap.mean(), "efficiency_gap_std": gap.std(),
                "credit_on_clean": masked(traj.action == CLEAN_ACTION),
                "credit_on_stay": masked(traj.action == STAY_ACTION),
                "credit_on_eat": masked(traj.agent_reward > 0),
                "clean_fraction": jnp.mean(traj.action == CLEAN_ACTION),
                "value_mean": traj.value.mean(),
            }
            per_agent = by_agent.mean(axis=(0, 2)) * config["ENV_KWARGS"]["num_inner_steps"]  # summed over an episode
            for i in range(n_agents):
                m[f"credit_episode_agent_{i}"] = per_agent[i]
            # [SADRL fix 2026-11-05] Logging only (round 12, C.4; round 13, L.2 and T.1):
            # raw credit vs shared advantage spread and correlation; credit critic's explained variance on its own
            # target (G_t, or hat A_t for local); per agent id, reward and clean actions per episode (agent-major).
            a_flat, c_flat = advantages.reshape(-1), credits.reshape(-1)
            m["advantage_std"] = a_flat.std()
            m["credit_adv_corr"] = jnp.mean((a_flat - a_flat.mean()) * (c_flat - c_flat.mean())) / (a_flat.std() * c_flat.std() + 1e-8)
            cc_target = advantages if mode == "local" else targets
            pred = critic_prediction(mode, traj.cq, traj.action)
            m["credit_critic_explained_var"] = 1.0 - jnp.var(cc_target - pred) / (jnp.var(cc_target) + 1e-8)
            per_episode = config["ENV_KWARGS"]["num_inner_steps"] / T
            rew = traj.agent_reward.reshape(T, n_agents, E).sum(0).mean(-1) * per_episode
            cln = (traj.action == CLEAN_ACTION).reshape(T, n_agents, E).sum(0).mean(-1) * per_episode
            for i in range(n_agents):
                m[f"reward_agent{i}"] = rew[i]
                m[f"clean_agent{i}"] = cln[i]
            if mode in ("coma", "qret"):
                m["q_consistency"] = (baseline(traj.cq, traj.probs) - traj.value).mean()
            return m

        rng, _rng = jax.random.split(rng)
        runner_state = ((actor_state, critic_state, cc_state), env_state, obsv,
                        jnp.zeros((config["NUM_ACTORS"]), dtype=bool), _rng)
        runner_state, metric = jax.lax.scan(_update_step, (runner_state, 0, jnp.float32(0.0)), None, config["NUM_UPDATES"])
        return {"runner_state": runner_state}

    return train
