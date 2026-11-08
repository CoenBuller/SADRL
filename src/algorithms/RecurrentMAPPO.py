import jax
import jax.numpy as jnp
import optax
import wandb

from typing import Any, NamedTuple
from flax.training.train_state import TrainState

from src.networks.networks2 import RecurrentMAPPOAC, LSTM
from src.algorithms.MAPPO import make_world_state
from src.environments.env_wrapper import Environment


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    reputation: jnp.ndarray
    log_prob: jnp.ndarray
    actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    world_state: jnp.ndarray
    info: Any


def make_train_recurrent(config):
    env = Environment(
        common_reward=config["COMMON_REWARD"],
        env_name=config["ENV_NAME"],
        n_agents=config["N_AGENTS"],
        n_envs=config["N_ENVS"],
        seed=config["SEED"],
        reputation=config["REPUTATION"],
        own_pos=tuple(config["OWN_POS"]),
        decay_coef=config["REP_DECAY"],
    )

    num_agents = env.n_agents
    num_envs = env.n_envs
    batch_size = num_envs * num_agents
    action_dim = env.env.action_space(0).n
    rng = jax.random.PRNGKey(config["SEED"])

    def train(rng=rng):
        obs, (env_state, rep), rng = env.initialize(rng)

        network = RecurrentMAPPOAC(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            small=config["SMALL"],
            activation=config["ACTIVATION"],
        )

        rng, _rng = jax.random.split(rng)
        init_hidden = LSTM.initialize_carry(batch_size, config["HIDDEN_DIM"])

        init_actor_obs = (
            jnp.zeros((1, batch_size, *obs.shape[2:])),
            jnp.zeros((1, batch_size), dtype=jnp.float32),
            jnp.zeros((1, batch_size), dtype=jnp.int32),
        )
        init_world_state = make_world_state(obs, num_envs, num_agents)[None, ...]

        network_params = network.init(_rng, init_hidden, init_actor_obs, init_world_state)

        num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
            return config["LR"] * frac

        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        tx = optax.adam(learning_rate=lr, eps=1e-5)

        agent_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        def _env_step(runner_state, unused):
            (
                agent_state,
                env_state,
                last_obs,
                rep,
                hidden,
                last_action,
                last_done,
                ep_returns,
                update_step,
                rng,
            ) = runner_state
            rng, _rng_action, _rng_step = jax.random.split(rng, 3)

            obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))
            world_batch = make_world_state(last_obs, num_envs, num_agents)
            dones_batch = last_done.reshape((batch_size,)).astype(jnp.float32)
            last_action_batch = last_action.reshape((batch_size,))

            actor_obs = (obs_batch, dones_batch, last_action_batch)
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], actor_obs)
            world_batch_t = world_batch[None, ...]

            new_hidden, pi, value_seq = network.apply(
                agent_state.params, hidden, actor_obs_t, world_batch_t
            )  # type: ignore

            action_seq = pi.sample(seed=_rng_action)
            action = action_seq.squeeze(0)
            log_prob = pi.log_prob(action_seq).squeeze(0)
            value = value_seq.squeeze(0)

            actions_per_env = action.reshape((num_envs, num_agents))
            next_obs, (next_env_state, next_rep), reward, done, info = env.step(
                _rng_step, actions_per_env, (env_state, rep)
            )

            reward_flat = reward.reshape((batch_size,)).astype(jnp.float32)  # type: ignore
            done_flat = done.reshape((batch_size,)).astype(jnp.float32)      # type: ignore

            new_ep_returns = ep_returns + reward_flat
            finished_ep_returns = new_ep_returns * done_flat
            ep_returns = new_ep_returns * (1.0 - done_flat)
            info["finished_ep_returns"] = finished_ep_returns

            transition = Transition(
                done=done_flat,
                action=action,
                value=value,
                reward=reward_flat,
                reputation=rep.reshape((batch_size,)),
                log_prob=log_prob,
                actor_obs=actor_obs,
                world_state=world_batch,
                info=info,
            )

            runner_state = (
                agent_state,
                next_env_state,
                next_obs,
                next_rep,
                new_hidden,
                actions_per_env,
                done_flat,
                ep_returns,
                update_step,
                rng,
            )
            return runner_state, transition

        def _update_step(runner_state, unused):
            initial_hstate = runner_state[4]

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            (
                agent_state,
                env_state,
                last_obs,
                rep,
                hidden,
                last_action,
                last_done,
                ep_returns,
                update_step,
                rng,
            ) = runner_state

            last_obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))  # type: ignore
            last_world_batch = make_world_state(last_obs, num_envs, num_agents)

            actor_obs = (
                last_obs_batch,
                last_done.reshape((batch_size,)),
                last_action.reshape((batch_size,)),
            )
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], actor_obs)
            last_world_t = last_world_batch[None, ...]

            _, _, last_val_seq = network.apply(
                agent_state.params, hidden, actor_obs_t, last_world_t
            )  # type: ignore
            last_val = last_val_seq.squeeze(0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1.0 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                return advantages, advantages + traj_batch.value

            advantages, targets = _calculate_gae(traj_batch, last_val)

            def _update_epoch(update_state, unused):
                def _update_minbatch(agent_state, batch_info):
                    init_hstate_mb, traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(params, init_hstate_mb, traj_batch, gae):
                        _, pi, _ = network.apply(
                            params,
                            init_hstate_mb,
                            traj_batch.actor_obs,
                            traj_batch.world_state,
                        )  # type: ignore
                        log_prob = pi.log_prob(traj_batch.action)
                        logratio = log_prob - traj_batch.log_prob
                        ratio = jnp.exp(logratio)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()
                        entropy = pi.entropy().mean()
                        approx_kl = ((ratio - 1.0) - logratio).mean()

                        return loss_actor - config["ENT_COEF"] * entropy, (loss_actor, entropy, approx_kl)

                    def _critic_loss_fn(params, init_hstate_mb, traj_batch, targets):
                        _, _, value = network.apply(
                            params,
                            init_hstate_mb,
                            traj_batch.actor_obs,
                            traj_batch.world_state,
                        )  # type: ignore
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        loss_critic = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        return config["VF_COEF"] * loss_critic, loss_critic

                    (actor_total, (loss_actor, entropy, approx_kl)), actor_grads = jax.value_and_grad(
                        _actor_loss_fn, has_aux=True
                    )(agent_state.params, init_hstate_mb, traj_batch, advantages)

                    (critic_total, loss_critic), critic_grads = jax.value_and_grad(
                        _critic_loss_fn, has_aux=True
                    )(agent_state.params, init_hstate_mb, traj_batch, targets)

                    clip_fn = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])
                    actor_grads, _ = clip_fn.update(actor_grads, optax.EmptyState())
                    critic_grads, _ = clip_fn.update(critic_grads, optax.EmptyState())
                    combined_grads = jax.tree_util.tree_map(lambda a, c: a + c, actor_grads, critic_grads)

                    agent_state = agent_state.apply_gradients(grads=combined_grads)
                    total_loss = actor_total + critic_total
                    return agent_state, (total_loss, (loss_actor, loss_critic, entropy, approx_kl))

                agent_state, init_hstate, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                permutation = jax.random.permutation(_rng, batch_size)

                shuffled_hstate = jnp.take(init_hstate, permutation, axis=0)
                minibatches_hstate = jnp.reshape(
                    shuffled_hstate, (config["NUM_MINIBATCHES"], -1, config["HIDDEN_DIM"])
                )

                seq_batch = (traj_batch._replace(info={}), advantages, targets)
                shuffled_seq = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), seq_batch
                )
                minibatches_seq = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                        ),
                        0,
                        1,
                    ),
                    shuffled_seq,
                )

                minibatches = (minibatches_hstate, *minibatches_seq)
                agent_state, loss_info = jax.lax.scan(_update_minbatch, agent_state, minibatches)
                update_state = (agent_state, init_hstate, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (agent_state, initial_hstate, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            agent_state = update_state[0]

            total_loss, (loss_actor, loss_critic, entropy, approx_kl) = loss_info

            finished_returns = traj_batch.info["finished_ep_returns"]
            num_done = jnp.maximum(traj_batch.done.sum(), 1.0)
            episode_return = finished_returns.sum() / num_done
            team_return = episode_return * num_agents

            def _log_to_wandb(metrics_dict):
                if wandb.run is not None:
                    wandb.log(metrics_dict)

            metrics = {
                "update_step": update_step,
                "env_step": (update_step + 1) * config["NUM_STEPS"] * num_envs,
                "rollout/episode_return": episode_return,
                "rollout/team_return": team_return,
                "rollout/mean_abs_reputation": jnp.abs(traj_batch.reputation).mean(),
                "rollout/std_reputation": traj_batch.reputation.std(),
                "rollout/mean_reward": traj_batch.reward.mean(),
                "rollout/std_reward": traj_batch.reward.std(),
                "rollout/mean_value": traj_batch.value.mean(),
                "losses/total_loss": total_loss.mean(),
                "losses/actor_loss": loss_actor.mean(),
                "losses/critic_loss": loss_critic.mean(),
                "losses/entropy": entropy.mean(),
                "losses/approx_kl": approx_kl.mean(),
            }

            if isinstance(traj_batch.info, dict):
                for k, v in traj_batch.info.items():
                    if k != "finished_ep_returns" and isinstance(v, jnp.ndarray) and v.size > 0:
                        metrics[f"env_info/{k}"] = v.mean()

            if config["WANDB"]:
                jax.debug.callback(_log_to_wandb, metrics)

            update_step = update_step + 1
            runner_state = (
                agent_state,
                env_state,
                last_obs,
                rep,
                hidden,
                last_action,
                last_done,
                ep_returns,
                update_step,
                rng,
            )
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        init_hidden = LSTM.initialize_carry(batch_size, config["HIDDEN_DIM"])
        init_last_action = jnp.zeros((num_envs, num_agents), dtype=jnp.int32)
        init_last_done = jnp.zeros((batch_size,), dtype=jnp.float32)
        init_ep_returns = jnp.zeros((batch_size,), dtype=jnp.float32)

        runner_state = (
            agent_state,
            env_state,
            obs,
            rep,
            init_hidden,
            init_last_action,
            init_last_done,
            init_ep_returns,
            0,
            _rng,
        )

        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train