import jax
import jax.numpy as jnp
import optax
import wandb

from typing import Any, NamedTuple
from flax.training.train_state import TrainState

from src.networks.networks2 import RecurrentOnlyMAPPOAC, LSTM, RecurrentOnlyResult
from src.environments.env_wrapper import Environment


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    last_action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]
    true_xy: jnp.ndarray
    reputation: jnp.ndarray
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

        network = RecurrentOnlyMAPPOAC(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            activation=config["ACTIVATION"],
            small=config["SMALL"],
        )

        rng, _rng = jax.random.split(rng)
        init_hidden = LSTM.initialize_carry(batch_size, config["HIDDEN_DIM"])

        init_actor_obs = (
            obs.reshape((batch_size, *obs.shape[2:])),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.int32),
        )

        init_global_map, init_agent_vec = env.reconstruct_critic_state(env_state)
        agent_id_oh_template = jnp.repeat(
            jnp.eye(num_agents)[None, :, :], num_envs, axis=0
        ).reshape((batch_size, num_agents))
        init_critic_obs = (init_global_map, init_agent_vec, agent_id_oh_template)

        init_actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], init_actor_obs)
        init_critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], init_critic_obs)

        network_params = network.init(
            _rng, init_hidden, init_actor_obs_t, init_critic_obs_t
        )

        if config["ANNEAL_LR"]:
            num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
            total_gradient_steps = (
                num_updates * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
            )
            lr_schedule = optax.linear_schedule(
                init_value=config["LR"],
                end_value=0.0,
                transition_steps=total_gradient_steps,
            )
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=lr_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )

        agent_state = TrainState.create(
            apply_fn=network.apply, params=network_params, tx=tx
        )

        def _env_step(runner_state, unused):
            (
                agent_state,
                env_state,
                last_obs,
                rep,
                hidden,
                last_action,
                last_done,
                update_step,
                rng,
            ) = runner_state
            rng, _rng_action, _rng_step = jax.random.split(rng, 3)

            obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))
            dones_batch = last_done.reshape((batch_size,)).astype(jnp.float32)
            last_action_batch = last_action.reshape((batch_size,))

            global_map, global_agent_vec = env.reconstruct_critic_state(env_state)

            actor_obs = (obs_batch, dones_batch, last_action_batch)
            critic_obs = (global_map, global_agent_vec, agent_id_oh_template)

            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], actor_obs)
            critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], critic_obs)

            out: RecurrentOnlyResult = network.apply(
                agent_state.params, hidden, actor_obs_t, critic_obs_t
            )  # type: ignore

            action_seq = out.pi.sample(seed=_rng_action)
            action = action_seq.squeeze(0)
            log_prob = out.pi.log_prob(action_seq).squeeze(0)
            value = out.value.squeeze(0)

            actions_per_env = action.reshape((num_envs, num_agents))
            next_obs, (next_env_state, next_rep), reward, done, info = env.step(
                _rng_step, actions_per_env, (env_state, rep)
            )

            true_xy = env.extract_normalized_xy(env_state)
            done_flat = done.reshape((batch_size,)).astype(jnp.float32)      # type: ignore
            reward_flat = reward.reshape((batch_size,)).astype(jnp.float32)  # type: ignore

            transition = Transition(
                done=done_flat,
                action=action,
                last_action=last_action_batch,
                value=value,
                reward=reward_flat,
                log_prob=log_prob, #type: ignore
                actor_obs=actor_obs,
                critic_obs=critic_obs,
                true_xy=true_xy,
                reputation=rep.reshape((batch_size,)),
                info=info,
            )

            runner_state = (
                agent_state,
                next_env_state,
                next_obs,
                next_rep,
                out.hidden,
                actions_per_env,
                done_flat,
                update_step,
                rng,
            )
            return runner_state, transition

        def _update_step(runner_state, unused):
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
                update_step,
                rng,
            ) = runner_state

            last_obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))  # type: ignore
            last_global_map, last_agent_vec = env.reconstruct_critic_state(env_state)

            actor_obs = (
                last_obs_batch,
                last_done.reshape((batch_size,)),
                last_action.reshape((batch_size,)),
            )
            critic_obs = (last_global_map, last_agent_vec, agent_id_oh_template)

            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], actor_obs)
            critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], critic_obs)

            out_last: RecurrentOnlyResult = network.apply(
                agent_state.params, hidden, actor_obs_t, critic_obs_t
            )  # type: ignore
            last_val = out_last.value.squeeze(0)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = (
                        reward + config["GAMMA"] * next_value * (1.0 - done) - value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1.0 - done)
                        * gae
                    )
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
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        mb_size = traj_batch.action.shape[1]
                        init_hidden_mb = LSTM.initialize_carry(
                            mb_size, config["HIDDEN_DIM"]
                        )

                        out: RecurrentOnlyResult = network.apply(
                            params,
                            init_hidden_mb,
                            traj_batch.actor_obs,
                            traj_batch.critic_obs,
                        )  # type: ignore

                        pi = out.pi
                        value = out.value
                        log_prob = pi.log_prob(traj_batch.action)

                        # Standard GAE Actor Loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae_norm = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae_norm
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae_norm
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        # Unclipped MSE Critic Loss
                        loss_critic = 0.5 * jnp.square(value - targets).mean()

                        # Auxiliary Self-Localization Loss
                        loss_loc = jnp.square(
                            out.pred_xy - traj_batch.true_xy
                        ).mean()

                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * loss_critic
                            - config["ENT_COEF"] * entropy
                            + 0.5 * loss_loc
                        )
                        return total_loss, (
                            loss_actor,
                            loss_critic,
                            entropy,
                            loss_loc,
                        )

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(
                        agent_state.params, traj_batch, advantages, targets
                    )
                    agent_state = agent_state.apply_gradients(grads=grads)
                    return agent_state, (total_loss, aux)

                agent_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch._replace(info={}), advantages, targets)

                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
                            + list(x.shape[2:]),
                        ),
                        0,
                        1,
                    ),
                    shuffled_batch,
                )
                agent_state, total_loss = jax.lax.scan(
                    _update_minbatch, agent_state, minibatches
                )
                update_state = (agent_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (agent_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            agent_state = update_state[0]

            total_loss, (loss_actor, loss_critic, entropy, loss_loc) = loss_info

            def _ep_ret(acc, x):
                reward, done = x
                acc = acc + reward
                return acc * (1.0 - done), acc * done

            _, finished = jax.lax.scan(
                _ep_ret,
                jnp.zeros_like(traj_batch.reward[0]),
                (traj_batch.reward, traj_batch.done),
            )
            episode_return = finished.sum() / jnp.maximum(traj_batch.done.sum(), 1.0)

            def _log_to_wandb(metrics_dict):
                if wandb.run is not None:
                    wandb.log(metrics_dict)

            metrics = {
                "update_step": update_step,
                "env_step": update_step * config["NUM_STEPS"] * num_envs,
                "rollout/episode_return": episode_return,
                "rollout/mean_abs_reputation": jnp.abs(traj_batch.reputation).mean(),
                "rollout/std_reputation": traj_batch.reputation.std(),
                "rollout/mean_reward": traj_batch.reward.mean(),
                "rollout/std_reward": traj_batch.reward.std(),
                "rollout/mean_value": traj_batch.value.mean(),
                "rollout/std_value": traj_batch.value.std(),
                "losses/total_loss": total_loss.mean(),
                "losses/actor_loss": loss_actor.mean(),
                "losses/critic_loss": loss_critic.mean(),
                "losses/loc_loss": loss_loc.mean(),
                "losses/entropy": entropy.mean(),
            }

            if isinstance(traj_batch.info, dict):
                for k, v in traj_batch.info.items():
                    if isinstance(v, jnp.ndarray) and v.size > 0:
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
                update_step,
                rng,
            )
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        init_hidden = LSTM.initialize_carry(batch_size, config["HIDDEN_DIM"])
        init_last_action = jnp.zeros((num_envs, num_agents), dtype=jnp.int32)
        init_last_done = jnp.zeros((batch_size,), dtype=jnp.float32)

        runner_state = (
            agent_state,
            env_state,
            obs,
            rep,
            init_hidden,
            init_last_action,
            init_last_done,
            0,
            _rng,
        )

        num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train