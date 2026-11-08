import jax
import jax.numpy as jnp
import optax
import wandb

from typing import Any, NamedTuple
from flax.training.train_state import TrainState

from src.networks.networks2 import FeedforwardCOMAAC
from src.environments.env_wrapper import Environment


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    q_taken: jnp.ndarray
    reward: jnp.ndarray
    reputation: jnp.ndarray
    log_prob: jnp.ndarray
    actor_obs: jnp.ndarray
    critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]
    info: Any


def make_train_COMA(config):
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

        network = FeedforwardCOMAAC(
            action_dim=action_dim,
            num_agents=num_agents,
            small=config["SMALL"],
            activation=config["ACTIVATION"],
        )

        rng, _rng = jax.random.split(rng)
        init_actor_obs = obs.reshape((batch_size, *obs.shape[2:]))
        init_global_map, init_agent_vec = env.reconstruct_critic_state(env_state)
        agent_id_oh_template = jnp.repeat(
            jnp.eye(num_agents)[None, :, :], num_envs, axis=0
        ).reshape((batch_size, num_agents))
        init_critic_obs = (
            init_global_map,
            init_agent_vec,
            agent_id_oh_template,
            jnp.zeros((batch_size, num_agents), dtype=jnp.int32),
        )

        network_params = network.init(_rng, init_actor_obs, init_critic_obs)

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
            agent_state, env_state, last_obs, rep, update_step, rng = runner_state
            rng, _rng_action, _rng_step = jax.random.split(rng, 3)

            obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))

            # 1. Step Feedforward Actor first to get current actions a_t
            pi = network.apply(
                agent_state.params,
                obs_batch,
                method=network.forward_actor,
            )  # type: ignore

            action = pi.sample(seed=_rng_action) #type: ignore
            log_prob = pi.log_prob(action)       #type: ignore
            probs = pi.probs                     #type: ignore

            actions_per_env = action.reshape((num_envs, num_agents))
            current_joint_actions = jnp.repeat(
                actions_per_env[:, None, :], num_agents, axis=1
            ).reshape((batch_size, num_agents))

            # 2. Step COMA Critic conditioned on S_t and current joint actions A_t
            global_map, global_agent_vec = env.reconstruct_critic_state(env_state)
            critic_obs = (
                global_map,
                global_agent_vec,
                agent_id_oh_template,
                current_joint_actions,
            )

            q_vals = network.apply(
                agent_state.params,
                critic_obs,
                method=network.forward_critic,
            )  # type: ignore

            value = (probs * q_vals).sum(axis=-1)
            q_taken = jnp.take_along_axis(q_vals, action[:, None], axis=-1).squeeze(-1) #type: ignore

            next_obs, (next_env_state, next_rep), reward, done, info = env.step(
                _rng_step, actions_per_env, (env_state, rep)
            )

            reward_flat = reward.reshape((batch_size,)).astype(jnp.float32)  # type: ignore
            done_flat = done.reshape((batch_size,)).astype(jnp.float32)      # type: ignore

            transition = Transition(
                done=done_flat,
                action=action,
                value=value,
                q_taken=q_taken,
                reward=reward_flat,
                reputation=rep.reshape((batch_size,)),
                log_prob=log_prob,
                actor_obs=obs_batch,
                critic_obs=critic_obs,
                info=info,
            )
            runner_state = (
                agent_state,
                next_env_state,
                next_obs,
                next_rep,
                update_step,
                rng,
            )
            return runner_state, transition

        def _update_step(runner_state, unused):
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            agent_state, env_state, last_obs, rep, update_step, rng = runner_state
            rng, _rng_boot = jax.random.split(rng)

            last_obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))  # type: ignore
            pi_last = network.apply(
                agent_state.params,
                last_obs_batch,
                method=network.forward_actor,
            )  # type: ignore

            boot_action = pi_last.sample(seed=_rng_boot) #type: ignore
            boot_actions_per_env = boot_action.reshape((num_envs, num_agents))
            boot_joint_actions = jnp.repeat(boot_actions_per_env[:, None, :], num_agents, axis=1).reshape((batch_size, num_agents))

            last_global_map, last_agent_vec = env.reconstruct_critic_state(env_state)
            last_critic_obs = (
                last_global_map,
                last_agent_vec,
                agent_id_oh_template,
                boot_joint_actions,
            )

            q_last = network.apply(
                agent_state.params,
                last_critic_obs,
                method=network.forward_critic,
            )  # type: ignore
            last_val = (pi_last.probs * q_last).sum(axis=-1) #type: ignore

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
                        pi, q_all_actions = network.apply(
                            params, traj_batch.actor_obs, traj_batch.critic_obs
                        )  # type: ignore
                        log_prob = pi.log_prob(traj_batch.action)

                        # Unnormalized Direct COMA Counterfactual Advantage
                        q_taken_pred = jnp.take_along_axis(q_all_actions, traj_batch.action[:, None], axis=-1,).squeeze(-1) #type: ignore
                        value_pred = (jax.lax.stop_gradient(pi.probs) * q_all_actions).sum(axis=-1)

                        adv_cf = jax.lax.stop_gradient(q_taken_pred - value_pred)

                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        loss_actor1 = ratio * adv_cf
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * adv_cf
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        # Unclipped MSE Critic Loss
                        loss_critic = 0.5 * jnp.square(q_taken_pred - targets).mean()

                        entropy = pi.entropy().mean()
                        total_loss = (
                            loss_actor
                            + config["VF_COEF"] * loss_critic
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (loss_actor, loss_critic, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(agent_state.params, traj_batch, advantages, targets)
                    agent_state = agent_state.apply_gradients(grads=grads)

                    return agent_state, (total_loss, aux)

                agent_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                total_batch_size = config["NUM_STEPS"] * batch_size
                permutation = jax.random.permutation(_rng, total_batch_size)

                batch = (traj_batch._replace(info={}), advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((total_batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
                agent_state, total_loss = jax.lax.scan(_update_minbatch, agent_state, minibatches)
                update_state = (agent_state, traj_batch, advantages, targets, rng)

                return update_state, total_loss

            update_state = (agent_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            agent_state = update_state[0]

            total_loss, (loss_actor, loss_critic, entropy) = loss_info

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
                "losses/entropy": entropy.mean(),
            }

            if isinstance(traj_batch.info, dict):
                for k, v in traj_batch.info.items():
                    if isinstance(v, jnp.ndarray) and v.size > 0:
                        metrics[f"env_info/{k}"] = v.mean()

            if config["WANDB"]:
                jax.debug.callback(_log_to_wandb, metrics)

            update_step = update_step + 1
            runner_state = (agent_state, env_state, last_obs, rep, update_step, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        runner_state = (agent_state, env_state, obs, rep, 0, _rng)
        num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train