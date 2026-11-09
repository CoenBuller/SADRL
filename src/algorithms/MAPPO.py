import jax
import jax.numpy as jnp
import optax
import wandb

from typing import Any, NamedTuple
from flax.training.train_state import TrainState

from src.networks.networks2 import MAPPOAC
from src.environments.env_wrapper import Environment


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    reputation: jnp.ndarray
    log_prob: jnp.ndarray
    obs: tuple[jnp.ndarray, jnp.ndarray]
    info: Any


def make_world_state(obs: jnp.ndarray, num_envs: int, num_agents: int) -> jnp.ndarray:
    h, w, c = obs.shape[2:]
    joint_obs = jnp.transpose(obs, (0, 2, 3, 1, 4)).reshape((num_envs, h, w, num_agents * c))
    return jnp.repeat(joint_obs, num_agents, axis=0)


def make_train_MAPPO(config):
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

        network = MAPPOAC(
            action_dim=action_dim,
            small=config["SMALL"],
            activation=config["ACTIVATION"],
        )

        rng, _rng = jax.random.split(rng)
        init_x_obs = jnp.zeros((batch_size, *obs.shape[2:]))
        init_x_world = make_world_state(obs, num_envs, num_agents)
        network_params = network.init(_rng, (init_x_obs, init_x_world))

        num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)

        def linear_schedule(count):
            frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / num_updates
            return config["LR"] * frac

        lr = linear_schedule if config["ANNEAL_LR"] else config["LR"]
        tx = optax.adam(learning_rate=lr, eps=1e-5)

        agent_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        def _env_step(runner_state, unused):
            agent_state, env_state, last_obs, rep, ep_returns, returned_ep_returns, update_step, rng = runner_state
            rng, _rng_action, _rng_step = jax.random.split(rng, 3)

            obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))
            world_batch = make_world_state(last_obs, num_envs, num_agents)

            pi, value = network.apply(agent_state.params, (obs_batch, world_batch))  # type: ignore

            action = pi.sample(seed=_rng_action)
            log_prob = pi.log_prob(action)

            actions_per_env = action.reshape((num_envs, num_agents))
            next_obs, (next_env_state, next_rep), reward, done, info = env.step(
                _rng_step, actions_per_env, (env_state, rep)
            )

            reward_flat = reward.reshape((batch_size,)).astype(jnp.float32)  # type: ignore
            done_flat = done.reshape((batch_size,)).astype(jnp.float32)      # type: ignore

            new_ep_returns = ep_returns + reward_flat
            new_returned_ep_returns = jnp.where(done_flat > 0.5, new_ep_returns, returned_ep_returns)
            ep_returns = new_ep_returns * (1.0 - done_flat)

            transition = Transition(
                done=done_flat,
                action=action,
                value=value,
                reward=reward_flat,
                reputation=rep.reshape((batch_size,)),
                log_prob=log_prob,
                obs=(obs_batch, world_batch),
                info=info,
            )
            runner_state = (agent_state, next_env_state, next_obs, next_rep, ep_returns, new_returned_ep_returns, update_step, rng)
            return runner_state, transition

        def _update_step(runner_state, unused):
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            agent_state, env_state, last_obs, rep, ep_returns, returned_ep_returns, update_step, rng = runner_state
            last_obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))  # type: ignore
            last_world_batch = make_world_state(last_obs, num_envs, num_agents)
            _, last_val = network.apply(agent_state.params, (last_obs_batch, last_world_batch))  # type: ignore

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
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(params, traj_batch, gae):
                        pi, _ = network.apply(params, traj_batch.obs)  # type: ignore
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

                    def _critic_loss_fn(params, traj_batch, targets):
                        _, value = network.apply(params, traj_batch.obs)  # type: ignore
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        loss_critic = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        return config["VF_COEF"] * loss_critic, loss_critic

                    (actor_total, (loss_actor, entropy, approx_kl)), actor_grads = jax.value_and_grad(
                        _actor_loss_fn, has_aux=True
                    )(agent_state.params, traj_batch, advantages)

                    (critic_total, loss_critic), critic_grads = jax.value_and_grad(
                        _critic_loss_fn, has_aux=True
                    )(agent_state.params, traj_batch, targets)

                    clip_fn = optax.clip_by_global_norm(config["MAX_GRAD_NORM"])
                    actor_grads, _ = clip_fn.update(actor_grads, optax.EmptyState())
                    critic_grads, _ = clip_fn.update(critic_grads, optax.EmptyState())
                    combined_grads = jax.tree_util.tree_map(lambda a, c: a + c, actor_grads, critic_grads)

                    agent_state = agent_state.apply_gradients(grads=combined_grads)
                    total_loss = actor_total + critic_total
                    return agent_state, (total_loss, (loss_actor, loss_critic, entropy, approx_kl))

                agent_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)

                total_samples = config["NUM_STEPS"] * batch_size
                permutation = jax.random.permutation(_rng, total_samples)

                batch = (traj_batch._replace(info={}), advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((total_samples,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)

                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                agent_state, loss_info = jax.lax.scan(_update_minbatch, agent_state, minibatches)
                update_state = (agent_state, traj_batch, advantages, targets, rng)
                return update_state, loss_info

            update_state = (agent_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            agent_state = update_state[0]

            total_loss, (loss_actor, loss_critic, entropy, approx_kl) = loss_info

            episode_return = returned_ep_returns.mean()
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
                "rollout/std_value": traj_batch.value.std(),
                "losses/total_loss": total_loss.mean(),
                "losses/actor_loss": loss_actor.mean(),
                "losses/critic_loss": loss_critic.mean(),
                "losses/entropy": entropy.mean(),
                "losses/approx_kl": approx_kl.mean(),
            }

            if isinstance(traj_batch.info, dict):
                for k, v in traj_batch.info.items():
                    if isinstance(v, jnp.ndarray) and v.size > 0:
                        metrics[f"env_info/{k}"] = v.mean()

            if config["WANDB"]:
                jax.debug.callback(_log_to_wandb, metrics)

            update_step = update_step + 1
            runner_state = (agent_state, env_state, last_obs, rep, ep_returns, returned_ep_returns, update_step, rng)
            return runner_state, metrics

        rng, _rng = jax.random.split(rng)
        init_ep_returns = jnp.zeros((batch_size,), dtype=jnp.float32)
        init_returned_ep_returns = jnp.zeros((batch_size,), dtype=jnp.float32)
        runner_state = (agent_state, env_state, obs, rep, init_ep_returns, init_returned_ep_returns, 0, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train