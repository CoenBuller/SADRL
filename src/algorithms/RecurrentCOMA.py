import jax
import jax.numpy as jnp
import optax
import wandb

from typing import Any, NamedTuple
from flax.training.train_state import TrainState

from src.networks.networks2 import RecurrentMAPPOAC, LSTM, MAPPOResult
from src.environments.env_wrapper import Environment


class Transition(NamedTuple):
    done: jnp.ndarray                                                       # (batch_size,)
    action: jnp.ndarray                                                     # (batch_size,)
    last_action: jnp.ndarray                                                # (batch_size,)
    value: jnp.ndarray                                                      # (batch_size,)
    q_taken: jnp.ndarray                                                    # (batch_size,)
    reward: jnp.ndarray                                                     # (batch_size,)
    log_prob: jnp.ndarray                                                   # (batch_size,)
    actor_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]                 # (obs_batch, dones_batch, last_action_batch)
    critic_obs: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]   # (global_map, global_agent_vec, agent_id_oh, current_joint_actions)
    true_xy: jnp.ndarray                                                    # (batch_size, 2)
    reputation: jnp.ndarray                                                 # (batch_size,)
    info: Any


def make_train_recurrent_COMA(config):
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
        # 1. Initialize environment
        obs, (env_state, rep), rng = env.initialize(rng)

        # 2. Initialize RecurrentMAPPOAC network
        network = RecurrentMAPPOAC(
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=config["HIDDEN_DIM"],
            activation=config["ACTIVATION"],
            small=config["SMALL"],
        )

        rng, _rng = jax.random.split(rng)

        init_hidden = LSTM.initialize_carry(
            batch_size=batch_size, hidden_dim=config["HIDDEN_DIM"]
        )

        init_actor_obs = (
            obs.reshape((batch_size, *obs.shape[2:])),
            jnp.zeros((batch_size,), dtype=jnp.float32),
            jnp.zeros((batch_size,), dtype=jnp.int32),
        )

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

        init_actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], init_actor_obs)
        init_critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], init_critic_obs)

        network_params = network.init(
            _rng,
            init_hidden,
            init_actor_obs_t,
            init_critic_obs_t,
        )

        # 3. Optimizer Setup
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

        # 4. Inner Environment Step (Rollout Collection)
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

            # Step 4a: Run Actor first to sample current actions a_t
            actor_obs = (obs_batch, dones_batch, last_action_batch)
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], actor_obs)

            new_hidden, pi, _ = network.apply( # type: ignore
                agent_state.params,
                hidden,
                actor_obs_t,
                method=network.forward_actor,
            )  # type: ignore

            action_seq = pi.sample(seed=_rng_action)               # type: ignore
            action = action_seq.squeeze(0)                         # (batch_size,)
            log_prob = pi.log_prob(action_seq).squeeze(0)          # type: ignore
            probs = pi.probs.squeeze(0)                            # type: ignore

            actions_per_env = action.reshape((num_envs, num_agents))
            current_joint_actions = jnp.repeat(
                actions_per_env[:, None, :], num_agents, axis=1
            ).reshape((batch_size, num_agents))

            # Step 4b: Run COMA Critic conditioned on S_t and current joint actions A_t
            global_map, global_agent_vec = env.reconstruct_critic_state(env_state)
            critic_obs = (
                global_map,
                global_agent_vec,
                agent_id_oh_template,
                current_joint_actions,
            )
            critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], critic_obs)

            q_vals_seq = network.apply(
                agent_state.params,
                critic_obs_t,
                method=network.forward_critic,
            )  # type: ignore
            q_vals = q_vals_seq.squeeze(0)                         # type: ignore

            value = (probs * q_vals).sum(axis=-1)
            q_taken = jnp.take_along_axis(
                q_vals, action[:, None], axis=-1
            ).squeeze(-1)

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
                q_taken=q_taken,
                reward=reward_flat,
                log_prob=log_prob,
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
                new_hidden,
                actions_per_env,
                done_flat,
                update_step,
                rng,
            )
            return runner_state, transition

        # 5. Outer Update Step (Rollout + GAE + PPO Minibatch Epochs)
        def _update_step(runner_state, unused):
            runner_state, traj_batch = jax.lax.scan(
                _env_step,
                runner_state,
                None,
                config["NUM_STEPS"],
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

            rng, _rng_boot = jax.random.split(rng)

            last_obs_batch = last_obs.reshape((batch_size, *last_obs.shape[2:]))  # type: ignore

            actor_obs = (
                last_obs_batch,
                last_done.reshape((batch_size,)),
                last_action.reshape((batch_size,)),
            )
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], actor_obs)

            _, pi_last, _ = network.apply( # type: ignore
                agent_state.params,
                hidden,
                actor_obs_t,
                method=network.forward_actor,
            )  # type: ignore

            boot_action = pi_last.sample(seed=_rng_boot).squeeze(0)  # type: ignore
            boot_actions_per_env = boot_action.reshape((num_envs, num_agents))
            boot_joint_actions = jnp.repeat(
                boot_actions_per_env[:, None, :], num_agents, axis=1
            ).reshape((batch_size, num_agents))

            last_global_map, last_agent_vec = env.reconstruct_critic_state(env_state)
            critic_obs = (
                last_global_map,
                last_agent_vec,
                agent_id_oh_template,
                boot_joint_actions,
            )
            critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], critic_obs)

            q_last = network.apply(
                agent_state.params,
                critic_obs_t,
                method=network.forward_critic,
            ).squeeze(0)  # type: ignore

            last_val = (pi_last.probs.squeeze(0) * q_last).sum(axis=-1)  # type: ignore

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

                        out: MAPPOResult = network.apply(
                            params,
                            init_hidden_mb,
                            traj_batch.actor_obs,
                            traj_batch.critic_obs,
                        )  # type: ignore

                        pi = out.pi
                        log_prob = pi.log_prob(traj_batch.action)

                        # 1. Unnormalized Direct COMA Counterfactual Advantage
                        q_taken_pred = jnp.take_along_axis(
                            out.q_all_actions,
                            traj_batch.action[:, :, None],
                            axis=-1,
                        ).squeeze(-1)
                        value_pred = (
                            jax.lax.stop_gradient(pi.probs) * out.q_all_actions
                        ).sum(axis=-1)

                        adv_cf = jax.lax.stop_gradient(q_taken_pred - value_pred)

                        # Clipped Policy Objective
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

                        # 2. Unclipped MSE Critic Loss
                        loss_critic = 0.5 * jnp.square(q_taken_pred - targets).mean()

                        # 3. Self-Localization Loss
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

                # Shuffle only along axis 1 (Batch) to keep axis 0 (Time) intact for BPTT
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                # Reshape into (NUM_MINIBATCHES, Time, Minibatch_Size, ...)
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

            total_loss, (
                loss_actor,
                loss_critic,
                entropy,
                loss_loc,
            ) = loss_info

            def _ep_ret(acc, x):
                reward, done = x
                acc = acc + reward
                return acc * (1.0 - done), acc * done

            _, finished = jax.lax.scan(
                _ep_ret,
                jnp.zeros_like(traj_batch.reward[0]),
                (traj_batch.reward, traj_batch.done),
            )
            episode_return = finished.sum() / jnp.maximum(
                traj_batch.done.sum(), 1.0
            )

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

        # 6. Run Training Loop
        rng, _rng = jax.random.split(rng)
        initial_update_step = 0
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
            initial_update_step,
            _rng,
        )

        num_updates = config["TOTAL_TIMESTEPS"] // (
            config["NUM_STEPS"] * num_envs
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, num_updates
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train