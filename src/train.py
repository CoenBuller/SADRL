import jax
import jax.numpy as jnp
import optax
import wandb

from typing import NamedTuple
from flax.training.train_state import TrainState

from src.algorithms.networks import ActorCritic
from src.environments.wrappers import Environment

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    reputation: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


def make_train_IPPO(config):
    # Initialize environment and rng key
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
    action_dim = env.env.action_space(0).n
    rng = jax.random.PRNGKey(config["SEED"])

    obs_shape = env.env.observation_space()[0].shape  #type:ignore : (H, W, C)

    def train(rng=rng):
        # 1. Reset first, so the real obs shape is known
        obs, (env_state, rep), rng = env.initialize(rng)

        # 2. Init network from the real obs shape
        network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((num_envs * num_agents, *obs.shape[2:]))   # (B, H, W, C_actual)
        network_params = network.init(_rng, init_x)

        # 3. Optimizer
        if config["ANNEAL_LR"]:
            num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
            total_gradient_steps = num_updates * config["UPDATE_EPOCHS"] * config["NUM_MINIBATCHES"]
            lr_schedule = optax.linear_schedule(
                init_value=config["LR"], end_value=0.0, transition_steps=total_gradient_steps
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

        agent_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)
 
        # 3. Inner Rollout Step 
        def _env_step(runner_state, unused):
            agent_state, env_state, last_obs, rep, update_step, rng = runner_state
            rng, _rng_action, _rng_step = jax.random.split(rng, 3)

            # Flatten across environments and agents for forward pass
            obs_batch = last_obs.reshape((num_envs * num_agents, *last_obs.shape[2:]))
            pi, value = network.apply(agent_state.params, obs_batch)                # Forward pass with agents     

            # Squeeze value if shape is (B, 1) to avoid silent broadcasting issues
            if value.ndim > 1 and value.shape[-1] == 1:                                 # type: ignore
                value = value.squeeze(-1)                                               # type: ignore
            
            action = pi.sample(seed=_rng_action)                                        # Pick action
            log_prob = pi.log_prob(action)                                              # Convert to log probabilities, used for GAE and loss

            # Actions as (num_envs, num_agents) array
            actions_per_env = action.reshape((num_envs, num_agents))

            # Step parallel environments (wrapper step is already batched)
            next_obs, (next_env_state, next_rep), reward, done, info = env.step(
                _rng_step, actions_per_env, (env_state, rep)
            )

            # Flatten rewards and terminations to (num_envs * num_agents,)
            reward_flat = reward.reshape(-1)    # type: ignore
            done_flat = done.reshape(-1)        # type: ignore

            transition = Transition(
                done=done_flat,
                action=action,
                value=value,                                                            # type: ignore , Static typing artifact in flax about the return type of value
                reward=reward_flat,
                reputation=rep.reshape(-1),
                log_prob=log_prob,
                obs=obs_batch,
                info=info,
            )
            runner_state = (agent_state, next_env_state, next_obs, next_rep, update_step, rng)
            return runner_state, transition

        # 4. Outer Update Step (Rollout + GAE + Update) 
        def _update_step(runner_state, unused):
            # collect rollout
            runner_state, traj_batch = jax.lax.scan( # Apllies the batched _env_step over the runner state for "NUM_STEPS"
                _env_step,          
                runner_state, 
                None, 
                config["NUM_STEPS"]
                )

            # COMPUTE GAE (Reverse Scan)
            agent_state, env_state, last_obs, rep, update_step, rng = runner_state
            last_obs_batch = last_obs.reshape((num_envs * num_agents, *last_obs.shape[2:])) #type: ignore
            _, last_val = network.apply(agent_state.params, last_obs_batch)
            if last_val.ndim > 1 and last_val.shape[-1] == 1:                                   # type: ignore
                last_val = last_val.squeeze(-1)                                                 # type: ignore

            # Generalized Advantage Estimation (GAE)
            def _calculate_gae(traj_batch, last_val):
                
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1.0 - done) - value        # delta = r + γ·V' · (1-done) - V
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - done) * gae   # gae = delta + γλ·(1-done)·gae
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

            # Update Epoch (Mini-batch SGD)
            def _update_epoch(update_state, unused):
                def _update_minbatch(agent_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        pi, value = network.apply(params, traj_batch.obs)
                        log_prob = pi.log_prob(traj_batch.action)

                        # Clipped policy objective
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)             # Relative P of taking a_t in s_t under new policy compare to old policy
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)               # Normalized GAE
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()  # Take minimum of normal and clipped ratio*gae

                        # Value objective
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        loss_critic = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Entropy bonus
                        entropy = pi.entropy().mean()

                        total_loss = loss_actor + config["VF_COEF"] * loss_critic - config["ENT_COEF"] * entropy
                        return total_loss, (loss_actor, loss_critic, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (total_loss, aux), grads = grad_fn(agent_state.params, traj_batch, advantages, targets) # grads are the gradients over the params of the network
                    agent_state = agent_state.apply_gradients(grads=grads)  # update the network with gradient descent using the grads
                    return agent_state, (total_loss, aux)

                agent_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Flatten batch over steps, environments, and agents
                batch_size = config["NUM_STEPS"] * num_envs * num_agents
                permutation = jax.random.permutation(_rng, batch_size)
                
                batch = (traj_batch._replace(info={}), advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                
                # Chunk into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                agent_state, total_loss = jax.lax.scan(_update_minbatch, agent_state, minibatches)
                update_state = (agent_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (agent_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            agent_state = update_state[0]


           # -------------------- W&B LOGGING -------------------- #
            total_loss, (loss_actor, loss_critic, entropy) = loss_info

            def _ep_ret(acc, x):
                reward, done = x
                acc = acc + reward
                return acc * (1.0 - done), acc * done      # reset at episode end, emit finished return

            _, finished = jax.lax.scan(
                _ep_ret, jnp.zeros_like(traj_batch.reward[0]), (traj_batch.reward, traj_batch.done)
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

            # Add any scalar stats present in traj_batch.info
            if isinstance(traj_batch.info, dict):
                for k, v in traj_batch.info.items():
                    if isinstance(v, jnp.ndarray) and v.size > 0:
                        metrics[f"env_info/{k}"] = v.mean()

            if config["WANDB"]:
                jax.debug.callback(_log_to_wandb, metrics)
            # ----------------------------------------------------- #

            update_step = update_step + 1
            runner_state = (agent_state, env_state, last_obs, rep, update_step, rng)
            return runner_state, metrics

        # 5. Run Total Updates 
        rng, _rng = jax.random.split(rng)
        initial_update_step = 0
        runner_state = (agent_state, env_state, obs, rep, initial_update_step, _rng)
        
        num_updates = config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, num_updates)
        return {"runner_state": runner_state, "metrics": metrics}

    return train