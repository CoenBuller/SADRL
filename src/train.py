import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from socialjax import make
from typing import NamedTuple
from flax.linen.initializers import orthogonal, constant
from flax.training.train_state import TrainState

from algorithms.IPPO import ActorCritic
from environments.wrappers import Environment

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    reputation: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict

def make_train(config):
    env = Environment(env_name=config["ENV_NAME"], n_agents=config["N_AGENTS"], n_envs=config["N_ENVS"], seed=config["SEED"])
    num_agents = env.n_agents
    num_envs = env.n_envs
    action_dim = env.env.action_space(0).n
    rng = jax.random.PRNGKey(config["SEED"])

    def train(rng=rng):
        # --- 1. Init Network & Optimizer ---
        network = ActorCritic(action_dim=action_dim)
        rng, _rng = jax.random.split(rng)
        init_x = jnp.zeros((num_envs * num_agents, *env.env.observation_space(0).shape))
        network_params = network.init(_rng, init_x)
        
        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(config["LR"], eps=1e-5),
        )
        train_state = TrainState.create(apply_fn=network.apply, params=network_params, tx=tx)

        # --- 2. Vectorized Reset ---
        obs, env_state, rng = env.initialize(rng)

        # --- 3. Inner Rollout Step ---
        def _env_step(runner_state, unused):
            train_state, env_state, last_obs, rng = runner_state
            rng, _rng_action, _rng_step = jax.random.split(rng, 3)

            # Flatten across environments and agents for forward pass
            obs_batch = last_obs.reshape((num_envs * num_agents, *last_obs.shape[2:]))
            logits, value = network.apply(train_state.params, obs_batch)
            
            action = jax.random.categorical(_rng_action, logits)
            log_prob = jax.nn.log_softmax(logits)[jnp.arange(action.shape[0]), action]

            # Format actions for SocialJax: list of length `num_agents`
            actions_per_env = action.reshape((num_envs, num_agents))
            env_actions = [actions_per_env[:, i] for i in range(num_agents)]

            # Step parallel environments
            step_rngs = jax.random.split(_rng_step, num_envs)
            next_obs, next_env_state, reward, done, _, info = jax.vmap(env.step)(
                step_rngs, env_state, env_actions
            )

            # Reshape rewards and terminations back to (num_envs * num_agents,)
            reward_flat = jnp.array(reward).T.reshape(-1)
            done_flat = jnp.repeat(jnp.array(done), num_agents)

            transition = Transition(
                done=done_flat,
                action=action,
                value=value,
                reward=reward_flat,
                reputation=...,
                log_prob=log_prob,
                obs=obs_batch,
                info=info,
            )
            runner_state = (train_state, next_env_state, next_obs, rng)
            return runner_state, transition

        # --- 4. Outer Update Step (Rollout + GAE + Update) ---
        def _update_step(runner_state, unused):
            # COLLECT ROLLOUT
            runner_state, traj_batch = jax.lax.scan(_env_step, runner_state, None, config["NUM_STEPS"])

            # COMPUTE GAE (Reverse Scan)
            train_state, env_state, last_obs, rng = runner_state
            last_obs_batch = last_obs.reshape((num_envs * num_agents, *last_obs.shape[2:]))
            _, last_val = network.apply(train_state.params, last_obs_batch)

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

            # UPDATE EPOCHS (Mini-batch SGD)
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, traj_batch, gae, targets):
                        logits, value = network.apply(params, traj_batch.obs)
                        log_prob = jax.nn.log_softmax(logits)[jnp.arange(traj_batch.action.shape[0]), traj_batch.action]

                        # Clipped policy objective
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * gae
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2).mean()

                        # Value objective
                        value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                            -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        loss_critic = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                        # Entropy bonus
                        probs = jax.nn.softmax(logits)
                        entropy = -jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1).mean()

                        total_loss = loss_actor + config["VF_COEF"] * loss_critic - config["ENT_COEF"] * entropy
                        return total_loss, (loss_actor, loss_critic, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                
                # Flatten batch over steps, environments, and agents
                batch_size = config["NUM_STEPS"] * num_envs * num_agents
                permutation = jax.random.permutation(_rng, batch_size)
                
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(lambda x: x.reshape((batch_size,) + x.shape[2:]), batch)
                shuffled_batch = jax.tree_util.tree_map(lambda x: jnp.take(x, permutation, axis=0), batch)
                
                # Chunk into minibatches
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])),
                    shuffled_batch,
                )
                train_state, total_loss = jax.lax.scan(_update_minbatch, train_state, minibatches)
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(_update_epoch, update_state, None, config["UPDATE_EPOCHS"])
            train_state = update_state[0]
            metric = traj_batch.info
            runner_state = (train_state, env_state, last_obs, rng)
            return runner_state, metric

        # --- 5. Run Total Updates ---
        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, obs, _rng)
        runner_state, metrics = jax.lax.scan(_update_step, runner_state, None, config["TOTAL_TIMESTEPS"] // (config["NUM_STEPS"] * num_envs))
        return {"runner_state": runner_state, "metrics": metrics}

    return train