""" 
Based on PureJaxRL & jaxmarl Implementation of PPO
Fixed bugs of jaxmarl
"""

import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from flax import traverse_util
from flax.training.train_state import TrainState
from gymnax.wrappers.purerl import LogWrapper
import socialjax
from socialjax.wrappers.baselines import MAPPOWorldStateWrapper, LogWrapper, SVOLogWrapper  # [SADRL fix 2026-10-18] SVOLogWrapper also logs the raw (pre-shaping) return
import hydra
from omegaconf import OmegaConf
import wandb
import copy
# Import shared MAPPO small network architectures and utilities
from algorithms.utils import (
    SmallActor,
    SmallCritic,
    Actor as FullActor,  # [SADRL fix 2026-10-20] NETWORK_SIZE=full: IPPO's CNN torso for the actor and the critic
    Critic as FullCritic,
    mappo_networks,  # [SADRL fix 2026-11-03] CNN_SIZE x FC_SIZE grid (round 10)
    batchify_dict,
    batchify_numpy,
    unbatchify,
    save_params,
    load_params,
    evaluate_mappo_style as evaluate,
    MAPPOTransition as Transition,
)

# Import shared MAPPO small network architectures and utilities

def actor_health(actor_network, actor_params, obs):
    """[SADRL fix 2026-10-21] Dead-actor check (round 9, D.1), logging only.

    dead_frac_<layer>: share of the layer's units whose pre-activation is <= 0 for every sample in the batch (and every
    position, for a conv channel), for each ReLU layer. The actor's last Dense gives the logits and has no ReLU.
    logit_std: std of each logit over the batch, averaged over actions; about 0 means every observation gets the same policy.
    """
    obs = jnp.reshape(obs, (-1, *obs.shape[-3:]))
    pi, state = actor_network.apply(
        jax.lax.stop_gradient(actor_params), obs,
        capture_intermediates=lambda mdl, _: isinstance(mdl, (nn.Conv, nn.Dense)), mutable=["intermediates"],
    )
    layers = traverse_util.flatten_dict(state["intermediates"])
    head = max((p for p in layers if len(p) == 2 and p[0].startswith("Dense_")), key=lambda p: int(p[0].split("_")[1]))
    health = {}
    for path, (out,) in layers.items():
        if path != head:
            health["dead_frac_" + "_".join(path[:-1]).lower()] = jnp.mean(jnp.all(out <= 0, axis=tuple(range(out.ndim - 1))))
    health["logit_std"] = jnp.std(pi.logits, axis=0).mean()
    return health


def make_train(config):
    env = socialjax.make(config["ENV_NAME"], **config["ENV_KWARGS"])
    config["NUM_ACTORS"] = env.num_agents * config["NUM_ENVS"]
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ACTORS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    config["CLIP_EPS"] = config["CLIP_EPS"] / env.num_agents if config["SCALE_CLIP_EPS"] else config["CLIP_EPS"]

    env = MAPPOWorldStateWrapper(env)
    env = SVOLogWrapper(env, replace_info=False)  # [SADRL fix 2026-10-18] was LogWrapper; adds returned_episode_original_returns (info["original_rewards"])

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):

        # INIT NETWORK
        # [SADRL fix 2026-10-20] NETWORK_SIZE: "small" (default, SocialJax's MAPPO) or "full" (IPPO's torso)
        # Actor, Critic = (FullActor, FullCritic) if config.get("NETWORK_SIZE", "small") == "full" else (SmallActor, SmallCritic)
        Actor, Critic = mappo_networks(config)  # [SADRL fix 2026-11-03] CNN_SIZE / FC_SIZE, each defaulting to NETWORK_SIZE
        actor_network = Actor(env.action_space().n, activation=config["ACTIVATION"])
        critic_network = Critic(activation=config["ACTIVATION"])

        rng, _rng_actor, _rng_critic = jax.random.split(rng, 3)
        ac_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape))
            
        actor_network_params = actor_network.init(_rng_actor, ac_init_x)

        obs_shape = env.observation_space()[0].shape
        global_shape = (1, *obs_shape[:-1], obs_shape[-1] * env.num_agents)
        cr_init_x = jnp.zeros(global_shape)
        # cr_init_x = jnp.zeros((1, *(env.observation_space()[0]).shape)) 

        critic_network_params = critic_network.init(_rng_critic, cr_init_x)

        if config["ANNEAL_LR"]:            
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            actor_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
            critic_tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        actor_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=actor_network_params,
            tx=actor_tx,
        )
        critic_train_state = TrainState.create(
            apply_fn=actor_network.apply,
            params=critic_network_params,
            tx=critic_tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = jax.vmap(env.reset, in_axes=(0,))(reset_rng)

        # TRAIN LOOP
        def _update_step(update_runner_state, unused):
            # COLLECT TRAJECTORIES
            runner_state, update_steps = update_runner_state
            def _env_step(runner_state, step_idx):
                train_states, env_state, last_obs, last_done, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)

                obs_batch = jnp.transpose(last_obs,(1,0,2,3,4)).reshape(-1, *(env.observation_space()[0]).shape)
                # obs_batch = jnp.stack([last_obs[a] for a in env.agents]).reshape(-1, *env.observation_space().shape)

                pi = actor_network.apply(train_states[0].params, obs_batch)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)
                env_act = unbatchify(
                    action, env.agents, config["NUM_ENVS"], env.num_agents
                )

                # env_act = {k: v.flatten() for k, v in env_act.items()}
                env_act = [v for v in env_act.values()]

                #VALUE
                world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
                world_state = jnp.expand_dims(world_state, axis=0)
                world_state = jnp.tile(world_state, (env.num_agents, 1, 1, 1, 1))
                world_state = jnp.reshape(world_state, (-1, *(world_state.shape[2:])))
                # world_state = last_obs["world_state"].swapaxes(0,1)  
                # world_state = world_state.reshape((config["NUM_ACTORS"],-1))
                value = critic_network.apply(train_states[1].params, world_state)
                # expanded_value = jnp.expand_dims(value, axis=0)
                # value = jnp.tile(expanded_value, (env.num_agents, 1))
                value = value.reshape(config["NUM_ACTORS"])

                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])

                obsv, env_state, reward, done, info = jax.vmap(
                    env.step, in_axes=(0, 0, 0)
                )(rng_step, env_state, env_act)

                info = jax.tree.map(lambda x: x.reshape((config["NUM_ACTORS"])), info)
                done_batch = batchify_dict(done, env.agents, config["NUM_ACTORS"]).squeeze()
                # [SADRL fix 2026-10-20] The world state is the same for all 7 agents of an env, so it is stored once
                # per env step (ws_env) and the transition keeps only its row index (actors are agent-major: actor k
                # is env k % NUM_ENVS). This cuts the rollout buffer's world-state memory by num_agents.
                ws_index = step_idx * config["NUM_ENVS"] + jnp.arange(config["NUM_ACTORS"]) % config["NUM_ENVS"]
                ws_env = world_state[: config["NUM_ENVS"]]
                transition = Transition(
                    jnp.tile(done["__all__"], env.num_agents),
                    last_done,
                    action,
                    value,
                    batchify_numpy(reward, env.agents, config["NUM_ACTORS"]).squeeze(),
                    log_prob,
                    obs_batch,
                    ws_index,
                    info
                )
                runner_state = (train_states, env_state, obsv, done_batch, rng)
                return runner_state, (transition, ws_env)

            runner_state, (traj_batch, ws_buffer) = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"])
            )
            ws_buffer = ws_buffer.reshape(-1, *ws_buffer.shape[2:])  # [SADRL fix 2026-10-20] (NUM_STEPS * NUM_ENVS, H, W, C)

            # CALCULATE ADVANTAGE
            train_states, env_state, last_obs, last_done, rng = runner_state

            # last_world_state = last_obs["world_state"].swapaxes(0,1)
            # last_world_state = last_world_state.reshape((config["NUM_ACTORS"],-1))
            last_world_state = jnp.transpose(last_obs, (0,2,3,1,4)).reshape(config["NUM_ENVS"], *(env.observation_space()[0]).shape[:-1], -1)
            last_val = critic_network.apply(train_states[1].params, last_world_state)
            last_val = jnp.expand_dims(last_val, axis=0)
            last_val = jnp.tile(last_val, (env.num_agents, 1))
            last_val = last_val.reshape((config["NUM_ACTORS"],-1))
            last_val = last_val.squeeze()

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
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

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    actor_train_state, critic_train_state = train_states
                    traj_batch, advantages, targets = batch_info

                    def _actor_loss_fn(actor_params, traj_batch, gae):
                        # RERUN NETWORK
                        pi = actor_network.apply(
                            actor_params,
                            jnp.reshape(traj_batch.obs, (-1, *(traj_batch.obs).shape[-3:])),
                        ) #.reshape(traj_batch.action.shape)

                        log_prob = pi.log_prob(traj_batch.action.reshape(-1))

                        # CALCULATE ACTOR LOSS
                        logratio = log_prob - jnp.reshape(traj_batch.log_prob, (-1,))
                        logratio = jnp.reshape(logratio, traj_batch.action.shape)
                        log_prob = jnp.reshape(log_prob, traj_batch.action.shape)
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()
                        # debug
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clip_frac = jnp.mean(jnp.abs(ratio - 1) > config["CLIP_EPS"])
                        
                        actor_loss = (
                            loss_actor
                            - config["ENT_COEF"] * entropy
                        )
                        return actor_loss, (loss_actor, entropy, ratio, approx_kl, clip_frac)

                    def _critic_loss_fn(critic_params, traj_batch, targets):
                        # RERUN NETWORK
                        world_state = jnp.take(ws_buffer, traj_batch.world_state.reshape(-1), axis=0)  # [SADRL fix 2026-10-20] look up the stored world state
                        value = critic_network.apply(critic_params, world_state) 
                        value = jnp.reshape(value, traj_batch.value.shape)
                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                            value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                        )
                        critic_loss = config["VF_COEF"] * value_loss
                        return critic_loss, (value_loss)

                    actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                    actor_loss, actor_grads = actor_grad_fn(
                        actor_train_state.params, traj_batch, advantages
                    )
                    critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                    critic_loss, critic_grads = critic_grad_fn(
                        critic_train_state.params, traj_batch, targets
                    )
                    # [SADRL fix 2026-10-21] LOG_ACTOR_HEALTH: dead-actor check (round 9), on the actor before this step
                    health = {}
                    if config.get("LOG_ACTOR_HEALTH", False):
                        health = actor_health(actor_network, actor_train_state.params, traj_batch.obs)
                        health["actor_grad_norm"] = optax.global_norm(actor_grads)  # before the MAX_GRAD_NORM clip
                    # [SADRL fix 2026-11-03] Gradient norms of actor and critic, always (before the MAX_GRAD_NORM clip).
                    health["actor_grad_norm"] = optax.global_norm(actor_grads)
                    health["critic_grad_norm"] = optax.global_norm(critic_grads)

                    actor_train_state = actor_train_state.apply_gradients(grads=actor_grads)
                    critic_train_state = critic_train_state.apply_gradients(grads=critic_grads)
                    
                    total_loss = actor_loss[0] + critic_loss[0]
                    loss_info = {
                        "total_loss": total_loss,
                        "actor_loss": actor_loss[0],
                        "value_loss": critic_loss[0],
                        "entropy": actor_loss[1][1],
                        "ratio": actor_loss[1][2],
                        "approx_kl": actor_loss[1][3],
                        "clip_frac": actor_loss[1][4],
                        **health,
                    }
                    
                    return (actor_train_state, critic_train_state), loss_info

                (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                ) = update_state
                rng, _rng = jax.random.split(rng)

                # batch = (
                #     traj_batch,
                #     advantages.squeeze(),
                #     targets.squeeze(),
                # )
                # permutation = jax.random.permutation(_rng, config["NUM_ACTORS"])

                # shuffled_batch = jax.tree_util.tree_map(
                #     lambda x: jnp.take(x, permutation, axis=1), batch
                # )
                
                # minibatches = jax.tree_util.tree_map(
                #     lambda x: jnp.swapaxes(
                #         jnp.reshape(
                #             x,
                #             [x.shape[0], config["NUM_MINIBATCHES"], -1]
                #             + list(x.shape[2:]),
                #         ),
                #         1,
                #         0,
                #     ),
                #     shuffled_batch,
                # )
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ACTORS"]
                ), "batch size must be equal to number of steps * number of actors"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )

                train_states, loss_info = jax.lax.scan(
                    _update_minbatch, train_states, minibatches
                )
                update_state = (
                    train_states,
                    traj_batch,
                    advantages,
                    targets,
                    rng,
                )
                return update_state, loss_info

            update_state = (
                train_states,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_states = update_state[0]
            metric = traj_batch.info
            # loss_info["ratio_0"] = loss_info["ratio"].at[0,0].get()
            # [SADRL fix 2026-10-21] Log the PPO diagnostics (means over epochs and minibatches) and the critic's
            # scale: mean value prediction at rollout time and mean GAE target. Logging only; training is unchanged.
            loss_info = jax.tree.map(lambda x: x.mean(), loss_info)
            metric = dict(metric)
            metric.update({k: loss_info[k] for k in ("actor_loss", "value_loss", "entropy", "approx_kl", "clip_frac")})
            metric.update({k: v for k, v in loss_info.items() if k.startswith("dead_frac_") or k in ("logit_std", "actor_grad_norm", "critic_grad_norm")})  # [SADRL fix 2026-10-21] (+ critic_grad_norm, 2026-11-03)
            # [SADRL fix 2026-11-03] Logging only (round 10, G.1):
            # critic_explained_var = 1 - Var(target - V) / Var(target), with V the critic's prediction at rollout time.
            metric["critic_explained_var"] = 1.0 - jnp.var(targets - traj_batch.value) / (jnp.var(targets) + 1e-8)
            # Who cleans and who eats: per env, each agent's raw reward and clean actions over the rollout, sorted
            # high to low within the env, averaged over envs, scaled to one episode. rank0 = the top agent.
            # info is env-major: actor index = env * num_agents + agent.
            per_episode = config["ENV_KWARGS"]["num_inner_steps"] / config["NUM_STEPS"]
            for key, name in (("agent_rewards", "reward"), ("clean_action_info", "clean")):
                per_agent = traj_batch.info[key].reshape(config["NUM_STEPS"], config["NUM_ENVS"], env.num_agents).sum(0)
                ranked = -jnp.sort(-per_agent, axis=1).mean(0) * per_episode
                by_id = per_agent.mean(0) * per_episode  # [SADRL fix 2026-11-05] per agent id, unsorted (round 13, L.2)
                for k in range(env.num_agents):
                    metric[f"{name}_rank{k}"] = ranked[k]
                    metric[f"{name}_agent{k}"] = by_id[k]
            metric["value_mean"] = traj_batch.value.mean()
            metric["target_mean"] = targets.mean()
            rng = update_state[-1]

            def callback(metric):
                wandb.log(metric)

            update_steps = update_steps + 1
            metric = jax.tree.map(lambda x: x.mean(), metric)
            metric["update_steps"] = update_steps
            metric["env_step"] = update_steps * config["NUM_STEPS"] * config["NUM_ENVS"]
            metric["clean_action_info"] = metric["clean_action_info"] * config["ENV_KWARGS"]["num_inner_steps"]

            # jax.experimental.io_callback(callback, None, metric)

            jax.debug.callback(callback, metric)
            
            runner_state = (train_states, env_state, last_obs, last_done, rng)
            return (runner_state, update_steps), metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            (actor_train_state, critic_train_state),
            env_state,
            obsv,
            jnp.zeros((config["NUM_ACTORS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, (runner_state, 0), None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state}

    return train

# Used by algorithms/train.py to dispatch through algorithms.MAPPO._runner.
SINGLE_RUN_KWARGS = {"wandb_name": "mappo_cnn_cleanup"}
TUNE_KWARGS       = {"sweep_name": "cleanup"}
