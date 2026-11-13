import jax
import jax.numpy as jnp
import numpy as np
import wandb
from pathlib import Path
from PIL import Image

from src.networks.networks2 import MAPPOAC, RecurrentMAPPOAC, RecurrentIPPOAC, LSTM, GRU
from src.algorithms.MAPPO import make_world_state
from src.environments.env_wrapper import Environment


def make_gif(params, config, num_frames=1000, out_dir="evaluation"):
    env = Environment(
        common_reward=config["COMMON_REWARD"],
        env_name=config["ENV_NAME"],
        n_agents=config["N_AGENTS"],
        n_envs=1,
        seed=config["SEED"],
        reputation=config["REPUTATION"],
        own_pos=tuple(config["OWN_POS"]),
        decay_coef=config["REP_DECAY"],
    )
    action_dim = env.env.action_space(0).n
    num_agents = env.n_agents
    algo = config["ALGO"]
    use_lstm = config.get("LSTM", False)
    predict_pos = config.get("PREDICT_POS", False)

    rng = jax.random.PRNGKey(config["SEED"])
    obs, (state, rep), rng = env.initialize(rng)

    if algo == "MAPPO":
        network = MAPPOAC(action_dim=action_dim, small=config["SMALL"], activation=config["ACTIVATION"])
        hidden = None

        @jax.jit
        def step_eval(params, hidden, obs, state, rep, last_done, last_action, rng):
            rng, k_act, k_step = jax.random.split(rng, 3)
            obs_batch = obs.reshape((num_agents, *obs.shape[2:]))
            world_batch = make_world_state(obs, 1, num_agents)
            pi, _ = network.apply(params, (obs_batch, world_batch))
            action = pi.sample(seed=k_act)
            actions_env = action.reshape((1, num_agents))
            next_obs, (next_state, next_rep), _, done, _ = env.step(k_step, actions_env, (state, rep))
            next_done = done.reshape((num_agents,)).astype(jnp.float32)
            return hidden, next_obs, next_state, next_rep, next_done, action, rng

    elif algo == "RecurrentMAPPO":
        network = RecurrentMAPPOAC(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            small=config["SMALL"],
            activation=config["ACTIVATION"],
            predict_pos=predict_pos,
            lstm=use_lstm,
        )
        hidden = (
            LSTM.initialize_carry(num_agents, config["HIDDEN_DIM"])
            if use_lstm
            else GRU.initialize_carry(num_agents, config["HIDDEN_DIM"])
        )

        @jax.jit
        def step_eval(params, hidden, obs, state, rep, last_done, last_action, rng):
            rng, k_act, k_step = jax.random.split(rng, 3)
            obs_batch = obs.reshape((num_agents, *obs.shape[2:]))
            world_batch = make_world_state(obs, 1, num_agents)
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], (obs_batch, last_done, last_action))
            world_batch_t = world_batch[None, ...]
            new_hidden, pi, _, _ = network.apply(params, hidden, actor_obs_t, world_batch_t)
            action = pi.sample(seed=k_act).squeeze(0)
            actions_env = action.reshape((1, num_agents))
            next_obs, (next_state, next_rep), _, done, _ = env.step(k_step, actions_env, (state, rep))
            next_done = done.reshape((num_agents,)).astype(jnp.float32)
            return new_hidden, next_obs, next_state, next_rep, next_done, action, rng

    elif algo == "RecurrentIPPO":
        network = RecurrentIPPOAC(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            activation=config["ACTIVATION"],
            predict_pos=predict_pos,
            lstm=use_lstm,
        )
        if use_lstm:
            init_hidden_a = LSTM.initialize_carry(num_agents, config["HIDDEN_DIM"])
            init_hidden_c = LSTM.initialize_carry(num_agents, config["HIDDEN_DIM"])
        else:
            init_hidden_a = GRU.initialize_carry(num_agents, config["HIDDEN_DIM"])
            init_hidden_c = GRU.initialize_carry(num_agents, config["HIDDEN_DIM"])
        hidden = (init_hidden_a, init_hidden_c)

        @jax.jit
        def step_eval(params, hidden, obs, state, rep, last_done, last_action, rng):
            rng, k_act, k_step = jax.random.split(rng, 3)
            obs_batch = obs.reshape((num_agents, *obs.shape[2:]))
            obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], (obs_batch, last_done, last_action))
            new_hidden, pi, _, _ = network.apply(params, hidden, obs_t)
            action = pi.sample(seed=k_act).squeeze(0)
            actions_env = action.reshape((1, num_agents))
            next_obs, (next_state, next_rep), _, done, _ = env.step(k_step, actions_env, (state, rep))
            next_done = done.reshape((num_agents,)).astype(jnp.float32)
            return new_hidden, next_obs, next_state, next_rep, next_done, action, rng
    else:
        raise ValueError(f"Unknown ALGO in evaluate: {algo}")

    last_action = jnp.zeros((num_agents,), dtype=jnp.int32)
    last_done = jnp.ones((num_agents,), dtype=jnp.float32)

    frames = []
    for _ in range(num_frames):
        frames.append(env.env.render(jax.tree_util.tree_map(lambda x: x[0], state)))
        hidden, obs, state, rep, last_done, last_action, rng = step_eval(
            params, hidden, obs, state, rep, last_done, last_action, rng
        )

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pos_tag = "_pos" if predict_pos else ""
    lstm_tag = "_lstm" if use_lstm else ""
    path = f"{out_dir}/{config['ALGO']}_{config['ENV_NAME']}_{num_agents}agents_seed{config['SEED']}_{config['REPUTATION']}{pos_tag}{lstm_tag}.gif"

    imgs = [Image.fromarray(np.array(f)) for f in frames]
    imgs[0].save(path, format="GIF", save_all=True, append_images=imgs[1:], duration=200, loop=0)

    if config["WANDB"] and wandb.run is not None:
        wandb.log({"Episode GIF": wandb.Video(path, caption="Evaluation episode", format="gif")})
    return path
