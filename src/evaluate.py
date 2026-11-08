import jax
import jax.numpy as jnp
import numpy as np
import wandb
from pathlib import Path
from PIL import Image

from src.networks.networks import ActorCritic
from src.networks.networks2 import (
    MAPPOAC,
    FeedforwardCOMAAC,
    RecurrentOnlyMAPPOAC,
    RecurrentMAPPOAC,
    LSTM,
)
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

    rng = jax.random.PRNGKey(config["SEED"])
    obs, (state, rep), rng = env.initialize(rng)

    # Set up network and recurrent state if needed
    is_recurrent = algo in ["RecurrentMAPPO", "RecurrentCOMA"]
    if algo == "IPPO":
        network = ActorCritic(action_dim=action_dim, activation=config["ACTIVATION"])
    elif algo == "MAPPO":
        network = MAPPOAC(action_dim=action_dim, small=config["SMALL"], activation=config["ACTIVATION"])
    elif algo == "COMA":
        network = FeedforwardCOMAAC(
            action_dim=action_dim, num_agents=num_agents, small=config["SMALL"], activation=config["ACTIVATION"]
        )
    elif algo == "RecurrentMAPPO":
        network = RecurrentOnlyMAPPOAC(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            activation=config["ACTIVATION"],
            small=config["SMALL"],
        )
    elif algo == "RecurrentCOMA":
        network = RecurrentMAPPOAC(
            action_dim=action_dim,
            num_agents=num_agents,
            hidden_dim=config["HIDDEN_DIM"],
            activation=config["ACTIVATION"],
            small=config["SMALL"],
        )
    else:
        raise ValueError(f"Unknown ALGO in evaluate: {algo}")

    hidden = LSTM.initialize_carry(num_agents, config["HIDDEN_DIM"]) if is_recurrent else None
    last_action = jnp.zeros((num_agents,), dtype=jnp.int32)
    last_done = jnp.zeros((num_agents,), dtype=jnp.float32)
    agent_id_oh = jnp.eye(num_agents)

    frames = []
    for _ in range(num_frames):
        frames.append(env.env.render(jax.tree.map(lambda x: x[0], state)))
        obs_batch = obs.reshape((num_agents, *obs.shape[2:]))  # type: ignore
        rng, k_act, k_step = jax.random.split(rng, 3)

        if algo == "IPPO":
            pi, _ = network.apply(params, obs_batch)
            action = pi.sample(seed=k_act)
        elif algo == "MAPPO":
            global_map, global_agent_vec = env.reconstruct_critic_state(state)
            pi, _ = network.apply(params, obs_batch, (global_map, global_agent_vec, agent_id_oh))
            action = pi.sample(seed=k_act)
        elif algo == "COMA":
            pi = network.apply(params, obs_batch, method=network.forward_actor)
            action = pi.sample(seed=k_act)
        elif algo == "RecurrentMAPPO":
            global_map, global_agent_vec = env.reconstruct_critic_state(state)
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], (obs_batch, last_done, last_action))
            critic_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], (global_map, global_agent_vec, agent_id_oh))
            out = network.apply(params, hidden, actor_obs_t, critic_obs_t)
            hidden = out.hidden
            action = out.pi.sample(seed=k_act).squeeze(0)
        elif algo == "RecurrentCOMA":
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], (obs_batch, last_done, last_action))
            hidden, pi, _ = network.apply(params, hidden, actor_obs_t, method=network.forward_actor)
            action = pi.sample(seed=k_act).squeeze(0)

        actions_env = action.reshape((1, num_agents))
        obs, (state, rep), _, done, _ = env.step(k_step, actions_env, (state, rep))
        last_action = action
        last_done = done.reshape((num_agents,)).astype(jnp.float32)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/{config['ALGO']}_{config['ENV_NAME']}_{num_agents}agents_seed{config['SEED']}_{config['REPUTATION']}.gif"
    imgs = [Image.fromarray(np.array(f)) for f in frames]
    imgs[0].save(path, format="GIF", save_all=True, append_images=imgs[1:], duration=200, loop=0)

    if config["WANDB"] and wandb.run is not None:
        wandb.log({"Episode GIF": wandb.Video(path, caption="Evaluation episode", format="gif")})
    return path