import jax
import jax.numpy as jnp
import numpy as np
import wandb
from pathlib import Path
from PIL import Image

from src.networks.networks2 import MAPPOAC, RecurrentMAPPOAC, LSTM
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

    rng = jax.random.PRNGKey(config["SEED"])
    obs, (state, rep), rng = env.initialize(rng)

    if algo == "MAPPO":
        network = MAPPOAC(action_dim=action_dim, small=config["SMALL"], activation=config["ACTIVATION"])
    elif algo == "RecurrentMAPPO":
        network = RecurrentMAPPOAC(
            action_dim=action_dim,
            hidden_dim=config["HIDDEN_DIM"],
            small=config["SMALL"],
            activation=config["ACTIVATION"],
            predict_pos=config.get("PREDICT_POS", False),  # must match training, or params won't fit
        )
    else:
        raise ValueError(f"Unknown ALGO in evaluate: {algo}")

    hidden = LSTM.initialize_carry(num_agents, config["HIDDEN_DIM"]) if algo == "RecurrentMAPPO" else None
    last_action = jnp.zeros((num_agents,), dtype=jnp.int32)
    last_done = jnp.ones((num_agents,), dtype=jnp.float32)

    frames = []
    for _ in range(num_frames):
        frames.append(env.env.render(jax.tree.map(lambda x: x[0], state)))
        obs_batch = obs.reshape((num_agents, *obs.shape[2:]))  # type: ignore
        world_batch = make_world_state(obs, 1, num_agents)
        rng, k_act, k_step = jax.random.split(rng, 3)

        if algo == "MAPPO":
            pi, _ = network.apply(params, (obs_batch, world_batch))
            action = pi.sample(seed=k_act)
        elif algo == "RecurrentMAPPO":
            actor_obs_t = jax.tree_util.tree_map(lambda x: x[None, ...], (obs_batch, last_done, last_action))
            world_batch_t = world_batch[None, ...]
            hidden, pi, _, _ = network.apply(params, hidden, actor_obs_t, world_batch_t)
            action = pi.sample(seed=k_act).squeeze(0)

        actions_env = action.reshape((1, num_agents))
        obs, (state, rep), _, done, _ = env.step(k_step, actions_env, (state, rep))
        last_action = action
        last_done = done.reshape((num_agents,)).astype(jnp.float32)

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pos_tag = "_pos" if config.get("PREDICT_POS", False) else ""
    path = f"{out_dir}/{config['ALGO']}_{config['ENV_NAME']}_{num_agents}agents_seed{config['SEED']}_{config['REPUTATION']}{pos_tag}.gif"
    imgs = [Image.fromarray(np.array(f)) for f in frames]
    imgs[0].save(path, format="GIF", save_all=True, append_images=imgs[1:], duration=200, loop=0)

    if config["WANDB"] and wandb.run is not None:
        wandb.log({"Episode GIF": wandb.Video(path, caption="Evaluation episode", format="gif")})
    return path