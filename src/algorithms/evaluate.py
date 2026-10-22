import jax, jax.numpy as jnp, numpy as np, wandb
from pathlib import Path
from PIL import Image
from src.algorithms.networks import ActorCritic
from src.environments.wrappers import Environment


def make_gif(params, config, num_frames=1000, out_dir="evaluation"):
    env = Environment(
        common_reward=config["COMMON_REWARD"], env_name=config["ENV_NAME"],
        n_agents=config["N_AGENTS"], n_envs=1, seed=config["SEED"],
        reputation=config["REPUTATION"], own_pos=tuple(config["OWN_POS"]),
        decay_coef=config["REP_DECAY"],
    )
    network = ActorCritic(action_dim=env.env.action_space(0).n, activation=config["ACTIVATION"])

    rng = jax.random.PRNGKey(0)
    obs, (state, rep), rng = env.initialize(rng)
    frames = []
    for _ in range(num_frames):
        frames.append(env.env.render(jax.tree.map(lambda x: x[0], state)))   # single-env state
        obs_batch = obs.reshape((env.n_agents, *obs.shape[2:])) # type: ignore
        pi, _ = network.apply(params, obs_batch)
        rng, k_act, k_step = jax.random.split(rng, 3)
        actions = pi.sample(seed=k_act).reshape((1, env.n_agents))
        obs, (state, rep), _, _, _ = env.step(k_step, actions, (state, rep))

    Path(out_dir).mkdir(parents=True, exist_ok=True)
    path = f"{out_dir}/{config['ALGO']}_{config['ENV_NAME']}_{env.n_agents}agents_seed{config['SEED']}_{[config['REPUTATION']]}.gif"
    imgs = [Image.fromarray(np.array(f)) for f in frames]
    imgs[0].save(path, format="GIF", save_all=True, append_images=imgs[1:], duration=200, loop=0)

    if config["WANDB"] and wandb.run is not None:
        wandb.log({"Episode GIF": wandb.Video(path, caption="Evaluation episode", format="gif")})
    return path

if __name__ == "__main__":
    from src.config import Config
    import dataclasses
    config = dataclasses.asdict(Config())

    env = Environment(
    common_reward=config["COMMON_REWARD"], env_name=config["ENV_NAME"],
    n_agents=config["N_AGENTS"], n_envs=1, seed=config["SEED"],
    reputation=config["REPUTATION"], own_pos=tuple(config["OWN_POS"]),
    decay_coef=config["REP_DECAY"],
    )
    obs, _, _ = env.initialize(jax.random.PRNGKey(0))          # (1, A, H, W, C+1)

    network = ActorCritic(action_dim=env.env.action_space(0).n, activation=config["ACTIVATION"])
    params = network.init(jax.random.PRNGKey(1), jnp.zeros((env.n_agents, *obs.shape[2:])))

    make_gif(params, config, num_frames=50)
    ac = ActorCritic