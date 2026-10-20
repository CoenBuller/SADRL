import jax
import jax.numpy as jnp
from src.environments.wrappers import Environment  # or from src.environments.wrappers import Environment

env = Environment("clean_up", n_agents=5, n_envs=1)
rng = jax.random.PRNGKey(seed=30)

obs, (st, rep), rng = env.initialize(rng)
raw = obs[..., :-1]

# Assign distinct reputation values [1.0, 2.0, 3.0, 4.0, 5.0] to agents 0..4
rep = jnp.arange(1, 6, dtype=jnp.float32)[None, :]
out = env.add_rep(raw, st, rep)[..., -1]  # Shape: (E, A, H, W)

print("=== 1. Self-Position Check ===")
for a in range(env.n_agents):
    val = float(out[0, a, 9, 5])
    assert val == a + 1, f"Agent {a} missing at own_pos (9,5)! Got {val}"
print("All agents correctly placed at own_pos (9, 5) with their own reputation.\n")

print("=== 2. Cardinal Rotation Unit Test (All 4 Headings) ===")
# Place Agent 0 at (10, 10) and test each heading d in {0, 1, 2, 3}:
#   Agent 1 is placed 2 cells AHEAD of Agent 0 -> should always appear at (9 - 2, 5) = (7, 5)
#   Agent 2 is placed 1 cell RIGHT of Agent 0  -> should always appear at (9, 5 + 1) = (9, 6)
forward_deltas = {0: (-2, 0), 1: (0, 2), 2: (2, 0), 3: (0, -2)}
right_deltas   = {0: (0, 1),  1: (1, 0), 2: (0, -1), 3: (-1, 0)}
dir_names      = {0: "North (0)", 1: "East (1)", 2: "South (2)", 3: "West (3)"}

for d in range(4):
    f_dr, f_dc = forward_deltas[d]
    r_dr, r_dc = right_deltas[d]
    
    fake_locs = jnp.array([[
        [10, 10, d],                # Agent 0 (observer) at (10, 10) facing d
        [10 + f_dr, 10 + f_dc, 0],  # Agent 1: 2 steps ahead of Agent 0 (rep = 2.0)
        [10 + r_dr, 10 + r_dc, 0],  # Agent 2: 1 step right of Agent 0  (rep = 3.0)
        [0, 0, 0],                  # Agent 3: far away
        [0, 0, 0],                  # Agent 4: far away
    ]], dtype=jnp.int32)
    
    fake_state = st.replace(agent_locs=fake_locs)
    chan = env.add_rep(raw, fake_state, rep)[0, 0, :, :, -1]  # Observer 0's reputation grid
    
    ahead_val = float(chan[7, 5])
    right_val = float(chan[9, 6])
    print(f"Facing {dir_names[d]:9s} | Expected (7,5)=2.0, Got {ahead_val} | Expected (9,6)=3.0, Got {right_val}")
    assert ahead_val == 2.0, f"Failed AHEAD check for direction {d}! FoundAgent1 at {jnp.argwhere(chan == 2.0)}"
    assert right_val == 3.0, f"Failed RIGHT check for direction {d}! FoundAgent2 at {jnp.argwhere(chan == 3.0)}"

print("\nAll 4 cardinal rotations passed!\n")

print("=== 3. Live State Inspection ===")
print("Global agent_locs (row, col, dir):\n", st.agent_locs[0])
for i in range(env.n_agents):
    coords = jnp.argwhere(out[0, i] > 0)
    vals = out[0, i][coords[:, 0], coords[:, 1]]
    print(f"Agent {i} (facing d={int(st.agent_locs[0, i, 2])}) sees reputations {vals.tolist()} at view coords {coords.tolist()}")