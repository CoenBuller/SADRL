"""Sanity check that JAX sees and can use the GPU on an ALICE compute node."""
import jax
import jax.numpy as jnp

print("JAX", jax.__version__, "| backend:", jax.default_backend(), "| devices:", jax.devices())
x = jnp.ones((4096, 4096))
print("matmul ok:", float((x @ x).sum()))
assert jax.default_backend() == "gpu", "JAX is not using the GPU"
