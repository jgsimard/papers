import flax.linen as nn
import jax.numpy as jnp
from jax import Array

my_init = nn.initializers.xavier_uniform


class StateActionNextState(nn.Module):
    base_cls: nn.Module
    obs_dim: int

    @nn.compact
    def __call__(
        self,
        observations: Array,
        actions: Array,
        *args,
        **kwargs,
    ) -> Array:
        inputs = jnp.concatenate([observations, actions], axis=-1)
        outputs = self.base_cls()(inputs, *args, **kwargs)

        residual = nn.Dense(self.obs_dim, kernel_init=my_init())(outputs)
        return observations + residual
