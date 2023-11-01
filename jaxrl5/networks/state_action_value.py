import flax.linen as nn
import jax.numpy as jnp
from jax import Array

from jaxrl5.networks import default_init


class StateActionValue(nn.Module):
    base_cls: nn.Module

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

        value = nn.Dense(1, kernel_init=default_init())(outputs)

        return jnp.squeeze(value, -1)
