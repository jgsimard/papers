import flax.linen as nn
from jax import Array

from jaxrl5.networks import default_init


class TanhDeterministic(nn.Module):
    base_cls: type[nn.Module]
    action_dim: int

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> Array:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(),
            name="OutputDenseMean",
        )(x)
        return nn.tanh(means)
