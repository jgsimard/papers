from collections.abc import Callable, Sequence

import flax.linen as nn
import jax.numpy as jnp
from jax import Array


def default_init(scale: float | None = jnp.sqrt(2)):
    return nn.initializers.orthogonal(scale)


class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Callable[[Array], Array] = nn.relu
    activate_final: bool = False
    use_layer_norm: bool = False
    scale_final: float | None = None
    dropout_rate: float | None = None
    use_pnorm: bool = False

    @nn.compact
    def __call__(self, x: Array, training: bool = False) -> Array:
        for i, size in enumerate(self.hidden_dims):
            if i + 1 == len(self.hidden_dims) and self.scale_final is not None:
                x = nn.Dense(size, kernel_init=default_init(self.scale_final))(x)
            else:
                x = nn.Dense(size, kernel_init=default_init())(x)

            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.dropout_rate is not None and self.dropout_rate > 0:
                    x = nn.Dropout(rate=self.dropout_rate)(
                        x,
                        deterministic=not training,
                    )
                if self.use_layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
        if self.use_pnorm:
            x /= jnp.linalg.norm(x, axis=-1, keepdims=True).clip(1e-10)
        return x
