
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from jaxrl5.networks import default_init


class PixelMultiplexer(nn.Module):
    encoder_cls: type[nn.Module]
    network_cls: type[nn.Module]
    latent_dim: int
    stop_gradient: bool = False
    pixel_keys: tuple[str, ...] = ("pixels",)
    depth_keys: tuple[str, ...] = ()

    @nn.compact
    def __call__(
        self,
        observations: FrozenDict | dict,
        actions: jnp.ndarray | None = None,
        training: bool = False,
    ) -> jnp.ndarray:
        observations = FrozenDict(observations)
        depth_keys = [None] * len(self.pixel_keys) if len(self.depth_keys) == 0 else self.depth_keys

        xs = []
        for i, (pixel_key, depth_key) in enumerate(zip(self.pixel_keys, depth_keys)):
            x = observations[pixel_key].astype(jnp.float32) / 255.0
            if depth_key is not None:
                # The last dim is always for stacking, even if it's 1.
                x = jnp.concatenate([x, observations[depth_key]], axis=-2)

            x = jnp.reshape(x, (*x.shape[:-2], -1))

            x = self.encoder_cls(name=f"encoder_{i}")(x)

            if self.stop_gradient:
                # We do not update conv layers with policy gradients.
                x = jax.lax.stop_gradient(x)

            x = nn.Dense(self.latent_dim, kernel_init=default_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)
            xs.append(x)

        x = jnp.concatenate(xs, axis=-1)

        if "state" in observations:
            y = nn.Dense(self.latent_dim, kernel_init=default_init())(observations["state"])
            y = nn.LayerNorm()(y)
            y = nn.tanh(y)

            x = jnp.concatenate([x, y], axis=-1)

        if actions is None:
            return self.network_cls()(x, training)
        else:
            return self.network_cls()(x, actions, training)
