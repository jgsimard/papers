import functools

import flax.linen as nn
import jax.numpy as jnp
import tensorflow_probability

from jaxrl5.distributions.tanh_transformed import TanhTransformedDistribution
from jaxrl5.networks import default_init

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class Normal(nn.Module):
    base_cls: type[nn.Module]
    action_dim: int
    log_std_min: float | None = -20
    log_std_max: float | None = 2
    state_dependent_std: bool = True
    squash_tanh: bool = False

    @nn.compact
    def __call__(self, inputs, *args, **kwargs) -> tfd.Distribution:
        x = self.base_cls()(inputs, *args, **kwargs)

        means = nn.Dense(
            self.action_dim,
            kernel_init=default_init(),
            name="OutputDenseMean",
        )(x)
        if self.state_dependent_std:
            log_stds = nn.Dense(
                self.action_dim,
                kernel_init=default_init(),
                name="OutputDenseLogStd",
            )(x)
        else:
            log_stds = self.param(
                "OutpuLogStd",
                nn.initializers.zeros,
                (self.action_dim,),
                jnp.float32,
            )

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = tfd.MultivariateNormalDiag(
            loc=means,
            scale_diag=jnp.exp(log_stds),
        )

        if self.squash_tanh:
            return TanhTransformedDistribution(distribution)
        return distribution


TanhNormal = functools.partial(Normal, squash_tanh=True)
