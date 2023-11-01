from typing import Any

import tensorflow_probability
from jax import Array

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions
tfb = tfp.bijectors


# Inspired by
# https://github.com/deepmind/acme/blob/300c780ffeb88661a41540b99d3e25714e2efd20/acme/jax/networks/distributional.py#L163
# but modified to only compute a mode.
class TanhTransformedDistribution(tfd.TransformedDistribution):
    def __init__(self, distribution: tfd.Distribution, validate_args: bool = False) -> None:
        super().__init__(
            distribution=distribution,
            bijector=tfb.Tanh(),
            validate_args=validate_args,
        )

    def mode(self) -> Array:
        return self.bijector.forward(self.distribution.mode())

    @classmethod
    def _parameter_properties(cls, dtype: Any | None, num_classes=None):
        td_properties = super()._parameter_properties(dtype, num_classes=num_classes)
        del td_properties["bijector"]
        return td_properties