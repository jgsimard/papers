from collections.abc import Callable
from functools import partial

import distrax
import jax
import jax.numpy as jnp
import numpy as np
from flax.training.train_state import TrainState

from papers.rl.data.dataset import DatasetDict
from papers.types import Params, PRNGKey


class Agent:
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(self._actor.apply_fn, self._actor.params, batch)

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
            self._actor.apply_fn,
            self._actor.params,
            observations,
        )

        return np.asarray(actions)

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng,
            self._actor.apply_fn,
            self._actor.params,
            observations,
        )

        self._rng = rng

        return np.asarray(actions)


# TODO (jgsimard): check if can put the functions directly into Agent
@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_log_prob_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    batch: DatasetDict,
) -> float:
    dist = actor_apply_fn({"params": actor_params}, batch["observations"])
    log_probs = dist.log_prob(batch["actions"])
    return log_probs.mean()


@partial(jax.jit, static_argnames="actor_apply_fn")
def eval_actions_jit(
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> jnp.ndarray:
    dist = actor_apply_fn({"params": actor_params}, observations)
    return dist.mode()


@partial(jax.jit, static_argnames="actor_apply_fn")
def sample_actions_jit(
    rng: PRNGKey,
    actor_apply_fn: Callable[..., distrax.Distribution],
    actor_params: Params,
    observations: np.ndarray,
) -> tuple[PRNGKey, jnp.ndarray]:
    dist = actor_apply_fn({"params": actor_params}, observations)
    rng, key = jax.random.split(rng)
    return rng, dist.sample(seed=key)
