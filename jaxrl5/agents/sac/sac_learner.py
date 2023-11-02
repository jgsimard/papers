"""Implementations of algorithms for continuous control."""

from collections.abc import Sequence
from functools import partial

import gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import struct
from flax.training.train_state import TrainState
from jax import Array

from jaxrl5.agents.agent import Agent
from jaxrl5.agents.sac.temperature import Temperature
from jaxrl5.data.dataset import DatasetDict
from jaxrl5.distributions import TanhNormal
from jaxrl5.networks import (
    MLP,
    Ensemble,
    MLPResNetV2,
    StateActionValue,
    subsample_ensemble,
)
from jaxrl5.utils import convert_to_numpy_array, decay_mask_fn, l2_distance, l2_norm


def make_mlp_actor(hidden_dims, use_pnorm, action_dim, observations, key, lr):
    actor_base_cls = partial(
        MLP,
        hidden_dims=hidden_dims,
        activate_final=True,
        use_pnorm=use_pnorm,
    )
    actor_def = TanhNormal(actor_base_cls, action_dim)
    actor_params = actor_def.init(key, observations)["params"]
    return TrainState.create(
        apply_fn=actor_def.apply,
        params=actor_params,
        tx=optax.adam(learning_rate=lr),
    )


def make_temp(init_temperature, key, lr):
    temp_def = Temperature(init_temperature)
    temp_params = temp_def.init(key)["params"]
    return TrainState.create(
        apply_fn=temp_def.apply,
        params=temp_params,
        tx=optax.adam(learning_rate=lr),
    )


def make_critic(
    use_critic_resnet,
    hidden_dims,
    critic_dropout_rate,
    critic_layer_norm,
    use_pnorm,
    num_qs,
    key,
    batch,
    critic_weight_decay,
    max_gradient_norm,
    critic_lr,
    num_min_qs,
):
    if use_critic_resnet:
        critic_base_cls = partial(
            MLPResNetV2,
            num_blocks=1,
        )
    else:
        critic_base_cls = partial(
            MLP,
            hidden_dims=hidden_dims,
            activate_final=True,
            dropout_rate=critic_dropout_rate,
            use_layer_norm=critic_layer_norm,
            use_pnorm=use_pnorm,
        )
    observations = batch["observations"]
    actions = batch["actions"]

    critic_cls = partial(StateActionValue, base_cls=critic_base_cls)
    critic_def = Ensemble(critic_cls, num=num_qs)
    critic_params = critic_def.init(key, observations, actions)["params"]
    if critic_weight_decay is not None:
        if max_gradient_norm is not None:
            tx = optax.chain(
                optax.clip_by_global_norm(max_gradient_norm),
                optax.adamw(
                    learning_rate=critic_lr,
                    weight_decay=critic_weight_decay,
                    mask=decay_mask_fn,
                ),
            )
        else:
            tx = optax.adamw(
                learning_rate=critic_lr,
                weight_decay=critic_weight_decay,
                mask=decay_mask_fn,
            )
    elif max_gradient_norm is not None:
        tx = optax.chain(
            optax.clip_by_global_norm(max_gradient_norm),
            optax.adam(learning_rate=critic_lr),
        )
    else:
        tx = optax.adam(learning_rate=critic_lr)

    critic = TrainState.create(
        apply_fn=critic_def.apply,
        params=critic_params,
        tx=tx,
    )
    target_critic_def = Ensemble(critic_cls, num=num_min_qs or num_qs)
    target_critic = TrainState.create(
        apply_fn=target_critic_def.apply,
        params=critic_params,
        tx=optax.GradientTransformation(lambda _: None, lambda _: None),
    )

    return critic, target_critic


class SACLearner(Agent):
    critic: TrainState
    target_critic: TrainState
    temp: TrainState
    tau: float
    discount: float
    target_entropy: float
    num_qs: int = struct.field(pytree_node=False)
    num_min_qs: int | None = struct.field(
        pytree_node=False,
    )  # See M in RedQ https://arxiv.org/abs/2101.05982
    backup_entropy: bool = struct.field(pytree_node=False)
    interior_linear_c: float = struct.field(pytree_node=False)
    interior_quadratic_c: float = struct.field(pytree_node=False)
    exterior_linear_c: float = struct.field(pytree_node=False)
    exterior_quadratic_c: float = struct.field(pytree_node=False)
    max_gradient_norm: float = struct.field(pytree_node=False)

    @classmethod
    def create(
        cls,
        seed: int,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        temp_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        discount: float = 0.99,
        tau: float = 0.005,
        num_qs: int = 2,
        num_min_qs: int | None = None,
        critic_dropout_rate: float | None = None,
        critic_weight_decay: float | None = None,
        max_gradient_norm: float | None = None,
        critic_layer_norm: bool = False,
        target_entropy: float | None = None,
        init_temperature: float = 1.0,
        backup_entropy: bool = True,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        interior_linear_c: float = 0.0,
        interior_quadratic_c: float = 0.0,
        exterior_linear_c: float = 0.0,
        exterior_quadratic_c: float = 0.0,
    ):
        """An implementation of the version of Soft-Actor-Critic described in https://arxiv.org/abs/1812.05905."""
        action_dim = action_space.shape[-1]
        observations = observation_space.sample()
        actions = action_space.sample()

        if target_entropy is None:
            target_entropy = -action_dim / 2

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key, temp_key = jax.random.split(rng, 4)

        actor = make_mlp_actor(
            hidden_dims,
            use_pnorm,
            action_dim,
            observations,
            actor_key,
            actor_lr,
        )

        critic, target_critic = make_critic(
            use_critic_resnet,
            hidden_dims,
            critic_dropout_rate,
            critic_layer_norm,
            use_pnorm,
            num_qs,
            critic_key,
            observations,
            actions,
            critic_weight_decay,
            max_gradient_norm,
            critic_lr,
            num_min_qs,
        )

        temp = make_temp(init_temperature, temp_key, temp_lr)

        return cls(
            rng=rng,
            actor=actor,
            critic=critic,
            target_critic=target_critic,
            temp=temp,
            target_entropy=target_entropy,
            tau=tau,
            discount=discount,
            num_qs=num_qs,
            num_min_qs=num_min_qs,
            backup_entropy=backup_entropy,
            interior_linear_c=interior_linear_c,
            interior_quadratic_c=interior_quadratic_c,
            exterior_linear_c=exterior_linear_c,
            exterior_quadratic_c=exterior_quadratic_c,
            max_gradient_norm=max_gradient_norm,
        )

    def reset_actor(
        self,
        action_space: gym.Space,
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        use_pnorm: bool = False,
    ):
        action_dim = action_space.shape[-1]
        key, rng = jax.random.split(self.rng)

        actor = make_mlp_actor(
            hidden_dims,
            use_pnorm,
            action_dim,
            self.batch["observations"],
            key,
            actor_lr,
        )

        # Replace the actor with the new random parameters
        return self.replace(actor=actor, rng=rng)

    def reset_critic(
        self,
        critic_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256),
        critic_dropout_rate: float | None = None,
        # critic_weight_decay: float | None = None, # noqa: ERA001
        critic_layer_norm: bool = False,
        use_pnorm: bool = False,
        use_critic_resnet: bool = False,
        max_gradient_norm: float | None = None,
    ):
        key, rng = jax.random.split(self.rng)

        critic, target_critic = make_critic(
            use_critic_resnet,
            hidden_dims,
            critic_dropout_rate,
            critic_layer_norm,
            use_pnorm,
            self.num_qs,
            key,
            self.batch,
            max_gradient_norm,
            critic_lr,
            self.num_min_qs,
        )

        # Replace the critic and target critic with the new random parameters
        return self.replace(critic=critic, target_critic=target_critic, rng=rng)

    def update_actor(
        self,
        batch: DatasetDict,
        output_range: tuple[Array, Array] | None,
    ) -> tuple[Agent, dict[str, float]]:
        key2, key, rng = jax.random.split(self.rng, 3)

        def actor_loss_fn(actor_params) -> tuple[Array, dict[str, float]]:
            dist = self.actor.apply_fn({"params": actor_params}, batch["observations"])
            actions = dist.sample(seed=key)
            log_probs = dist.log_prob(actions)
            qs = self.critic.apply_fn(
                {"params": self.critic.params},
                batch["observations"],
                actions,
                True,
                rngs={"dropout": key2},
            )  # training=True
            q = qs.mean(axis=0)

            # create a mask if the action is out of range (0 if out of range, 1 if in range)
            mask = 1 - jnp.logical_or(
                jnp.any(actions < output_range[0], axis=-1),
                jnp.any(actions > output_range[1], axis=-1),
            )

            exterior_actions = jnp.where(mask[:, None], jnp.zeros_like(actions), actions)
            interior_actions = jnp.where(mask[:, None], actions, jnp.zeros_like(actions))

            exterior_l1_penalty = jnp.sum(jnp.abs(exterior_actions), axis=-1)
            exterior_l2_penalty = jnp.sum(exterior_actions**2, axis=-1)
            exterior_l2_penalty = jnp.sqrt(exterior_l2_penalty)

            # interior_linear_penalty = jnp.sum(jnp.abs(interior_actions), axis=-1) # noqa: ERA001
            interior_quadratic_penalty = jnp.sum(interior_actions**4, axis=-1)

            penalty_function = (
                self.interior_quadratic_c
                * (self.exterior_linear_c / (3 * output_range[1] ** 3))
                * interior_quadratic_penalty
                + self.exterior_linear_c * exterior_l1_penalty
                + self.exterior_quadratic_c * exterior_l2_penalty
            )

            actor_loss = (
                log_probs * self.temp.apply_fn({"params": self.temp.params}) - q + penalty_function
            ).mean()

            # compute how many actions are out of range
            oor_actions = jnp.mean(1 - mask)
            info_dict = {
                "actor_loss": actor_loss,
                "entropy": -log_probs.mean(),
                "oor_actions": oor_actions,
                "penalty_function": penalty_function.mean(),
            }

            return actor_loss, info_dict

        grads, actor_info = jax.grad(actor_loss_fn, has_aux=True)(self.actor.params)
        actor = self.actor.apply_gradients(grads=grads)

        return self.replace(actor=actor, rng=rng), actor_info

    def update_temperature(self, entropy: float) -> tuple[Agent, dict[str, float]]:
        def temperature_loss_fn(temp_params):
            temperature = self.temp.apply_fn({"params": temp_params})
            temp_loss = temperature * (entropy - self.target_entropy).mean()
            return temp_loss, {
                "temperature": temperature,
                "temperature_loss": temp_loss,
            }

        grads, temp_info = jax.grad(temperature_loss_fn, has_aux=True)(self.temp.params)
        temp = self.temp.apply_gradients(grads=grads)

        return self.replace(temp=temp), temp_info

    def update_critic(
        self,
        batch: DatasetDict,
    ) -> tuple[TrainState, dict[str, float]]:
        dist = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"],
        )

        rng = self.rng

        key, rng = jax.random.split(rng)
        next_actions = dist.sample(seed=key)

        # Used only for REDQ.
        key, rng = jax.random.split(rng)
        target_params = subsample_ensemble(
            key,
            self.target_critic.params,
            self.num_min_qs,
            self.num_qs,
        )

        key, rng = jax.random.split(rng)
        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        key, rng = jax.random.split(rng)

        def critic_loss_fn(critic_params) -> tuple[Array, dict[str, float]]:
            qs = self.critic.apply_fn(
                {"params": critic_params},
                batch["observations"],
                batch["actions"],
                True,
                rngs={"dropout": key},
            )  # training=True
            critic_loss = ((qs - target_q) ** 2).mean()
            return critic_loss, {"critic_loss": critic_loss, "q": qs.mean()}

        grads, info = jax.grad(critic_loss_fn, has_aux=True)(self.critic.params)

        # Compute the gradient magnitudes
        critic_grad_magnitudes = l2_norm(grads)

        critic_params_before = self.critic.params
        critic = self.critic.apply_gradients(grads=grads)
        critic_params_after = critic.params

        critic_params_change = l2_distance(
            critic_params_before,
            critic_params_after,
        )

        target_critic_params = optax.incremental_update(
            critic.params,
            self.target_critic.params,
            self.tau,
        )
        target_critic = self.target_critic.replace(params=target_critic_params)

        info["critic_grad_magnitudes"] = critic_grad_magnitudes
        info["critic_weight_change"] = critic_params_change

        return self.replace(critic=critic, target_critic=target_critic, rng=rng), info

    def compute_td_error(self, batch: DatasetDict):
        dist = self.actor.apply_fn(
            {"params": self.actor.params},
            batch["next_observations"],
        )
        key_sample, key_action, key_critic, key_critic_next, self.rng = jax.random.split(
            self.rng,
            5,
        )
        next_actions = dist.sample(seed=key_action)

        # Used only for REDQ.
        target_params = subsample_ensemble(
            key_sample,
            self.target_critic.params,
            self.num_min_qs,
            self.num_qs,
        )

        next_qs = self.target_critic.apply_fn(
            {"params": target_params},
            batch["next_observations"],
            next_actions,
            True,
            rngs={"dropout": key_critic_next},
        )  # training=True
        next_q = next_qs.min(axis=0)

        target_q = batch["rewards"] + self.discount * batch["masks"] * next_q

        if self.backup_entropy:
            next_log_probs = dist.log_prob(next_actions)
            target_q -= (
                self.discount
                * batch["masks"]
                * self.temp.apply_fn({"params": self.temp.params})
                * next_log_probs
            )

        qs = self.critic.apply_fn(
            {"params": self.critic.params},
            batch["observations"],
            batch["actions"],
            True,
            rngs={"dropout": key_critic},
        )  # training=True

        return (qs - target_q) ** 2  # td-error

    @partial(jax.jit, static_argnames=("utd_ratio", "actor_delay"))
    def update_with_delay(
        self,
        batch: DatasetDict,
        utd_ratio: int,
        output_range: tuple[Array, Array] | None = None,
        actor_delay: int = 1,
    ):
        new_agent = self
        for actor_i in range(utd_ratio // actor_delay):
            for critic_i in range(actor_delay):

                def _slice(x):
                    assert x.shape[0] % utd_ratio == 0
                    batch_size = x.shape[0] // utd_ratio
                    start = batch_size * (critic_i + actor_i * actor_delay)
                    end = batch_size * (critic_i + actor_i * actor_delay + 1)
                    return x[start:end]

                mini_batch = jax.tree_util.tree_map(_slice, batch)
                new_agent, critic_info = new_agent.update_critic(mini_batch)

            new_agent, actor_info = new_agent.update_actor(mini_batch, output_range=output_range)
            new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}

    @partial(jax.jit, static_argnames="utd_ratio")
    def update(
        self,
        batch: DatasetDict,
        utd_ratio: int,
        output_range: tuple[Array, Array] | None = None,
    ):
        new_agent = self
        for i in range(utd_ratio):

            def _slice(x):
                assert x.shape[0] % utd_ratio == 0
                batch_size = x.shape[0] // utd_ratio
                return x[batch_size * i : batch_size * (i + 1)]

            mini_batch = jax.tree_util.tree_map(_slice, batch)
            new_agent, critic_info = new_agent.update_critic(mini_batch)

        new_agent, actor_info = new_agent.update_actor(mini_batch, output_range=output_range)
        new_agent, temp_info = new_agent.update_temperature(actor_info["entropy"])

        return new_agent, {**actor_info, **critic_info, **temp_info}

    @jax.jit
    def get_value(self, observation: np.ndarray) -> float:
        # Convert the observation to a JAX numpy array
        jax_observation = convert_to_numpy_array(observation)

        rng = self.rng
        key, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)

        # Use the actor network to generate an action
        dist = self.actor.apply_fn({"params": self.actor.params}, jax_observation[None])
        action = dist.sample(seed=key)[0]  # Generate a single action

        # Use the critic network to estimate the state-action value
        value = self.critic.apply_fn(
            {"params": self.critic.params},
            jax_observation[None],  # Add a batch dimension to the observation
            action[None],  # Add a batch dimension to the action
            True,
            rngs={"dropout": key2},
        )
        # average over the ensemble
        value = value.mean(axis=0)[0]  # Remove the batch dimension
        return jnp.array(value, float)

    @jax.jit
    def self_perception(self, batch: DatasetDict) -> float:
        rng = self.rng
        key, rng = jax.random.split(rng)
        key2, rng = jax.random.split(rng)

        # Use the actor network to generate an action
        dist = self.actor.apply_fn({"params": self.actor.params}, batch["observations"])
        actions = dist.sample(seed=key)

        # Use the critic network to estimate the state-action value
        value = self.critic.apply_fn(
            {"params": self.critic.params},
            batch["observations"],
            actions,
            True,
            rngs={"dropout": key2},
        )
        # average over the ensemble
        value = value.mean(axis=0)
        return jnp.array(value, float)[0]
