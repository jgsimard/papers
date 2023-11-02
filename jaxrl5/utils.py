import flax
import jax
import jax.numpy as jnp


def convert_to_numpy_array(param):
    if isinstance(param, list):
        return jax.tree_map(convert_to_numpy_array, param)
    return jnp.array(param)


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))


def l2_distance(before_params, after_params):
    squares = jax.tree_map(lambda p1, p2: jnp.sum((p2 - p1) ** 2), before_params, after_params)
    return jnp.sqrt(jnp.sum(jnp.array(jax.tree_util.tree_leaves(squares))))


def l2_norm(grads):
    squares = jax.tree_map(lambda g: jnp.sum(g**2), grads)
    return jnp.sqrt(jnp.sum(jnp.array(jax.tree_util.tree_leaves(squares))))
