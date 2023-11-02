import flax
import jax
import jax.numpy as jnp


def convert_to_numpy_array(param):
    if isinstance(param, list):
        return jax.tree_map(convert_to_numpy_array, param)
    return jnp.array(param)


def tree_multimap(func, tree1, tree2):
    """Apply a function element-wise to two trees."""
    return jax.tree_map(lambda x, y: func(x, y), tree1, tree2)


# From https://colab.research.google.com/github/huggingface/notebooks/blob/master/examples/text_classification_flax.ipynb#scrollTo=ap-zaOyKJDXM
def decay_mask_fn(params):
    flat_params = flax.traverse_util.flatten_dict(params)
    flat_mask = {path: path[-1] != "bias" for path in flat_params}
    return flax.core.FrozenDict(flax.traverse_util.unflatten_dict(flat_mask))
