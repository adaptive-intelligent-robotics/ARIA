from typing import Tuple, Callable
import os

import jax
import jax.numpy as jnp
from chex import ArrayTree
from qdax.tasks.arm import arm

from qdax.custom_types import Descriptor, ExtraScores, Fitness, Genotype, RNGKey

from aria.utils.tree_utils import get_tree_keys


def arm_unbounded_scoring_function(params: Genotype, random_key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
    # Putting all params between 0 and 1
    params = jnp.mod(params, 1.)

    fitness, descriptor = jax.vmap(arm)(params)

    return fitness, descriptor, {}, random_key

def make_noisy_scoring_function(
    scoring_fn: Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]],
    std_params: float,
    std_fitness: float,
    std_descriptor: float,
) -> Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]]:
    """
    Returns a scoring function that adds noise to the original scoring function.
    """
    def noisy_scoring_fn(params: Genotype, random_key: RNGKey) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        random_key, p_subkey = jax.random.split(random_key)

        # noise = jax.random.normal(subkey, shape=params.shape) * params_std
        # noisy_genotype = jax.tree_map(lambda x: x + noise, params)
        tree_keys = get_tree_keys(genotype=params, random_key=p_subkey)
        noisy_params = jax.tree_util.tree_map(lambda x, y: x + jax.random.normal(y, shape=x.shape) * std_params,
                                              params,
                                              tree_keys)

        random_key, key_scoring = jax.random.split(random_key)
        fitness, descriptor, extra_scores, _ = scoring_fn(noisy_params, key_scoring)

        random_key, f_subkey, d_subkey = jax.random.split(random_key, num=3)
        fitness_noisy = fitness + jax.random.normal(f_subkey, shape=fitness.shape) * std_fitness
        descriptor_noisy = descriptor + jax.random.normal(d_subkey, shape=descriptor.shape) * std_descriptor

        return fitness_noisy, descriptor_noisy, extra_scores, random_key

    return noisy_scoring_fn
