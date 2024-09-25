from __future__ import annotations

import functools
from typing import Tuple

import jax
import jax.numpy as jnp
from chex import ArrayTree
from qdax.custom_types import Genotype, RNGKey


def get_tree_keys(
  genotype: Genotype,
  random_key: RNGKey
) -> ArrayTree:
    nb_leaves = len(jax.tree_util.tree_leaves(genotype))
    random_key, subkey = jax.random.split(random_key)
    subkeys = jax.random.split(subkey,
                               num=nb_leaves)
    keys_tree = jax.tree_util.tree_unflatten(jax.tree_util.tree_structure(genotype),
                                   subkeys)
    return keys_tree


def select_index_pytree(pytree_optimised_gens,
                        index_optimised_gen
                        ):
    return jax.tree_map(
        lambda x: x[index_optimised_gen],
        pytree_optimised_gens
    )


def get_batch_size(tree: ArrayTree) -> int:
    if len(jax.tree_util.tree_leaves(tree)) == 0:
        return 0

    batch_size = jax.tree_util.tree_leaves(tree)[0].shape[0]
    return batch_size



def complete_genotype(genotype: Genotype,
                      number_to_complete: int,
                      ) -> Genotype:
    if number_to_complete == 0:
        return genotype
    empty_gen = jax.tree_map(lambda x: jnp.zeros((number_to_complete,) + x.shape[1:]),
                             genotype)
    return jax.tree_map(lambda x, y: jnp.concatenate([x, y], axis=0),
                        genotype,
                        empty_gen)

def truncate_genotype(genotype: Genotype,
                      number_to_truncate: int,
                      ) -> Genotype:
    return jax.tree_map(lambda x: x[:number_to_truncate],
                        genotype)



def custom_vmap(fn, chunk_size, activate_internal_vmap=True):
    @functools.wraps(fn)
    def wrapper(tuple_args):
        batch_size = get_batch_size(tuple_args)

        num_remainder = batch_size % chunk_size
        tuple_args = complete_genotype(tuple_args, num_remainder)

        num_batches = (batch_size+num_remainder) // chunk_size

        tuple_args = jax.tree_map(
            lambda x: x.reshape((num_batches, -1) + x.shape[1:]),
            tuple_args
        )

        if activate_internal_vmap:
            func = jax.jit(jax.vmap(
                lambda xs: fn(*xs),
            ))
        else:
            func = jax.jit(
                lambda xs: fn(*xs),
            )

        res = jax.lax.map(
            func,
            tuple_args
        )

        res = truncate_genotype(res, batch_size)

        res = jax.tree_map(
            lambda x: x.reshape((-1,) + x.shape[2:]),
            res
        )

        return res
    return wrapper