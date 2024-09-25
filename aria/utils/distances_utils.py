from functools import partial

import jax
import jax.numpy as jnp
import numpy as np


def dist(v, w, p=2):
    return jnp.linalg.norm(v.ravel() - w.ravel(), ord=p)

v_dist = jax.vmap(dist, in_axes=(None, 0))
vv_dist = jax.vmap(v_dist, in_axes=(0, None))

pairwise_dist = jax.vmap(dist)

v_dist_inf_norm = jax.vmap(partial(dist, p=jnp.inf), in_axes=(None, 0))


def automatic_cell_inf_norm_radius_calculator(grid_shape,
                                              min_bd,
                                              max_bd
                                              ):
    grid_shape = tuple(grid_shape)
    assert np.all(np.array(grid_shape) == grid_shape[
        0]), "only uniform grid supported for the moment"
    total_length = max_bd - min_bd
    cell_length = total_length / grid_shape[0]
    radius = cell_length / 2.
    assert max_bd > min_bd
    assert radius > 0.
    return radius


def neg_novelty(vec, reference_array, num_nearest_neighbors: int,):
    array_dist = v_dist(vec.ravel(), reference_array)
    novelty_score, _ = jax.lax.top_k(-array_dist, k=num_nearest_neighbors)
    return jnp.mean(novelty_score)

def novelty(vec, reference_array, num_nearest_neighbors: int,):
    return -neg_novelty(vec, reference_array, num_nearest_neighbors)

def main():
    l = jnp.array([[4, 2, 3], [1, 2, 3], [1, 2, 3]])
    w = jnp.array([[-1, 2, 3], [4, 2, 3]])
    matrix_dist = vv_dist(l, w)
    argmin_w = jnp.argmin(matrix_dist, axis=1)

    random_key = jax.random.PRNGKey(0)
    random_key, key_1, key_2 = jax.random.split(random_key, num=3)
    x = jax.random.uniform(key=key_1, shape=(50, 2, 1))
    y = jax.random.uniform(key=key_2, shape=(50, 2))

    print("matrix_dist", matrix_dist)
    num_nearest_neighbours = 2
    neg_novelty_scores = jax.vmap(neg_novelty, in_axes=(0, None, None))(w,
                                                                        l,
                                                                        num_nearest_neighbours)
    num_individuals_to_choose = 1
    _, indexes = jax.lax.top_k(neg_novelty_scores.ravel(), k=num_individuals_to_choose)
    print("indexes", indexes)
    print("neg_novelty_scores", neg_novelty_scores)

    print(novelty(jnp.asarray([4,2,3]), l, 2))




if __name__ == '__main__':
    main()
