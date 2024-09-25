from __future__ import annotations

import jax
import jax.numpy as jnp
from qdax.custom_types import Centroid, Descriptor

from aria.utils.distances_utils import v_dist_inf_norm, \
    automatic_cell_inf_norm_radius_calculator


class ProbabilityCellEuclideanGrid:
    """
    Class to calculate the probability of belonging to a cell in a grid
    """

    def __init__(self,
                 grid_shape,
                 min_bd,
                 max_bd,
                 ):
        self.grid_shape = grid_shape
        self.min_bd = min_bd
        self.max_bd = max_bd
        self.radius_neighborhood = automatic_cell_inf_norm_radius_calculator(self.grid_shape, self.min_bd, self.max_bd)

    def _detect_closest_centroid(self, one_desc: Descriptor):
        grid_shape_array = jnp.asarray(self.grid_shape)

        coordinates_cell_in_grid = jnp.floor((one_desc - self.min_bd) / (self.max_bd - self.min_bd) * grid_shape_array)
        coordinates_cell_in_grid += 0.5
        coordinates_centroid = coordinates_cell_in_grid * (self.max_bd - self.min_bd) / grid_shape_array + self.min_bd
        return coordinates_centroid

    def calculate_proba_belong_to_cell_closest_centroid(self,
                                                        descriptors_v: Descriptor,
                                                        ):
        mean_descs = jnp.mean(descriptors_v, axis=0)
        closest_centroid = self._detect_closest_centroid(mean_descs)
        all_probas = self.calculate_proba_belong_to_cell(descriptors_v, closest_centroid)
        return jnp.max(all_probas)

    def calculate_proba_belong_to_cell_closest_centroid_v(self,
                                                          descriptors_bv: Descriptor,
                                                          ):
        all_probas = jax.vmap(self.calculate_proba_belong_to_cell_closest_centroid)(descriptors_bv)
        return all_probas

    def calculate_proba_belong_to_cell(self,
                                       descriptors_v: Descriptor,
                                       centroid: Centroid,
                                       ) -> float:
        distance_inf_norm = v_dist_inf_norm(centroid, descriptors_v)

        does_belong_to_cell = jnp.where(distance_inf_norm < self.radius_neighborhood, 1, 0)
        proba_belong_to_cell = jnp.mean(does_belong_to_cell)
        return proba_belong_to_cell

    def calculate_proba_belong_to_cell_v(self,
                                         descriptors_bv: Descriptor,
                                         centroids_b: Centroid,
                                         ):
        all_probas = jax.vmap(self.calculate_proba_belong_to_cell)(descriptors_bv, centroids_b)

        return all_probas


def test():
    a = jnp.arange(10).reshape(5, 2)
    centroid = jnp.array([1, 2])
    proba_cell = ProbabilityCellEuclideanGrid(grid_shape=(5, 5),
                                              min_bd=0,
                                              max_bd=5)

    closest_centroid = proba_cell._detect_closest_centroid(jnp.array([1.9, 2.1]))
    print(a, centroid)
    print("proba", closest_centroid)


if __name__ == '__main__':
    test()
