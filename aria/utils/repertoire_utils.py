from typing import Tuple

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Fitness, Descriptor, Genotype, Centroid

from aria.utils.tree_utils import select_index_pytree


def get_indices_non_empty_cells(repertoire: MapElitesRepertoire) -> jnp.ndarray:
    condition = ~jnp.isinf(repertoire.fitnesses)
    indices = jnp.asarray(condition).nonzero()[0]
    return indices.ravel()

def extract_non_empty_cells(repertoire: MapElitesRepertoire) -> Tuple[
    Genotype, Fitness, Descriptor, Centroid]:
    indices = get_indices_non_empty_cells(repertoire)
    
    filtered_genotypes = select_index_pytree(repertoire.genotypes, indices)
    filtered_fitnesses = select_index_pytree(repertoire.fitnesses, indices)
    filtered_descriptors = select_index_pytree(repertoire.descriptors, indices)
    filtered_centroids = select_index_pytree(repertoire.centroids, indices)

    return filtered_genotypes, filtered_fitnesses, filtered_descriptors, filtered_centroids