from __future__ import annotations

import time
from pathlib import Path
from typing import Tuple, List

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import Genotype, RNGKey, Descriptor, Centroid
from qdax.utils.metrics import CSVLogger
from tqdm import tqdm

from aria.aria_es_init import ARIA_ES_Init, RepertoireUnevaluatedIndividuals, UnevaluatedIndividual
# Centered rank from: https://arxiv.org/pdf/1703.03864.pdf
from aria.reevaluator_score import ReEvaluator
from aria.reproducibility_improvers.improver_standard import ReproducibilityImprover
from aria.utils import distances_utils
from aria.utils.distances_utils import neg_novelty
from aria.utils.repertoire_utils import extract_non_empty_cells
from aria.utils.tree_utils import select_index_pytree, get_batch_size


class ARIAMapElitesInit(ARIA_ES_Init):
    def __init__(self,
                 robustness_improver: ReproducibilityImprover,
                 reevaluator: ReEvaluator,
                 centroids: Centroid,
                 number_robust_iterations_initial: int,
                 number_robust_iterations_loop: int,
                 number_parallel_optimisations: int,
                 config,
                 scoring_fn,
                 total_map_elites_evaluations: int,
                 emitter: Emitter,
                 metrics_fn_map_elites,
                 ):
        super().__init__(robustness_improver,
                         reevaluator,
                         centroids,
                         number_robust_iterations_initial,
                         number_robust_iterations_loop,
                         number_parallel_optimisations,
                         config,
                         scoring_fn)

        self.total_map_elites_evaluations = total_map_elites_evaluations

        self.map_elites = MAPElites(
            scoring_function=self.scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn_map_elites,
        )

    def choose_best_pair_centroids(self,
                                   list_centroids_to_go: List[jnp.ndarray],
                                   list_visited_centroids: List[jnp.ndarray],
                                   _: Descriptor,
                                   random_key: RNGKey,
                                   ) -> Tuple[Centroid, jnp.ndarray, Centroid]:

        array_centroids_to_go = jnp.asarray(list_centroids_to_go)
        array_visited_centroids = jnp.asarray(list_visited_centroids)

        # Choosing closest non-visited centroids to start_bd
        num_knn_novelty_scores = 1  # Forced to use 1 if we want to target a cell that is a neighbour of an existing cell
        neg_novelty_scores = jax.vmap(neg_novelty,
                                      in_axes=(0, None, None))(array_centroids_to_go,
                                                               array_visited_centroids,
                                                               num_knn_novelty_scores,
                                                               )

        values, argmins_vector_dist_centroids_to_go = jax.lax.top_k(neg_novelty_scores.ravel(),
                                                                    k=self.number_parallel_optimisations)

        array_centroids_to_go = array_centroids_to_go[argmins_vector_dist_centroids_to_go]

        # Choosing closest visited centroids to centroids_to_go
        def compute_closest_centroid(_centroid_to_go):
            vector_dist = distances_utils.v_dist(_centroid_to_go, array_visited_centroids).ravel()
            argmin_vector_dist_centroids_visited = jnp.argmin(vector_dist)
            return array_visited_centroids[argmin_vector_dist_centroids_visited]

        array_closest_centroids_already_explored = jax.vmap(compute_closest_centroid)(array_centroids_to_go)

        centroids_to_go_indexes = argmins_vector_dist_centroids_to_go
        return array_centroids_to_go, centroids_to_go_indexes, array_closest_centroids_already_explored

    def get_map_elites_repertoire(self,
                                  initial_gen: Genotype,
                                  random_key: RNGKey,
                                  ):
        counter_evaluations = 0

        random_key, subkey_reevaluate = jax.random.split(random_key)
        repertoire, emitter_state, _ = self.map_elites.init(initial_gen,
                                                            self.centroids,
                                                            subkey_reevaluate, )
        counter_evaluations += get_batch_size(initial_gen)

        index_generation = 0

        csv_logger = CSVLogger(
            "metrics.csv",
            header=["evaluations", "loop", "qd_score", "max_fitness", "coverage", "time"]
        )

        start_time = time.time()

        with tqdm(total=self.total_map_elites_evaluations, desc="Map-Elites Progress") as pbar:
            while counter_evaluations < self.total_map_elites_evaluations:
                random_key, subkey_update = jax.random.split(random_key)
                repertoire, emitter_state, metrics, _ = self.map_elites.update(repertoire,
                                                                               emitter_state=emitter_state,
                                                                               random_key=subkey_update)

                if index_generation % 20 == 0:
                    timelapse = time.time() - start_time
                    logged_metrics = {"time": timelapse, "loop": 1 + index_generation,
                                      "evaluations": counter_evaluations}
                    logged_metrics = {**logged_metrics, **metrics}
                    csv_logger.log(logged_metrics)

                index_generation += 1
                batch_size = self.map_elites._emitter.batch_size
                counter_evaluations += batch_size
                pbar.update(batch_size)
        return repertoire

    def initialise_optimised_repertoire(self,
                                        initial_gen: Genotype,
                                        random_key: RNGKey,
                                        ) -> Tuple[RepertoireUnevaluatedIndividuals, List[Centroid]]:

        random_key, subkey_map_elites = jax.random.split(random_key)
        repertoire = self.get_map_elites_repertoire(initial_gen, subkey_map_elites)

        self.save_repertoire(repertoire,
                             str(Path(self.FOLDER_PARTIAL_OPTIMISED) / "map_elites_lucky_repertoire.pickle"))

        random_key, subkey_reevaluate = jax.random.split(random_key)
        repertoire_reeval = self.reevaluate_repertoire(repertoire, subkey_reevaluate)

        self.save_repertoire(repertoire_reeval,
                             str(Path(self.FOLDER_PARTIAL_OPTIMISED) / "map_elites_reevaluated_repertoire.pickle"))

        random_key, subkey_improve_reproducibility = jax.random.split(random_key)
        optimised_repertoire, list_centroids_to_go = self.get_optimised_repertoire_from_repertoire(repertoire_reeval,
                                                                                                   subkey_improve_reproducibility)

        return optimised_repertoire, list_centroids_to_go

    def get_optimised_repertoire_from_repertoire(self, repertoire_reeval: MapElitesRepertoire, random_key: RNGKey) \
            -> Tuple[RepertoireUnevaluatedIndividuals, List[Centroid]]:
        optimised_repertoire = RepertoireUnevaluatedIndividuals(
            unevaluated_individuals=[],
        )

        list_centroids_to_go = list(self.centroids)

        filtered_genotypes, _, _, filtered_centroids = extract_non_empty_cells(repertoire_reeval)

        num_genotypes = get_batch_size(filtered_genotypes)

        with tqdm(total=num_genotypes, desc="Running Robustness Improver") as pbar:
            for index_genotype in range(num_genotypes):
                progress_info = f"Improving robustness for genotype {index_genotype}/{num_genotypes} in repertoire"
                pbar.set_description(progress_info)

                selected_genotype = select_index_pytree(pytree_optimised_gens=filtered_genotypes,
                                                        index_optimised_gen=index_genotype)
                selected_centroid = select_index_pytree(pytree_optimised_gens=filtered_centroids,
                                                        index_optimised_gen=index_genotype)

                random_key, subkey_robustness_improver = jax.random.split(random_key)
                robustified_initial_gen, last_optimizer_state_robust = self.robustness_improver.run(
                    initial_gen=selected_genotype,
                    expected_bd=selected_centroid,
                    random_key=subkey_robustness_improver,
                    number_iterations=self.number_robust_iterations_loop,
                    use_gcrl_scoring_fn=True,
                    optimizer_state=None,
                )

                optimised_repertoire.add(
                    UnevaluatedIndividual.create(genotype=robustified_initial_gen,
                                                 centroid=selected_centroid,
                                                 optimizer_state=last_optimizer_state_robust,
                                                 )
                )

                list_centroids_to_go = [
                    centroid
                    for centroid in list_centroids_to_go
                    if not jnp.allclose(centroid, selected_centroid)
                ]

                pbar.update(1)
        return optimised_repertoire, list_centroids_to_go

    def reevaluate_repertoire(self, repertoire: MapElitesRepertoire, random_key: RNGKey) -> MapElitesRepertoire:
        """
        Reevaluate the repertoire and return the repertoire resulting from the mean of the reevaluations.
        """
        filtered_genotypes, filtered_fitnesses, filtered_descriptors, filtered_centroids = extract_non_empty_cells(
            repertoire)

        num_genotypes = get_batch_size(filtered_genotypes)

        list_mean_fitnesses = []
        list_descriptors = []

        for index_genotype in range(num_genotypes):
            selected_genotype = select_index_pytree(filtered_genotypes, index_genotype)

            random_key, subkey_reevaluate = jax.random.split(random_key)
            mean_fit, mean_desc = self.reevaluator.mean_reevals(selected_genotype, subkey_reevaluate,
                                                                add_dimension=True)
            list_mean_fitnesses.append(mean_fit)
            list_descriptors.append(mean_desc)

        array_fitnesses = jnp.asarray(list_mean_fitnesses).ravel()
        array_descriptors = jnp.asarray(list_descriptors).reshape((num_genotypes, -1))

        repertoire_reevals = MapElitesRepertoire.init(
            genotypes=filtered_genotypes,
            fitnesses=array_fitnesses,
            descriptors=array_descriptors,
            centroids=self.centroids,
            extra_scores=None,
        )

        return repertoire_reevals
