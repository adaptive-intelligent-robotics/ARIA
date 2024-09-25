from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple, List, Union

import flax.struct
import jax
import jax.numpy as jnp
import optax
from qdax.custom_types import Genotype, RNGKey, Descriptor, Centroid
from tqdm import tqdm

from aria.algos.abstract_algo import AbstractAlgo
# Centered rank from: https://arxiv.org/pdf/1703.03864.pdf
from aria.reevaluator_score import ReEvaluator
from aria.reproducibility_improvers.improver_standard import ReproducibilityImprover
from aria.utils import distances_utils
from aria.utils.saving_loading_utils import save_pytree
from aria.utils.tree_utils import select_index_pytree


@flax.struct.dataclass
class UnevaluatedIndividual:
    genotype: Genotype
    centroid: jnp.ndarray
    optimizer_state: optax.OptState

    @classmethod
    def create(cls,
               genotype,
               centroid,
               optimizer_state=None,
               ):
        return cls(
            genotype=genotype,
            centroid=centroid,
            optimizer_state=optimizer_state,
        )

    @classmethod
    def create_empty(cls):
        return cls(
            genotype={},
            centroid={},
            optimizer_state=None,
        )

    def is_empty(self):
        return len(self.centroid) == 0


class RepertoireUnevaluatedIndividuals:
    def __init__(self, unevaluated_individuals: List[UnevaluatedIndividual]):
        self.unevaluated_individuals = unevaluated_individuals

    def add(self, *individuals):
        self.unevaluated_individuals.extend(individuals)

    def get_indiv_from_centroid(self, centroid: jnp.ndarray) -> UnevaluatedIndividual:
        if len(self.unevaluated_individuals) == 0:
            raise ValueError("No individuals left")
        distances = jnp.linalg.norm(self.get_array_centroids() - centroid, axis=1)
        return self.unevaluated_individuals[jnp.argmin(distances)]

    def get_list_centroids(self) -> List[jnp.ndarray]:
        return [indiv.centroid for indiv in self.unevaluated_individuals]

    def get_array_centroids(self) -> jnp.ndarray:
        return jnp.asarray(self.get_list_centroids())

    def get_list_indivs_from_centroids(self,
                                       all_closest_centroid_already_explored: jnp.ndarray) -> List[
        UnevaluatedIndividual]:
        return [self.get_indiv_from_centroid(centroid) for centroid in
                all_closest_centroid_already_explored]

    def get_individuals_from_centroids(self,
                                       all_closest_centroid_already_explored: jnp.ndarray,
                                       ) -> UnevaluatedIndividual:
        list_indivs = self.get_list_indivs_from_centroids(all_closest_centroid_already_explored)
        # list_gens = [indiv.genotype for indiv in list_indivs]

        return jax.tree_map(
            lambda *array_gen: jnp.asarray(array_gen),
            *list_indivs
        )


class ARIA_ES_Init(AbstractAlgo):
    FINAL_REPERTOIRE_FILENAME = "final_repertoire_unevaluated_individuals.pickle"
    FOLDER_PARTIAL_OPTIMISED = "optimised_repertoires"

    def __init__(self,
                 robustness_improver: ReproducibilityImprover,
                 reevaluator: ReEvaluator,
                 centroids: Centroid,
                 number_robust_iterations_initial: int,
                 number_robust_iterations_loop: int,
                 number_parallel_optimisations: int,
                 config,
                 scoring_fn
                 ):
        super().__init__(config,
                         scoring_fn)
        self.robustness_improver = robustness_improver
        self.reevaluator = reevaluator
        self.centroids = centroids

        self.number_robust_iterations_initial = number_robust_iterations_initial
        self.number_robust_iterations_loop = number_robust_iterations_loop

        self.number_parallel_optimisations = number_parallel_optimisations

        self.robustness_improver_vmap = jax.vmap(self.robustness_improver.run, in_axes=(0, 0, 0, None))

    @classmethod
    def save_repertoire(cls, repertoire_unevaluated: RepertoireUnevaluatedIndividuals, path: str):
        save_pytree(data=repertoire_unevaluated, path=str(path), overwrite=True)

    @classmethod
    def save_final_repertoire(cls, repertoire_unevaluated: RepertoireUnevaluatedIndividuals):
        cls.save_repertoire(repertoire_unevaluated=repertoire_unevaluated, path=cls.FINAL_REPERTOIRE_FILENAME)

    def run(self, initial_genotypes: Genotype, random_key: RNGKey) -> RepertoireUnevaluatedIndividuals:

        # ------ choose first centroids as closest to start_bd --------

        random_key, subkey = jax.random.split(random_key)
        optimised_repertoire, list_centroids_to_go = self.initialise_optimised_repertoire(initial_genotypes, subkey)

        self.save_repertoire(repertoire_unevaluated=optimised_repertoire,
                             path=str(Path(self.FOLDER_PARTIAL_OPTIMISED) / "repertoire_after_initialisation.pickle"),
                             )

        random.shuffle(list_centroids_to_go)

        initial_centroid = optimised_repertoire.get_list_centroids()[0]

        count_loop = 0
        with tqdm(total=len(list_centroids_to_go), desc="Optimising to reach remaining centroids") as pbar:
            while list_centroids_to_go:
                pbar.set_postfix({"Centroids left": len(list_centroids_to_go)})

                random_key, subkey = jax.random.split(random_key)
                list_visited_centroids = optimised_repertoire.get_list_centroids()
                pbar.set_postfix({"Centroids left": len(list_centroids_to_go),
                                  "Visited": len(list_visited_centroids)})

                random_key, subkey = jax.random.split(random_key)
                centroids_to_go, index_centroids_to_go, closest_centroids_already_explored = self.choose_best_pair_centroids(
                    list_centroids_to_go,
                    list_visited_centroids,
                    initial_centroid,
                    subkey)

                random_key, subkey = jax.random.split(random_key)
                pytree_optimised_gens, optimizer_states = self.optimise_starting_from_centroids(optimised_repertoire,
                                                                                                init_centroids=closest_centroids_already_explored,
                                                                                                centroids_to_go=centroids_to_go,
                                                                                                random_key=subkey)

                for index_optimised_gen in range(self.number_parallel_optimisations):
                    optimised_gen = select_index_pytree(pytree_optimised_gens, index_optimised_gen)
                    optimizer_state = select_index_pytree(optimizer_states, index_optimised_gen)
                    centroid_to_go = centroids_to_go[index_optimised_gen]
                    optimised_repertoire.add(
                        UnevaluatedIndividual(genotype=optimised_gen,
                                              centroid=centroid_to_go,
                                              optimizer_state=optimizer_state)
                    )

                list_centroids_to_go = [
                    centroid
                    for i, centroid in enumerate(list_centroids_to_go)
                    if i not in index_centroids_to_go.ravel()
                ]

                count_loop += 1
                pbar.update(self.number_parallel_optimisations)
        self.save_final_repertoire(optimised_repertoire)
        return optimised_repertoire

    @classmethod
    def get_path_save_partial_repertoire(cls, suffix: Union[str, int]):
        return Path(cls.FOLDER_PARTIAL_OPTIMISED) / f"partial_optimised_repertoire_iter_{suffix}.pickle"

    def choose_best_pair_centroids(self,
                                   list_centroids_to_go: List[jnp.ndarray],
                                   list_visited_centroids: List[jnp.ndarray],
                                   start_bd: Descriptor,
                                   random_key: RNGKey,
                                   ) -> Tuple[Centroid, jnp.ndarray, Centroid]:
        array_centroids_to_go = jnp.asarray(list_centroids_to_go)
        array_visited_centroids = jnp.asarray(list_visited_centroids)

        # Choosing closest non-visited centroids to start_bd
        vector_dist = distances_utils.v_dist(start_bd, array_centroids_to_go).ravel()
        # argmin_vector_dist_centroids_to_go = jnp.argmin(vector_dist)
        _, argmins_vector_dist_centroids_to_go = jax.lax.top_k(-vector_dist, k=self.number_parallel_optimisations)
        array_centroids_to_go = array_centroids_to_go[argmins_vector_dist_centroids_to_go]

        # Choosing closest visited centroids to centroids_to_go
        def compute_closest_centroid(_centroid_to_go):
            vector_dist = distances_utils.v_dist(_centroid_to_go, array_visited_centroids).ravel()
            argmin_vector_dist_centroids_visited = jnp.argmin(vector_dist)
            return array_visited_centroids[argmin_vector_dist_centroids_visited]

        array_closest_centroids_already_explored = jax.vmap(compute_closest_centroid)(array_centroids_to_go)

        centroids_to_go_indexes = argmins_vector_dist_centroids_to_go
        return array_centroids_to_go, centroids_to_go_indexes, array_closest_centroids_already_explored

    def initialise_optimised_repertoire(self,
                                        initial_gen: Genotype,
                                        random_key: RNGKey,
                                        ):
        """
        Initialises the optimised repertoire with one genotype optimised for maximising the fitness only.
        And then robustify to be closer to its closest centroid.
        """
        optimised_repertoire = RepertoireUnevaluatedIndividuals(
            unevaluated_individuals=[],
        )

        list_centroids_to_go = list(self.centroids)

        # First optimise for fitness only
        self.logger.info("Running ES to optimise for fitness only for one individual")
        random_key, subkey_fitness_improver = jax.random.split(random_key)
        optimized_initial_gen, last_optimizer_state_es = self.robustness_improver.run(initial_gen=initial_gen,
                                                                                      expected_bd=None,  # no expected_bd to only improve fitness
                                                                                      random_key=subkey_fitness_improver,
                                                                                      number_iterations=self.number_robust_iterations_initial,
                                                                                      use_gcrl_scoring_fn=False)

        random_key, subkey_reeval = jax.random.split(random_key)
        add_dimension = True
        estimated_mean_bd = self.get_estimated_bd_from_gen(optimized_initial_gen, add_dimension, subkey_reeval)

        # Choosing the closest centroid to the estimated_mean_bd
        closest_centroid, argmin_closest_centroid = self.find_closest_point_in_array(point_ref=estimated_mean_bd,
                                                                                     points_array=self.centroids)

        # And robustify optimised gen to be sure to get closer to centroid.
        self.logger.info("Robustifying the initial genotype to be closer to the closest centroid")
        random_key, subkey_robustness_improver = jax.random.split(random_key)
        robustified_initial_gen, last_optimizer_state_robust = self.robustness_improver.run(
            initial_gen=optimized_initial_gen,
            expected_bd=closest_centroid,
            random_key=subkey_robustness_improver,
            number_iterations=self.number_robust_iterations_initial,
            use_gcrl_scoring_fn=True,
            optimizer_state=last_optimizer_state_es)

        optimised_repertoire.add(
            UnevaluatedIndividual(genotype=robustified_initial_gen,
                                  centroid=closest_centroid,
                                  optimizer_state=last_optimizer_state_robust,
                                  )
        )

        list_centroids_to_go.pop(argmin_closest_centroid)

        assert len(optimised_repertoire.get_list_centroids()) == 1, "Initialisation should have only one individual"

        return optimised_repertoire, list_centroids_to_go

    # @timeit
    def optimise_starting_from_centroids(self,
                                         optimised_repertoire,
                                         init_centroids,
                                         centroids_to_go,
                                         random_key
                                         ):
        """
        Optimises the genotypes in the optimised repertoire from init_centroids to be closer to the centroids_to_go.
        """
        already_visited_individuals: UnevaluatedIndividual = optimised_repertoire.get_individuals_from_centroids(
            init_centroids)

        random_key, key_robustness_improver = jax.random.split(random_key)
        list_subkeys_robustness_improver = jax.random.split(key_robustness_improver,
                                                            self.number_parallel_optimisations)

        # we use vmap to parallelise the robustness improver on self.number_parallel_optimisations different genotypes & centroids.
        pytree_optimised_gens, optimizer_states = self.robustness_improver_vmap(already_visited_individuals.genotype,
                                                                                centroids_to_go,
                                                                                list_subkeys_robustness_improver,
                                                                                self.number_robust_iterations_loop,
                                                                                optimizer_state=already_visited_individuals.optimizer_state,
                                                                                )
        return pytree_optimised_gens, optimizer_states

    @staticmethod
    def find_closest_point_in_array(point_ref: Descriptor,
                                    points_array: Descriptor
                                    ) -> Tuple[Descriptor, int]:
        vector_dist = distances_utils.v_dist(point_ref,
                                             points_array).ravel()
        argmin_closest_centroid = jnp.argmin(vector_dist)
        closest_point_in_array = points_array[argmin_closest_centroid]
        return closest_point_in_array, argmin_closest_centroid

    def get_estimated_bd_from_gen(self,
                                  optimized_initial_gen,
                                  add_dimension: bool,
                                  random_key,
                                  ):
        random_key, subkey = jax.random.split(random_key)
        _, mean_bds = self.reevaluator.mean_reevals(optimized_initial_gen, subkey, add_dimension=add_dimension)
        return mean_bds
