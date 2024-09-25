import abc
from abc import ABC
from typing import Tuple
import logging

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.custom_types import Genotype, RNGKey, Fitness, Descriptor
from qdax.utils.metrics import CSVLogger

from aria.counter_evals import CounterEvals
from aria.metrics.proba_cell import ProbabilityCellEuclideanGrid
from aria.utils.normaliser import Normaliser, NoNormaliser
from aria.utils.saving_loading_utils import save_pytree
from aria.utils.tree_utils import get_batch_size
from aria.utils.types import Distance
from aria.utils.distances_utils import v_dist


class AbstractAlgo(abc.ABC):
    def __init__(self, config, scoring_fn):
        self.config = config
        self.scoring_fn = scoring_fn

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    @abc.abstractmethod
    def run(self, initial_genotypes: Genotype, random_key: RNGKey):
        ...


class ReevaluationBasedAlgo(AbstractAlgo,
                            ABC):
    """
    Abstract class for reevaluation-based algorithms i.e. algorithms that reevaluate the fitness and descriptor of
    genotypes multiple times, to get a more accurate estimate of the mean fitness and descriptor.
    """

    FINAL_REPERTOIRE_FILENAME = "final_repertoire.pickle"

    def __init__(self, config, scoring_fn, centroids):
        super().__init__(config, scoring_fn)

        self.counter = CounterEvals.create_from_config(config)
        self.centroids = centroids

        self.evals_per_scoring_call_per_gen = config.task.reeval.evals_per_gen

        self.probability_calculator = ProbabilityCellEuclideanGrid(
            grid_shape=self.config.task.grid_shape,
            min_bd=self.config.task.min_bd,
            max_bd=self.config.task.max_bd,
        )

    def evaluate_keep_reevals(self,
                              genotype,
                              random_key,
                              ) -> Tuple[Fitness, Descriptor]:
        random_key, subkey = jax.random.split(random_key)
        array_keys_v = jax.random.split(subkey, num=self.evals_per_scoring_call_per_gen)
        array_keys_v = jnp.asarray(array_keys_v)
        fitnesses_bv, descriptors_bv, _, _ = jax.vmap(self.scoring_fn, in_axes=(None, 0), out_axes=1)(genotype, array_keys_v)

        return fitnesses_bv, descriptors_bv

    @classmethod
    def get_mean_fitnesses(cls, fitness_v: Fitness):
        mean_fitnesses = jnp.mean(fitness_v)
        return mean_fitnesses

    @classmethod
    def get_mean_descriptors(cls, descriptor_v: Descriptor):
        mean_descriptors = jnp.mean(descriptor_v, axis=0)
        return mean_descriptors

    @classmethod
    def get_means(cls, fitness_v: Fitness, descriptor_v: Descriptor) -> Tuple[Fitness, Descriptor]:
        mean_fitnesses = cls.get_mean_fitnesses(fitness_v)
        mean_descriptors = cls.get_mean_descriptors(descriptor_v)

        return mean_fitnesses, mean_descriptors

    @classmethod
    def unbiased_variance_desc(cls, descriptor_v: Descriptor) -> Distance:
        variance_desc_value = jnp.sum(jnp.var(descriptor_v, axis=0, ddof=1))  # ddof=1 for unbiased estimator

        return variance_desc_value

    def get_probability_belongs_to_current_cell(self,
                                                descriptor_v: Descriptor,
                                                ):
        """
        Returns the probability of a descriptor belonging to the cell of the closest centroid
        """
        return self.probability_calculator.calculate_proba_belong_to_cell_closest_centroid(descriptor_v)

    def get_second_score(self, descriptors_bv: Descriptor, dist_normaliser: Normaliser = None):
        """
        Returns the second score of the QD algorithm
        Used by Multi-Objective ME as second score, as well as by naive sampling.
        """

        if dist_normaliser is None:
            dist_normaliser = NoNormaliser()

        if self.config.algo.use_proba_fit:
            score_2 = jax.vmap(self.get_probability_belongs_to_current_cell)(descriptors_bv)
        else:  # Using mean distance
            score_2 = -1. * dist_normaliser(jax.vmap(self.unbiased_variance_desc)(descriptors_bv))

        return score_2

    @property
    @abc.abstractmethod
    def map_elites(self):
        ...

    @classmethod
    def create_csv_logger(cls):
        csv_logger = CSVLogger(
            "qd-logs.csv",
            header=["evaluations", "qd_score", "max_fitness", "coverage"]
        )
        return csv_logger

    @classmethod
    def save_final_repertoire(cls, repertoire: MapElitesRepertoire):
        save_pytree(data=repertoire, path=cls.FINAL_REPERTOIRE_FILENAME, overwrite=True)

    def run(self, initial_genotypes: Genotype, random_key: RNGKey):
        random_key, subkey_init = jax.random.split(random_key)
        repertoire, emitter_state, _ = self.map_elites.init(initial_genotypes, self.centroids, random_key=subkey_init)
        self.counter.increment_custom_sampling_size(get_batch_size(initial_genotypes))

        csv_logger = self.create_csv_logger()

        while not self.counter.should_stop():

            if self.counter.counter_increments % self.config.algo.log_every == 0:
                self.counter.print_info()

            self.counter.increment_standard_sampling_size()

            random_key, subkey_update = jax.random.split(random_key)
            repertoire, emitter_state, metrics, _ = self.map_elites.update(repertoire,
                                                                           emitter_state=emitter_state,
                                                                           random_key=subkey_update)

            csv_logger.log(
                metrics={
                    "evaluations": self.counter.counter_evals,
                    **metrics,
                }
            )

        self.save_final_repertoire(repertoire)
