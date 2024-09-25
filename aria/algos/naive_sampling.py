from typing import Callable, Tuple

import jax
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import RNGKey, Genotype, Fitness, Descriptor, ExtraScores, Metrics, \
    Centroid

from aria.algos.abstract_algo import ReevaluationBasedAlgo
from aria.counter_evals import CounterEvals
from aria.utils.normaliser import Normaliser
from aria.utils.types import Distance


class NaiveSampling(ReevaluationBasedAlgo):
    def __init__(self,
                 config,
                 scoring_fn: Callable[
                     [Genotype, RNGKey],
                     Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
                 ],
                 centroids: Centroid,
                 fitness_normaliser: Normaliser,
                 dist_normaliser: Normaliser,
                 emitter: Emitter,
                 metrics_fn: Callable[[MapElitesRepertoire], Metrics],
                 ):
        super().__init__(config,
                         scoring_fn,
                         centroids,
                         )

        config_algo = config.algo
        self.weight_fitness_obj = config_algo.weight_fitness_obj
        assert 0. <= self.weight_fitness_obj <= 1.

        self.fitness_normaliser = fitness_normaliser
        self.dist_normaliser = dist_normaliser

        self._map_elites = MAPElites(
            scoring_function=self.reeval_scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
        )

    @property
    def map_elites(self):
        return self._map_elites

    def reeval_scoring_fn(self, genotype: Genotype, random_key) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        fitnesses_bv, descriptors_bv = self.evaluate_keep_reevals(genotype, subkey)
        mean_fitnesses_b, mean_descriptors_b = jax.vmap(self.get_means)(fitnesses_bv, descriptors_bv)

        score_2 = self.get_second_score(descriptors_bv=descriptors_bv,
                                        dist_normaliser=self.dist_normaliser)

        true_fitnesses = self.weight_fitness_obj * self.fitness_normaliser(mean_fitnesses_b) + \
                         (1. - self.weight_fitness_obj) * score_2

        return true_fitnesses, mean_descriptors_b, {}, random_key
