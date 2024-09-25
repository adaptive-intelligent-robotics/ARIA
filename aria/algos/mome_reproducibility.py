import functools
from typing import Tuple, Callable

import jax
import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.mome import MOME
from qdax.custom_types import Genotype, RNGKey, Fitness, Descriptor, ExtraScores, Metrics, \
    Centroid

from aria.algos.abstract_algo import ReevaluationBasedAlgo
from aria.algos.qdax_algos.mome import MOMEFixed
from aria.counter_evals import CounterEvals


class MOMEReproducibilityAlgo(ReevaluationBasedAlgo):
    def __init__(self,
                 config,
                 scoring_fn,
                 centroids: Centroid,
                 emitter: Emitter,
                 metrics_fn: Callable[[MapElitesRepertoire], Metrics],
                 ):
        super().__init__(config,
                         scoring_fn,
                         centroids)

        self.pareto_front_max_length = config.algo.pareto_front_max_length

        self._map_elites = MOMEFixed(
            scoring_function=self.mome_scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
        )

    @property
    def map_elites(self) -> MOMEFixed:
        return self._map_elites

    @functools.partial(jax.jit, static_argnames=('self',))
    def mome_scoring_fn(self, genotype: Genotype, random_key) -> Tuple[Fitness, Descriptor, ExtraScores, RNGKey]:
        random_key, subkey = jax.random.split(random_key)
        fitnesses_bv, descriptors_bv = self.evaluate_keep_reevals(genotype, subkey)
        mean_fitnesses_b, mean_descriptors_b = jax.vmap(self.get_means)(fitnesses_bv, descriptors_bv)

        score_2 = self.get_second_score(descriptors_bv, )

        fitnesses = jnp.stack([mean_fitnesses_b, score_2], axis=-1)

        return fitnesses, mean_descriptors_b, {}, random_key

    def run(self,
            initial_genotypes: Genotype,
            random_key: RNGKey
            ):

        random_key, subkey_init = jax.random.split(random_key)
        repertoire, emitter_state, _ = self.map_elites.init(
            initial_genotypes,
            self.centroids,
            pareto_front_max_length=self.pareto_front_max_length,
            random_key=subkey_init,
        )

        while not self.counter.should_stop():
            self.counter.increment_standard_sampling_size()
            if self.counter.counter_increments % self.config.algo.log_every == 0:
                self.counter.print_info()

            random_key, subkey_update = jax.random.split(random_key)
            repertoire, emitter_state, metrics, _ = self.map_elites.update(repertoire,
                                                                           emitter_state=emitter_state,
                                                                           random_key=subkey_update)
            
        self.save_final_repertoire(repertoire)
