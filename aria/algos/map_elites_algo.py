from typing import Callable, Tuple

import jax
from qdax.core.containers.mapelites_repertoire import MapElitesRepertoire
from qdax.core.emitters.emitter import Emitter
from qdax.core.map_elites import MAPElites
from qdax.custom_types import RNGKey, Genotype, Fitness, Descriptor, ExtraScores, Metrics, \
    Centroid

from aria.algos.abstract_algo import ReevaluationBasedAlgo, AbstractAlgo
from aria.counter_evals import CounterEvals
from aria.utils.types import Distance


class MAPElitesAlgo(ReevaluationBasedAlgo):
    def __init__(self,
                 config,
                 scoring_fn: Callable[
                     [Genotype, RNGKey],
                     Tuple[Fitness, Descriptor, ExtraScores, RNGKey],
                 ],
                 centroids: Centroid,
                 emitter: Emitter,
                 metrics_fn: Callable[[MapElitesRepertoire], Metrics],
                 ):
        super().__init__(config,
                         scoring_fn,
                         centroids,
                         )
        self._map_elites = MAPElites(
            scoring_function=self.scoring_fn,
            emitter=emitter,
            metrics_function=metrics_fn,
        )

    @property
    def map_elites(self):
        return self._map_elites
