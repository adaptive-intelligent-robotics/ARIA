from __future__ import annotations

from typing import Callable

from jax import numpy as jnp
from qdax.custom_types import Fitness

from aria.reproducibility_improvers.improver_standard import ReproducibilityImprover
from aria.reproducibility_improvers.fitness_shaping import FitnessShaping
from aria.utils.types import Distance


class ReproducibilityImproverQDRLinearCombination(ReproducibilityImprover):
    def __init__(self,
                 perturbation_std: float,
                 population_size: int,
                 scoring_fn: Callable,
                 weight_fitness_obj: float,
                 fitness_normaliser: Callable[[Fitness], Fitness],
                 distance_normaliser: Callable[[Distance], Distance],
                 fitness_shaping: FitnessShaping,
                 learning_rate: float,
                 center_fitness: bool,
                 ):
        super().__init__(perturbation_std,
                         population_size,
                         scoring_fn,
                         fitness_shaping,
                         learning_rate,
                         center_fitness,
                         )

        self.weight_fitness_obj = weight_fitness_obj
        assert 0. <= self.weight_fitness_obj <= 1.
        self.fitness_normaliser = fitness_normaliser
        self.distance_normaliser = distance_normaliser

    def get_gcrl_scoring_fn(self, scoring_fn):
        def gcrl_scoring_fn(gen, goal_desc, random_key):
            fit, desc, infos, random_key = scoring_fn(gen, random_key)

            dist_array = jnp.linalg.norm(desc - goal_desc, ord=2, axis=1)
            dist_score = -1. * dist_array

            gcrl_fit = \
                self.weight_fitness_obj * self.fitness_normaliser(fit) \
                + (1. - self.weight_fitness_obj) * self.distance_normaliser(dist_score)

            print("dist scores:", jnp.mean(dist_score))
            print("fitness", jnp.mean(fit))
            print("gcrl_fit", jnp.mean(gcrl_fit))

            return gcrl_fit, desc, infos, random_key
        return gcrl_scoring_fn
