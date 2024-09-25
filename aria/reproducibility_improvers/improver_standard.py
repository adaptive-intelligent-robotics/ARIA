from __future__ import annotations

import abc
import functools
from typing import Callable, Tuple
import logging

import jax
import optax
from jax import numpy as jnp
from qdax.custom_types import Params, RNGKey, Genotype, Descriptor

from aria.reproducibility_improvers.fitness_shaping import FitnessShaping


class ReproducibilityImprover(abc.ABC):
    def __init__(self,
                 perturbation_std: float,
                 population_size: int,
                 scoring_fn: Callable,
                 fitness_shaping: FitnessShaping,
                 learning_rate: float,
                 center_fitness: bool,
                 ):
        self.perturbation_std = perturbation_std
        self.population_size = population_size
        self.l2coeff = 0.0
        self.scoring_fn = scoring_fn
        self.gcrl_scoring_fn = self.get_gcrl_scoring_fn(scoring_fn)

        self.fitness_shaping = fitness_shaping

        self.learning_rate = learning_rate
        self.optimizer = optax.adam(learning_rate=self.learning_rate)
        self.center_fitness = center_fitness

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        assert self.center_fitness or self.fitness_shaping == FitnessShaping.CENTERED_RANK

    @abc.abstractmethod
    def get_gcrl_scoring_fn(self, scoring_fn):
        ...

    def add_noise(
        self,
        params: Params,
        key: RNGKey,
    ) -> Tuple[Params, Params, Params]:
        num_vars = len(jax.tree_util.tree_leaves(params))
        treedef = jax.tree_util.tree_structure(params)
        all_keys = jax.random.split(key,
                                    num=num_vars)
        noise = jax.tree_util.tree_map(
            lambda g,
                   k: jax.random.normal(k,
                                        shape=g.shape,
                                        dtype=g.dtype),
            params,
            jax.tree_util.tree_unflatten(treedef,
                                         all_keys)
        )
        params_with_noise = jax.tree_util.tree_map(lambda g,
                                                          n: g + n * self.perturbation_std,
                                                   params,
                                                   noise)
        params_with_anti_noise = jax.tree_util.tree_map(lambda g,
                                                               n: g - n * self.perturbation_std,
                                                        params,
                                                        noise)
        return params_with_noise, params_with_anti_noise, noise

    def calculate_gradient(self,
                           one_param,
                           one_expected_bd,
                           random_key,
                           use_gcrl_scoring_fn=True,
                           ):

        params = jax.tree_util.tree_map(
            lambda x: jnp.repeat(
                jnp.expand_dims(x,
                                axis=0),
                self.population_size,
                axis=0),
            one_param)
        random_key, key_noise, key_es_eval = jax.random.split(random_key,
                                                              3)
        # generate perturbations
        params_with_noise, params_with_anti_noise, noise = self.add_noise(
            params,
            key_noise)

        pparams = jax.tree_util.tree_map(lambda a,
                                                b: jnp.concatenate([a, b],
                                                                   axis=0),
                                         params_with_noise,
                                         params_with_anti_noise)
        if use_gcrl_scoring_fn:
            expected_bd = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x[None, :],
                                     2 * self.population_size,
                                     axis=0),
                one_expected_bd,
            )
            eval_scores, _, _, _ = self.gcrl_scoring_fn(pparams, expected_bd, key_es_eval)
        else:
            eval_scores, _, _, _ = self.scoring_fn(pparams, key_es_eval)

        weights = jnp.reshape(eval_scores,
                              [-1])

        weights = self.fitness_shaping.value(weights)

        if self.center_fitness:
            weights = (weights - jnp.mean(weights)) / (1E-6 + jnp.std(weights))

        weights1, weights2 = jnp.split(weights,
                                       2)
        weights = weights1 - weights2

        delta = jax.tree_util.tree_map(
            functools.partial(self.compute_delta,
                              weights=weights),
            one_param,
            noise)

        return delta

    def compute_delta(
        self,
        params: jnp.ndarray,
        noise: jnp.ndarray,
        weights: jnp.ndarray,
    ) -> jnp.ndarray:
        """
            Compute the delta, i.e.
            the update to be passed to the optimizer.
            Args:
              params: Policy parameter leaf.
              noise: Noise leaf, with dimensions (population_size,) + params.shape
              weights: Fitness weights, vector of length population_size.
            Returns:
            """
        # NOTE: The trick "len(weights) -> len(weights) * perturbation_std" is
        # equivalent to tuning the l2_coef.
        weights = jnp.reshape(weights,
                              ([self.population_size] + [1] * (jnp.ndim(noise) - 1)))
        delta = jnp.sum(noise * weights,
                        axis=0) / jnp.asarray(self.population_size)
        # l2coeff controls the weight decay of the parameters of our policy network.
        # This prevents the parameters from growing very large compared to the
        # perturbations.
        delta = delta - params * self.l2coeff
        # Return -delta because the optimizer is set up to go against the gradient.
        return -delta

    @functools.partial(jax.jit,
                       static_argnames=("self", "use_gcrl_scoring_fn",)
                       )
    def improve_genotype_one_step(self,
                                  gen: Genotype,
                                  expected_bd: Descriptor,
                                  optimizer_state: Params,
                                  random_key: RNGKey,
                                  use_gcrl_scoring_fn=True,
                                  ):
        self.logger.debug(f"Jitting improve_genotype_one_step with use_gcrl_scoring_fn={use_gcrl_scoring_fn}")
        gradient_genotype = self.calculate_gradient(gen, expected_bd, random_key, use_gcrl_scoring_fn)

        params_update, optimizer_state = self.optimizer.update(
            gradient_genotype,
            optimizer_state)
        new_gen = optax.apply_updates(gen,
                                      params_update)

        return new_gen, optimizer_state

    def run(self,
            initial_gen: Genotype,
            expected_bd: Descriptor,
            random_key: RNGKey,
            number_iterations: int,
            use_gcrl_scoring_fn=True,
            optimizer_state=None,
            ):
        gen = initial_gen
        if optimizer_state is None:
            optimizer_state = self.optimizer.init(gen)

        def _step_fn(_carry, _):
            gen, optimizer_state, random_key = _carry
            key_iteration, random_key = jax.random.split(random_key)
            new_gen, new_optimizer_state = self.improve_genotype_one_step(gen, expected_bd, optimizer_state,
                                                                          key_iteration, use_gcrl_scoring_fn)
            return (new_gen, new_optimizer_state, random_key), None

        random_key, key_run_optim = jax.random.split(random_key)
        carry = (gen, optimizer_state, key_run_optim)
        for _iter in range(number_iterations):
            carry, _ = _step_fn(carry, None)
        optimised_gen, last_optimizer_state, _ = carry

        return optimised_gen, last_optimizer_state


class ReproducibilityImproverQDR(ReproducibilityImprover):
    def __init__(self,
                 perturbation_std: float,
                 population_size: int,
                 scoring_fn: Callable,
                 radius_acceptance_bd: float,
                 fitness_shaping: FitnessShaping,
                 learning_rate: float,
                 center_fitness: bool
                 ):
        self.radius_acceptance_bd = radius_acceptance_bd

        super().__init__(perturbation_std,
                         population_size,
                         scoring_fn,
                         fitness_shaping,
                         learning_rate,
                         center_fitness)

    def get_gcrl_scoring_fn(self, scoring_fn):
        def gcrl_scoring_fn(gen, goal_desc, random_key):
            fit, desc, infos, random_key = scoring_fn(gen, random_key)

            dist_l_inf = jnp.linalg.norm(desc - goal_desc, ord=jnp.inf, axis=1)
            use_dist_in_score = jnp.where(dist_l_inf < self.radius_acceptance_bd, 0., 1.)
            use_fitness_in_score = 1. - use_dist_in_score

            offset_fit = 1e6

            dist_l_2 = jnp.linalg.norm(desc - goal_desc, ord=2, axis=1)
            dist_score = -1. * dist_l_2
            gcrl_fit = dist_score * use_dist_in_score + (fit + offset_fit) * use_fitness_in_score

            return gcrl_fit, desc, infos, random_key

        return gcrl_scoring_fn
