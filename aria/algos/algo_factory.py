import functools

import jax.numpy as jnp
from qdax.core.containers.mapelites_repertoire import compute_euclidean_centroids
from qdax.core.emitters.mutation_operators import isoline_variation
from qdax.core.emitters.pga_me_emitter import PGAMEEmitter, PGAMEConfig
from qdax.core.emitters.standard_emitters import MixingEmitter
from qdax.utils.metrics import default_qd_metrics, default_moqd_metrics

from aria.algos.abstract_algo import AbstractAlgo
from aria.algos.map_elites_algo import MAPElitesAlgo
from aria.algos.mome_reproducibility import MOMEReproducibilityAlgo
from aria.algos.naive_sampling import NaiveSampling
from aria.aria_es_init import ARIA_ES_Init
from aria.aria_mapelites_init import ARIAMapElitesInit
from aria.reevaluator_score import ReEvaluator
from aria.reproducibility_improvers.fitness_shaping import FitnessShaping
from aria.reproducibility_improvers.improver_linear_comb import \
    ReproducibilityImproverQDRLinearCombination
from aria.reproducibility_improvers.improver_standard import ReproducibilityImproverQDR
from aria.tasks.task_factory import TaskInfo
from aria.utils.distances_utils import automatic_cell_inf_norm_radius_calculator
from aria.utils.normaliser import Normaliser, NormaliserMinMax


class FactoryAlgo:
    def __init__(self, config, task_info: TaskInfo):
        self.config = config
        self.config_algo = config.algo
        self.config_task = config.task

        self.algo_name = self.config_algo.algo_name
        self.scoring_fn = task_info.scoring_fn

        self.env = task_info.env
        self.policy_network = task_info.policy_network

    def _automatic_radius_calculator(self) -> float:
        radius_acceptance_bd = automatic_cell_inf_norm_radius_calculator(
            grid_shape=self.config_task.grid_shape,
            min_bd=self.config_task.min_bd,
            max_bd=self.config_task.max_bd,
        )
        return radius_acceptance_bd

    def _get_robustness_improver_aria(self):
        config_task_aria = self.config_task.aria

        learning_rate = config_task_aria.learning_rate

        batch_size = config_task_aria.batch_size

        radius_acceptance_bd = self._automatic_radius_calculator()

        robustness_improver = ReproducibilityImproverQDR(
            perturbation_std=config_task_aria.perturbation_std,
            population_size=batch_size,
            scoring_fn=self.scoring_fn,
            fitness_shaping=FitnessShaping.CENTERED_RANK,
            radius_acceptance_bd=radius_acceptance_bd,
            learning_rate=learning_rate,
            center_fitness=False,
        )

        return robustness_improver

    def _get_fitness_normaliser(self) -> Normaliser:
        config_task_normaliser = self.config_task.normaliser

        return NormaliserMinMax(
            min_val=config_task_normaliser.min_fitness,
            max_val=config_task_normaliser.max_fitness,
        )

    def _get_distance_normaliser(self) -> Normaliser:
        config_task_normaliser = self.config_task.normaliser

        return NormaliserMinMax(
            min_val=config_task_normaliser.min_distance,
            max_val=config_task_normaliser.max_distance,
        )

    def _get_robustness_improver_aria_linearcomb(self):

        config_task_aria = self.config_task.aria

        learning_rate = config_task_aria.learning_rate
        batch_size = config_task_aria.batch_size
        weight_fitness_obj = self.config_algo.weight_fitness_obj

        robustness_improver = ReproducibilityImproverQDRLinearCombination(
            perturbation_std=config_task_aria.perturbation_std,
            population_size=batch_size,
            scoring_fn=self.scoring_fn,
            fitness_shaping=FitnessShaping.CENTERED_RANK,
            learning_rate=learning_rate,
            center_fitness=False,
            fitness_normaliser=self._get_fitness_normaliser(),
            distance_normaliser=self._get_distance_normaliser(),
            weight_fitness_obj=weight_fitness_obj,
        )

        return robustness_improver

    def get_centroids(self):
        grid_shape = tuple(self.config_task.grid_shape)
        min_bd = self.config_task.min_bd
        max_bd = self.config_task.max_bd

        centroids = compute_euclidean_centroids(
            grid_shape=grid_shape,
            minval=min_bd,
            maxval=max_bd,
        )

        return centroids

    def _get_reevaluator_aria(self) -> ReEvaluator:
        config_task_aria = self.config_task.aria

        reevaluator = ReEvaluator(scoring_fn=self.scoring_fn,
                                  num_reevals=config_task_aria.num_reevals_estimate_mean_initial)

        return reevaluator

    def _get_aria_es_init_algo(self) -> ARIA_ES_Init:
        config_task_aria = self.config_task.aria

        reevaluator = self._get_reevaluator_aria()

        centroids = self.get_centroids()

        robustness_improver = self._get_robustness_improver_aria()

        aria_es_init = ARIA_ES_Init(robustness_improver,
                                    reevaluator=reevaluator,
                                    centroids=centroids,
                                    number_robust_iterations_initial=config_task_aria.number_robust_iterations_initial,
                                    number_robust_iterations_loop=config_task_aria.number_robust_iterations_loop,
                                    number_parallel_optimisations=config_task_aria.number_parallel_optimisations,
                                    config=self.config,
                                    scoring_fn=self.scoring_fn,
                                    )

        return aria_es_init

    def _get_aria_mapelites_init_algo(self) -> ARIAMapElitesInit:
        config_task_aria = self.config_task.aria

        reevaluator = self._get_reevaluator_aria()

        centroids = self.get_centroids()

        robustness_improver = self._get_robustness_improver_aria()

        total_map_elites_evaluations = self.config_algo.total_map_elites_evaluations

        aria_mapelites_init_algo = ARIAMapElitesInit(robustness_improver,
                                                     reevaluator=reevaluator,
                                                     centroids=centroids,
                                                     number_robust_iterations_initial=config_task_aria.number_robust_iterations_initial,
                                                     number_robust_iterations_loop=config_task_aria.number_robust_iterations_loop,
                                                     number_parallel_optimisations=config_task_aria.number_parallel_optimisations,
                                                     config=self.config,
                                                     scoring_fn=self.scoring_fn,
                                                     total_map_elites_evaluations=total_map_elites_evaluations,
                                                     emitter=self._get_uniform_emitter(consider_reevals=False),
                                                     metrics_fn_map_elites=self._get_default_qd_metrics(),
                                                     )

        return aria_mapelites_init_algo

    def _get_aria_pga_init_algo(self) -> ARIAMapElitesInit:
        config_task_aria = self.config_task.aria

        reevaluator = self._get_reevaluator_aria()

        centroids = self.get_centroids()

        robustness_improver = self._get_robustness_improver_aria()

        total_map_elites_evaluations = self.config_algo.total_map_elites_evaluations

        pga_emitter = self._get_pga_emitter(consider_reevals=False,
                                            override_env_batch_size=self.config_algo.env_batch_size)

        aria_mapelites_init_algo = ARIAMapElitesInit(robustness_improver,
                                                     reevaluator=reevaluator,
                                                     centroids=centroids,
                                                     number_robust_iterations_initial=config_task_aria.number_robust_iterations_initial,
                                                     number_robust_iterations_loop=config_task_aria.number_robust_iterations_loop,
                                                     number_parallel_optimisations=config_task_aria.number_parallel_optimisations,
                                                     config=self.config,
                                                     scoring_fn=self.scoring_fn,
                                                     total_map_elites_evaluations=total_map_elites_evaluations,
                                                     emitter=pga_emitter,
                                                     metrics_fn_map_elites=self._get_default_qd_metrics(),
                                                     )

        return aria_mapelites_init_algo

    def _get_aria_linearcomb(self) -> ARIA_ES_Init:
        config_task_aria = self.config_task.aria

        reevaluator = self._get_reevaluator_aria()

        centroids = self.get_centroids()

        robustness_improver = self._get_robustness_improver_aria_linearcomb()

        total_map_elites_evaluations = self.config_algo.total_map_elites_evaluations

        aria_map_elites_init = ARIAMapElitesInit(robustness_improver,
                                                 reevaluator=reevaluator,
                                                 centroids=centroids,
                                                 number_robust_iterations_initial=config_task_aria.number_robust_iterations_initial,
                                                 number_robust_iterations_loop=config_task_aria.number_robust_iterations_loop,
                                                 number_parallel_optimisations=config_task_aria.number_parallel_optimisations,
                                                 config=self.config,
                                                 scoring_fn=self.scoring_fn,
                                                 total_map_elites_evaluations=total_map_elites_evaluations,
                                                 emitter=self._get_uniform_emitter(consider_reevals=False),
                                                 metrics_fn_map_elites=self._get_default_qd_metrics(),
                                                 )

        return aria_map_elites_init

    def _get_variation_fn(self):
        config_variation_fn = self.config_task.variation_fn

        iso_sigma = config_variation_fn.iso_sigma
        line_sigma = config_variation_fn.line_sigma

        # Define emitter
        variation_fn = functools.partial(
            isoline_variation,
            iso_sigma=iso_sigma,
            line_sigma=line_sigma
        )

        return variation_fn

    def _get_uniform_emitter(self, consider_reevals=False) -> MixingEmitter:
        if not consider_reevals:
            batch_size = self.config_task.budget_per_eval
        else:
            assert self.config_task.budget_per_eval % self.config_task.reeval.evals_per_gen == 0
            batch_size = self.config_task.budget_per_eval // self.config_task.reeval.evals_per_gen

        variation_fn = self._get_variation_fn()

        mixing_emitter = MixingEmitter(
            mutation_fn=None,
            variation_fn=variation_fn,
            variation_percentage=1.0,
            batch_size=batch_size,
        )
        return mixing_emitter

    def _get_default_qd_metrics(self):
        qd_offset = 1e6  # Analysis is done in preprocessing anyway...

        metrics_fn = functools.partial(
            default_qd_metrics,
            qd_offset=qd_offset,
        )

        return metrics_fn

    def _get_naive_sampling(self) -> NaiveSampling:
        metrics_fn = self._get_default_qd_metrics()

        emitter = self._get_uniform_emitter(consider_reevals=True)

        naive_sampling_algo = NaiveSampling(
            config=self.config,
            scoring_fn=self.scoring_fn,
            centroids=self.get_centroids(),
            fitness_normaliser=self._get_fitness_normaliser(),
            dist_normaliser=self._get_distance_normaliser(),
            emitter=emitter,
            metrics_fn=metrics_fn,
        )

        return naive_sampling_algo

    def _get_mome_reproducibility(self) -> MOMEReproducibilityAlgo:
        reference_point = jnp.array([1e6, 1e6])  # Analysis is done in preprocessing anyway...

        # how to compute metrics from a repertoire
        metrics_fn = functools.partial(
            default_moqd_metrics,
            reference_point=reference_point
        )

        uniform_emitter = self._get_uniform_emitter(consider_reevals=True)

        mome_reproducibility_algo = MOMEReproducibilityAlgo(
            config=self.config,
            scoring_fn=self.scoring_fn,
            centroids=self.get_centroids(),
            emitter=uniform_emitter,
            metrics_fn=metrics_fn,
        )

        return mome_reproducibility_algo

    def _get_map_elites(self) -> MAPElitesAlgo:
        metrics_fn = self._get_default_qd_metrics()

        map_elites_algo = MAPElitesAlgo(
            config=self.config,
            scoring_fn=self.scoring_fn,
            centroids=self.get_centroids(),
            emitter=self._get_uniform_emitter(),
            metrics_fn=metrics_fn,
        )

        return map_elites_algo

    def _get_pga_emitter(self, override_env_batch_size: int = None, consider_reevals=False) -> PGAMEEmitter:
        if consider_reevals:
            raise NotImplementedError("Reevals not implemented for PGA")

        variation_fn = self._get_variation_fn()

        if override_env_batch_size is None:
            env_batch_size = self.config_task.budget_per_eval
        else:
            env_batch_size = override_env_batch_size

        pga_emitter_config = PGAMEConfig(
            env_batch_size=env_batch_size,
            batch_size=self.config_algo.transitions_batch_size,
            proportion_mutation_ga=self.config_algo.proportion_mutation_ga,
            critic_hidden_layer_size=self.config_algo.critic_hidden_layer_size,
            critic_learning_rate=self.config_algo.critic_learning_rate,
            greedy_learning_rate=self.config_algo.greedy_learning_rate,
            policy_learning_rate=self.config_algo.policy_learning_rate,
            noise_clip=self.config_algo.noise_clip,
            policy_noise=self.config_algo.policy_noise,
            discount=self.config_algo.discount,
            reward_scaling=self.config_algo.reward_scaling,
            replay_buffer_size=self.config_algo.replay_buffer_size,
            soft_tau_update=self.config_algo.soft_tau_update,
            num_critic_training_steps=self.config_algo.num_critic_training_steps,
            num_pg_training_steps=self.config_algo.num_pg_training_steps,
            policy_delay=self.config_algo.policy_delay,
        )

        assert self.env is not None
        assert self.policy_network is not None

        pga_me_emitter = PGAMEEmitter(
            config=pga_emitter_config,
            policy_network=self.policy_network,
            env=self.env,
            variation_fn=variation_fn,
        )

        return pga_me_emitter

    def _get_pga_map_elites(self) -> MAPElitesAlgo:
        metrics_fn = self._get_default_qd_metrics()

        emitter = self._get_pga_emitter()

        map_elites_algo = MAPElitesAlgo(
            config=self.config,
            scoring_fn=self.scoring_fn,
            centroids=self.get_centroids(),
            emitter=emitter,
            metrics_fn=metrics_fn,
        )

        return map_elites_algo

    def create(self) -> AbstractAlgo:
        if self.algo_name == "aria_es_init":
            aria_algo = self._get_aria_es_init_algo()
            return aria_algo
        elif self.algo_name == "aria_mapelites_init":
            aria_mapelites_init_algo = self._get_aria_mapelites_init_algo()
            return aria_mapelites_init_algo
        elif self.algo_name == "aria_pga_init":
            aria_pga_init_algo = self._get_aria_pga_init_algo()
            return aria_pga_init_algo
        elif self.algo_name == "aria_linearcomb":
            aria_linearcomb_algo = self._get_aria_linearcomb()
            return aria_linearcomb_algo
        elif self.algo_name == "naive_sampling":
            naive_sampling_algo = self._get_naive_sampling()
            return naive_sampling_algo
        elif self.algo_name == "mome_reproducibility":
            mome_reproducibility_algo = self._get_mome_reproducibility()
            return mome_reproducibility_algo
        elif self.algo_name == "map_elites":
            map_elites_algo = self._get_map_elites()
            return map_elites_algo
        elif self.algo_name == "pga_me":
            pga_map_elites_algo = self._get_pga_map_elites()
            return pga_map_elites_algo
        else:
            raise ValueError(f"Unknown algo_name: {self.algo_name}")
