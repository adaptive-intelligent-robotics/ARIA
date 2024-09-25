import logging

import numpy as np


class CounterEvals:
    """
    Can only be used with reevaluation-based algorithms.
    Is there to keep track of the number of evaluations done, and
    ensure that the total number of evaluations is not exceeded.
    """

    def __init__(self,
                 total_number_evals,
                 sampling_size,
                 ):
        self.total_number_evals = total_number_evals
        # self.evals_per_scoring_call_per_genotype = evals_per_scoring_call_per_genotype
        # self.batch_size = batch_size
        self.sampling_size = sampling_size

        self.counter_evals = 0
        self.counter_increments = 0

        # Set up logging
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

    def should_stop(self):
        return self.counter_evals >= self.total_number_evals

    def increment_standard_sampling_size(self):
        self.counter_evals += self.sampling_size
        self.counter_increments += 1

    def increment_custom_sampling_size(self, custom_sampling_size):
        self.counter_evals += custom_sampling_size
        # self.counter_increments += 1

    def print_info(self):
        percentage = 100. * self.counter_evals / self.total_number_evals
        self.logger.info(f"CounterEvals: {self.counter_evals}/{self.total_number_evals} "
                        f"({percentage:.2f}%)")

    @classmethod
    def _get_total_number_evals(cls,
                                grid_shape,
                                number_robust_iterations_initial_aria,
                                num_reevals_estimate_mean_initial_aria,
                                number_robust_iterations_loop,
                                evals_per_optimization_step,
                                ):
        number_centroids = np.prod(grid_shape)
        total_iterations = number_robust_iterations_initial_aria
        total_iterations += num_reevals_estimate_mean_initial_aria
        total_iterations += number_centroids * number_robust_iterations_loop

        return total_iterations * evals_per_optimization_step

    @classmethod
    def create_from_config(cls,
                           config,
                           ):
        config_task = config.task
        config_task_aria = config.task.aria

        total_number_evals = cls._get_total_number_evals(
            grid_shape=config_task.grid_shape,
            number_robust_iterations_initial_aria=config_task_aria.number_robust_iterations_initial,
            num_reevals_estimate_mean_initial_aria=config_task_aria.num_reevals_estimate_mean_initial,
            number_robust_iterations_loop=config_task_aria.number_robust_iterations_loop,
            # Each individual is considered with its opposite
            # (w.r.t. to the mean of the normal distribution)
            evals_per_optimization_step=config_task_aria.batch_size * 2,
        )

        evals_per_scoring_call_per_gen = config_task.reeval.evals_per_gen

        assert config_task.budget_per_eval % evals_per_scoring_call_per_gen == 0
        # batch_size_different_gens = config_task.budget_per_eval // evals_per_scoring_call_per_gen

        return cls(
            total_number_evals=total_number_evals,
            sampling_size=config_task.budget_per_eval,
        )
