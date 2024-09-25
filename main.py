from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import hydra
import jax
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from aria.algos.algo_factory import FactoryAlgo
from aria.tasks.task_factory import FactoryQDTask


def register_new_resolvers():
    OmegaConf.register_new_resolver('cond_weight_fit',
                                    lambda var: f'_weight-fit-{var}' if var is not None else '')
    OmegaConf.register_new_resolver('cond_proba_fit',
                                    lambda var: f'_use-proba-cell-{var}' if var is not None else '')


def get_init_batch_size(full_config):
    use_single_init_genotype = full_config.algo.use_single_init_genotype

    config_algo = full_config.algo
    name_algo = config_algo.algo_name

    if use_single_init_genotype:
        return None

    if name_algo == "aria_pga_init":
        batch_size_init = config_algo.env_batch_size
        return batch_size_init

    batch_size_init = full_config.task.budget_per_eval
    return batch_size_init


@hydra.main(config_path="configs/",
            config_name="config")
def robustify(full_config):
    seed = full_config.task.seed

    random_key = jax.random.PRNGKey(seed)

    random_key, subkey_scoring = jax.random.split(random_key)

    batch_size_init = get_init_batch_size(full_config)

    task_info = FactoryQDTask.get_scoring_fn_initial_gen(config_task=full_config.task,
                                                         random_key=subkey_scoring,
                                                         batch_size_init=batch_size_init,
                                                         )

    factory_algo = FactoryAlgo(config=full_config,
                               task_info=task_info,
                               )

    algo = factory_algo.create()
    random_key, subkey_algo = jax.random.split(random_key)

    with jax.disable_jit() if full_config.debug_mode else nullcontext():
        algo.run(
            initial_genotypes=task_info.initial_gen,
            random_key=subkey_algo,
        )


if __name__ == "__main__":
    register_new_resolvers()
    cs = ConfigStore.instance()
    robustify()
