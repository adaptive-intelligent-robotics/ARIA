from __future__ import annotations

import dataclasses
from typing import Tuple, Callable, Optional

import brax.envs
import jax
import qdax.environments
from jax import numpy as jnp
from qdax.core.neuroevolution.networks.networks import MLP

from qdax.tasks.brax_envs import create_default_brax_task_components
from qdax.custom_types import RNGKey, Genotype

from aria.tasks.arm import make_noisy_scoring_function, arm_unbounded_scoring_function
from aria.utils.types import ScoringFnType


def get_infos_brax_env(env_name,
                       episode_length: int,
                       is_reset_based: bool,
                       random_key):
    """
    Returns the environment, the policy network and the scoring function for a brax environment
    """
    env, policy_network, scoring_fn, _ = create_default_brax_task_components(env_name=env_name,
                                                                             random_key=random_key,
                                                                             deterministic=not is_reset_based,
                                                                             episode_length=episode_length,
                                                                             )
    return env, policy_network, scoring_fn


@dataclasses.dataclass
class TaskInfo:
    scoring_fn: ScoringFnType
    initial_gen: Genotype
    env: Optional[brax.envs.Env]
    policy_network: Optional[MLP]


class FactoryQDTask:
    """
    Factory class to get all the components of a task, such as the scoring function,
    the initial genotype(s), the environment, the policy network, etc.
    """

    @classmethod
    def _get_scoring_fn_initial_gen_arm(cls,
                                        std_params,
                                        std_fitness,
                                        std_descriptor,
                                        num_params,
                                        random_key,
                                        batch_size_init: int = None,
                                        ):
        scoring_fn_arm = make_noisy_scoring_function(
            scoring_fn=arm_unbounded_scoring_function,
            std_params=std_params,
            std_fitness=std_fitness,
            std_descriptor=std_descriptor,
        )
        random_key, subkey = jax.random.split(random_key)
        if batch_size_init is None:
            initial_gen_arm = jax.random.uniform(subkey,
                                                 shape=(num_params,))
        else:
            initial_gen_arm = jax.random.uniform(subkey,
                                                 shape=(batch_size_init, num_params))
        return scoring_fn_arm, initial_gen_arm

    @classmethod
    def _get_task_info_brax(cls,
                            env_name: str,
                            episode_length: int,
                            is_reset_based: bool,
                            random_key: RNGKey,
                            batch_size_init: int = None,
                            ) -> TaskInfo:
        random_key, subkey = jax.random.split(random_key)

        env, policy_network, scoring_fn = get_infos_brax_env(env_name,
                                                             episode_length,
                                                             is_reset_based,
                                                             subkey)

        random_key, subkey_init = jax.random.split(random_key)
        if batch_size_init is None:
            fake_obs = jnp.zeros(shape=(env.observation_size,))
            init_genotype = policy_network.init(subkey_init,
                                                fake_obs)
        else:
            fake_obs = jnp.zeros(shape=(batch_size_init, env.observation_size))
            array_subkeys_init = jax.random.split(subkey_init, num=batch_size_init)
            init_genotype = jax.vmap(policy_network.init)(array_subkeys_init,
                                                          fake_obs)

        return TaskInfo(
            scoring_fn=scoring_fn,
            initial_gen=init_genotype,
            env=env,
            policy_network=policy_network,
        )

    @classmethod
    def get_scoring_fn_initial_gen(cls,
                                   config_task,
                                   random_key: RNGKey,
                                   batch_size_init: int = None,
                                   ) -> TaskInfo:
        name = config_task.env_name
        if name == "arm":
            std_params = 0.
            std_fitness = config_task.std_fitness
            std_descriptor = config_task.std_descriptor
            num_params = config_task.size_arm

            random_key, subkey = jax.random.split(random_key)
            scoring_fn, initial_gen_arm = cls._get_scoring_fn_initial_gen_arm(
                std_params,
                std_fitness,
                std_descriptor,
                num_params,
                subkey,
                batch_size_init,
            )
            return TaskInfo(
                scoring_fn=scoring_fn,
                initial_gen=initial_gen_arm,
                env=None,
                policy_network=None,
            )

        elif name in qdax.environments._qdax_custom_envs: # brax envs
            random_key, subkey = jax.random.split(random_key)
            task_info = cls._get_task_info_brax(env_name=config_task.env_name,
                                                episode_length=config_task.episode_length,
                                                is_reset_based=config_task.is_reset_based,
                                                random_key=subkey,
                                                batch_size_init=batch_size_init,
                                                )
            return task_info
        else:
            raise ValueError(f"Unknown env name: {name}")



