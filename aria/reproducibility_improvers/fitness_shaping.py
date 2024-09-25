from __future__ import annotations

import enum
import functools

from jax import numpy as jnp


def centered_rank(x: jnp.ndarray) -> jnp.ndarray:
    x = jnp.argsort(jnp.argsort(x))
    x /= (len(x) - 1)
    return x - .5


def wierstra(x: jnp.ndarray) -> jnp.ndarray:
    x = len(x) - jnp.argsort(jnp.argsort(x))
    x = jnp.maximum(0,
                    jnp.log(len(x) / 2.0 + 1) - jnp.log(x))
    return x / jnp.sum(x) - 1.0 / len(x)


class FitnessShaping(enum.Enum):
    ORIGINAL = functools.partial(lambda x: x)
    CENTERED_RANK = functools.partial(centered_rank)
    WIERSTRA = functools.partial(wierstra)
