from __future__ import annotations

from typing import Callable, Tuple
from typing_extensions import TypeAlias

from jax import numpy as jnp
from qdax.custom_types import Genotype, RNGKey, Fitness, Descriptor, ExtraScores

Distance: TypeAlias = jnp.ndarray
ScoringFnType: TypeAlias = Callable[[Genotype, RNGKey], Tuple[Fitness, Descriptor, ExtraScores, RNGKey]]
