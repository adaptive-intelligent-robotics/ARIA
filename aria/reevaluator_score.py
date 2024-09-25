import jax
import jax.numpy as jnp

from qdax.custom_types import Genotype, RNGKey


class ReEvaluator:
    """
    Class to reevaluate a genotype several times
    """
    def __init__(self, scoring_fn, num_reevals):
        self._scoring_fn = scoring_fn
        self._reeval_scoring_fn = jax.jit(jax.vmap(self._scoring_fn, in_axes=(None, 0), out_axes=1))
        self._num_reevals = num_reevals
        
    @staticmethod
    def add_dimension_to_pytree(pytree, dim=0):
        return jax.tree_map(lambda x: jnp.expand_dims(x, dim), pytree)

    def reeval(self, 
               genotype_single: Genotype, 
               random_key: RNGKey, 
               add_dimension: bool = False):
        """
        Returns the reevaluations of one genotype
        """

        if add_dimension:
            genotype_single = self.add_dimension_to_pytree(genotype_single)
        
        subkeys = jax.random.split(random_key, num=self._num_reevals)

        fit, desc, _, _ = self._reeval_scoring_fn(genotype_single, jnp.asarray(subkeys))
        return fit, desc

    def mean_reevals(self, genotype_single: Genotype, random_key: RNGKey, add_dimension: bool = False):
        """
        Returns the mean of the reevaluations
        """
        random_key, subkey = jax.random.split(random_key)
        fit_reevals, desc_reevals = self.reeval(genotype_single, random_key=subkey, add_dimension=add_dimension)
        return jnp.mean(fit_reevals, axis=1), jnp.mean(desc_reevals, axis=1)

