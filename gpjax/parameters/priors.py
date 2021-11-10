import warnings

import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from ..types import Array, NoneType


def log_density(param: jnp.DeviceArray, density: tfd.Distribution) -> Array:
    return density.log_prob(param)


def evaluate_prior(params: dict, priors: dict) -> Array:
    return jnp.array(0.0)


def evaluate_prior(params: dict, priors: dict) -> Array:
    if isinstance(priors, NoneType):
        return jnp.array(0.0)
    else:
        lpd = jnp.array(0)
        for param, val in priors.items():
            lpd += jnp.sum(log_density(params[param], priors[param]))
        return lpd
