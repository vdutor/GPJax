import warnings

import jax.numpy as jnp
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from ..types import Array, NoneType


@dispatch(Array, tfd.Distribution)
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


def prior_checks(gp, priors: dict) -> dict:
    if "latent" in priors.keys():
        latent_prior = priors["latent"]
        if latent_prior.name != "Normal":
            warnings.warn(
                f"A {latent_prior.name} distribution prior has been placed on the latent function. It is strongly afvised that a unit-Gaussian prior is used."
            )
        return priors
    else:
        priors["latent"] = tfd.Normal(loc=0.0, scale=1.0)
        return priors
