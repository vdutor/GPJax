from typing import Callable, Dict, Optional, Union

import jax.numpy as jnp
from chex import dataclass
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from .types import Array
from .utils import identity


@dataclass(repr=False)
class Likelihood:
    link_function: Callable[[jnp.DeviceArray], jnp.DeviceArray] = identity
    name: Optional[str] = "Likelihood"
    parameters = {}

    def __repr__(self):
        return f"{self.name} likelihood function"


@dataclass(repr=False)
class Gaussian(Likelihood):
    name: Optional[str] = "Gaussian"
    parameters = {"obs_noise": jnp.array([1.0])}


@dataclass(repr=False)
class Bernoulli(Likelihood):
    link_function = tfd.ProbitBernoulli
    name: Optional[str] = "Bernoulli"


@dispatch(Bernoulli)
def predictive_moments(likelihood: Bernoulli) -> Callable:
    link = likelihood.link_function

    def moments(mean: jnp.DeviceArray, variance: jnp.DeviceArray) -> tfd.Distribution:
        rv = link(mean / jnp.sqrt(1 + variance))
        return rv

    return moments


@dataclass(repr=False)
class Poisson(Likelihood):
    link_function = jnp.exp
    name: Optional[str] = "Poisson"


NonConjugateLikelihoods = (Bernoulli, Poisson)
NonConjugateLikelihoodType = Union[Bernoulli, Poisson]
