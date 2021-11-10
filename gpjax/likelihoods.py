from typing import Callable, Optional

import jax.numpy as jnp
from dataclasses import dataclass
from tensorflow_probability.substrates.jax import distributions as tfd

from treeo import Tree, field
from .utils import identity
from .parameters import PositiveParameter, ShiftedPositive


@dataclass(repr=False)
class Likelihood(Tree):
    # link_function: Callable[[jnp.DeviceArray], jnp.DeviceArray] = identity
    name: Optional[str] = "Likelihood"

    def __repr__(self):
        return f"{self.name} likelihood function"


@dataclass(repr=False)
class Gaussian(Likelihood):
    # link_function: Callable[[jnp.DeviceArray], jnp.DeviceArray] = identity
    name: Optional[str] = "Gaussian"
    obs_noise: jnp.array = field(
        default=jnp.array([1.0]), node=True, kind=ShiftedPositive
    )


@dataclass(repr=False)
class Bernoulli(Likelihood):
    # link_function = tfd.ProbitBernoulli
    name: Optional[str] = "Bernoulli"
    num_datapoints: int = 1


def predictive_moments(likelihood: Likelihood) -> Callable:
    link = likelihood.link_function

    def moments(mean: jnp.DeviceArray, variance: jnp.DeviceArray) -> tfd.Distribution:
        rv = link(mean / jnp.sqrt(1 + variance))
        return rv

    return moments


NonConjugateLikelihoods = Bernoulli
NonConjugateLikelihoodType = Bernoulli  # Union[Bernoulli]
