from typing import Optional

import jax.numpy as jnp
from dataclasses import dataclass
from jax import vmap
from treeo import Tree, field

from gpjax.types import Array
from ..parameters import PositiveParameter


@dataclass(repr=False)
class Kernel(Tree):
    ndims: Optional[int] = 1
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "Kernel"

    def __call__(self, x: Array, y: Array, params: dict) -> Array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}:\n\t Stationary: {self.stationary}\n\t Spectral form: {self.spectral} \n\t ARD structure: {self.ard}"

    @property
    def ard(self):
        return True if self.ndims > 1 else False


class RBF(Kernel):
    lengthscale: Optional[Array] = field(
        default=jnp.array([1.0]), node=True, kind=PositiveParameter
    )
    variance: Optional[Array] = field(
        default=jnp.array([1.0]), node=True, kind=PositiveParameter
    )
    ndims: Optional[int] = 1
    stationary: Optional[bool] = False
    spectral: Optional[bool] = False
    name: Optional[str] = "RBF"

    def __call__(self, x: Array, y: Array):
        ell = self.lengthscale
        sigma = self.variance
        x = x / ell
        y = y / ell
        sq_distance = distance(x, y, power=2)
        val = sigma * jnp.exp(-0.5 * sq_distance)
        return val.squeeze()


def distance(x: jnp.array, y: jnp.array, power: int = 1):
    return jnp.sum((x - y) ** power)


def gram(kernel: Kernel, inputs: Array) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1))(inputs))(inputs)


def cross_covariance(kernel: Kernel, x: Array, y: Array) -> Array:
    return vmap(lambda x1: vmap(lambda y1: kernel(x1, y1))(x))(y)
