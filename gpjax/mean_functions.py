from typing import Optional

import jax.numpy as jnp
from dataclasses import dataclass
from multipledispatch import dispatch
from treeo import Tree, field
from .parameters import Parameter

from .types import Array


@dataclass(repr=False)
class MeanFunction(Tree):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Mean function"

    def __call__(self, x: jnp.array) -> jnp.array:
        raise NotImplementedError

    def __repr__(self):
        return f"{self.name}\n\t Output dimension: {self.output_dim}"


@dataclass(repr=False)
class Zero(MeanFunction):
    output_dim: Optional[int] = 1
    name: Optional[str] = "Zero mean function"

    def __call__(self, x: jnp.array) -> jnp.array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.zeros(shape=out_shape)


@dataclass(repr=False)
class Constant(MeanFunction):
    offset: jnp.array = field(default=jnp.array([1.0]), node=True, kind=Parameter)
    output_dim: Optional[int] = 1
    name: Optional[str] = "Constant mean function"

    def __call__(self, x: jnp.array) -> jnp.array:
        out_shape = (x.shape[0], self.output_dim)
        return jnp.ones(shape=out_shape) * self.offset
