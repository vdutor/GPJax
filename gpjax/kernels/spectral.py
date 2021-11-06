from typing import Optional, Dict

import jax.numpy as jnp
from chex import dataclass
from multipledispatch import dispatch
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax.kernels import RBF

from ..types import Array
from ..utils import I, sort_dictionary
from .base import Kernel
from .utils import scale, stretch
from ..config import get_defaults


@dataclass(repr=False)
class SpectralKernel:
    num_basis: int
    spectral_density: tfd.Distribution = None


@dataclass(repr=False)
class SpectralRBF(Kernel, SpectralKernel):
    name: Optional[str] = "Spectral RBF"
    stationary: str = True

    def __post_init__(self):
        config = get_defaults()
        # Define kernel's spectral density and sample frequencies
        self.spectral_density = tfd.Normal(loc=jnp.array(0.0), scale=jnp.array(1.0))
        freqs = self.spectral_density.sample(
            sample_shape=(self.num_basis, self.ndims), seed=config.seed
        )
        # Store parameters
        self.parameters: Dict[str, Array] = {
            "basis_fns": freqs,
            "lengthscale": jnp.array([1.0] * self.ndims),
            "variance": jnp.array([1.0]),
        }

    def __repr__(self):
        return f"{self.name}:\n\t Number of basis functions: {self.num_basis}\n\t Stationary: {self.stationary} \n\t ARD structure: {self.ard}"

    def __call__(self, x: jnp.DeviceArray, y: jnp.DeviceArray, params: dict) -> Array:
        phi = self._build_phi(x, params)
        return jnp.matmul(phi, jnp.transpose(phi)) / self.num_basis

    def _build_phi(self, x: jnp.DeviceArray, params):
        scaled_freqs = scale(params["basis_fns"], params["lengthscale"])
        phi = jnp.matmul(x, jnp.transpose(scaled_freqs))
        return jnp.hstack([jnp.cos(phi), jnp.sin(phi)])


@dispatch(RBF, int)
def to_spectral(kernel: RBF, num_basis: int):
    return SpectralRBF(num_basis=num_basis)
