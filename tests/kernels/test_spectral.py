import jax.numpy as jnp
import jax.random as jr
import pytest
from tensorflow_probability.substrates.jax import distributions as tfd

from gpjax.kernels import RBF
from gpjax.core import initialise
from gpjax.kernels.spectral import (
    SpectralRBF,
    to_spectral,
)


@pytest.mark.parametrize("n_basis", [1, 2, 10])
def test_initialise(n_basis):
    key = jr.PRNGKey(123)
    kernel = SpectralRBF(num_basis=n_basis)
    params = initialise(kernel)
    assert list(params.keys()) == ["basis_fns", "lengthscale", "variance"]
    for v in params.values():
        assert v.dtype == jnp.float64


@pytest.mark.parametrize("n_basis", [1, 2, 10])
def test_to_spectral(n_basis):
    base_kern = RBF()
    spectral = to_spectral(base_kern, n_basis)
    assert isinstance(spectral, SpectralRBF)
    assert spectral.num_basis == n_basis
    assert spectral.stationary


def test_spectral_density():
    kernel = SpectralRBF(num_basis=10)
    sdensity = kernel.spectral_density
    assert isinstance(sdensity, tfd.Normal)


def test_call():
    key = jr.PRNGKey(123)
    kernel = SpectralRBF(num_basis=10)
    params = initialise(kernel)
    x, y = jnp.array([[1.0]]), jnp.array([[0.5]])
    point_corr = kernel(x, y, params)
    assert isinstance(point_corr, jnp.DeviceArray)
    assert point_corr.shape == (1, 1)
