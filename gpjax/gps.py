from typing import Optional, Callable

from dataclasses import dataclass
from abc import abstractmethod
from jax.scipy.linalg import cho_factor, cho_solve, solve_triangular
import jax.numpy as jnp
from treeo import Tree, field

from gpjax.parameters.base import Parameter

from .types import Array, Dataset
from .kernels import Kernel, gram, cross_covariance
from .likelihoods import (
    Gaussian,
    Likelihood,
    NonConjugateLikelihoods,
    NonConjugateLikelihoodType,
    predictive_moments,
)
from .mean_functions import MeanFunction, Zero
from .utils import I


##############
# GP priors
##############
@dataclass(repr=False)
class GP(Tree):
    kernel: Kernel
    mean_function: MeanFunction = Zero()
    name: Optional[str] = "Prior"

    def __repr__(self):
        meanf_string = self.prior.mean_function.__repr__()
        kernel_string = self.prior.kernel.__repr__()
        likelihood_string = self.likelihood.__repr__()
        return f"{self.name}\n{'-'*80}\n- {meanf_string}\n- {kernel_string}\n- {likelihood_string}"

    @abstractmethod
    def mean(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError

    @abstractmethod
    def variance(self) -> Callable[[Dataset], Array]:
        raise NotImplementedError


@dataclass
class Prior(GP):
    def __mul__(self, likelihood: Likelihood):
        return construct_posterior(self, likelihood)

    def mean(self) -> Callable[[Dataset], Array]:
        def mean_fn(data: Dataset):
            X = data.X
            mu = self.mean_function(X)
            return mu

        return mean_fn

    def variance(self) -> Callable[[Dataset], Array]:
        def variance_fn(data: Dataset, jitter_amount: float = 1e-8):
            X = data.X
            n_data = data.n
            Kff = gram(self.kernel, X)
            jitter_matrix = I(n_data) * jitter_amount
            covariance_matrix = Kff + jitter_matrix
            return covariance_matrix

        return variance_fn


##############
# GP Posteriors
##############
@dataclass
class Posterior(Tree):
    prior: Prior
    likelihood: Likelihood
    parameters = {}
    name: Optional[str] = "Posterior"

    def mean(self, training_data: Dataset) -> Callable[[Dataset], Array]:
        return super().mean(training_data)


@dataclass
class ConjugatePosterior(Posterior):
    prior: Prior
    likelihood: Gaussian
    name: Optional[str] = "Conjugate Posterior"

    def mean(self, training_data: Dataset) -> Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        n_train = training_data.n
        # Precompute covariance matrices
        prior_mean = self.prior.mean_function(X)
        Kff = gram(self.prior.kernel, X)
        Kff += I(n_train) * 1e-6
        obs_noise = self.likelihood.obs_noise
        L = cho_factor(Kff + I(n_train) * obs_noise, lower=True)
        prior_distance = y - prior_mean
        weights = cho_solve(L, prior_distance)

        def mean_fn(testing_data: Dataset) -> Array:
            x_test = testing_data.X
            prior_mean_at_test_inputs = self.prior.mean_function(x_test)
            Kfx = cross_covariance(self.prior.kernel, X, x_test)
            return prior_mean_at_test_inputs + jnp.dot(Kfx, weights)

        return mean_fn

    def variance(self, training_data: Dataset) -> Callable[[Dataset], Array]:
        X = training_data.X
        n_train = training_data.n
        obs_noise = self.likelihood.obs_noise
        n_train = training_data.n
        Kff = gram(self.prior.kernel, X)
        Kff += I(n_train) * 1e-6
        L = cho_factor(Kff + I(n_train) * obs_noise, lower=True)

        def variance_fn(test_inputs: Dataset) -> Array:
            x_test = test_inputs.X
            Kfx = cross_covariance(self.prior.kernel, X, x_test)
            Kxx = gram(self.prior.kernel, x_test)
            latents = cho_solve(L, Kfx.T)
            return Kxx - jnp.dot(Kfx, latents)

        return variance_fn


@dataclass
class NonConjugatePosterior(Posterior):
    prior: Prior
    likelihood: NonConjugateLikelihoodType
    name: Optional[str] = "Non-conjugate Posterior"

    def __post_init__(self):
        self.latent_vals: Array = field(
            default=jnp.zeros(shape=(self.likelihood.n, 1)), node=True, kind=Parameter
        )

    def mean(self, training_data: Dataset) -> Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X)
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def mean_fn(test_inputs: Dataset) -> Array:
            x_test = test_inputs.X
            Kfx = cross_covariance(self.prior.kernel, X, x_test)
            Kxx = gram(self.prior.kernel, x_test)
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, self.latent_vals)

            lvar = jnp.diag(latent_var)

            moment_fn = predictive_moments(self.likelihood)
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.mean()

        return mean_fn

    def variance(self, training_data: Dataset) -> Callable[[Dataset], Array]:
        X, y = training_data.X, training_data.y
        N = training_data.n
        Kff = gram(self.prior.kernel, X)
        L = jnp.linalg.cholesky(Kff + I(N) * 1e-6)

        def variance_fn(test_inputs: Dataset) -> Array:
            x_test = test_inputs.X
            Kfx = cross_covariance(self.prior.kernel, X, x_test)
            Kxx = gram(self.prior.kernel, x_test)
            A = solve_triangular(L, Kfx.T, lower=True)
            latent_var = Kxx - jnp.sum(jnp.square(A), -2)
            latent_mean = jnp.matmul(A.T, self.latent_vals)

            lvar = jnp.diag(latent_var)

            moment_fn = predictive_moments(self.likelihood)
            pred_rv = moment_fn(latent_mean.ravel(), lvar)
            return pred_rv.variance()

        return variance_fn


def construct_posterior(prior: Prior, likelihood: Likelihood) -> Posterior:
    if isinstance(likelihood, Gaussian):
        return ConjugatePosterior(prior, likelihood)
    elif likelihood in NonConjugateLikelihoods:
        return NonConjugatePosterior(prior, likelihood)
    else:
        raise NotImplementedError(
            f"No posterior implemented for {likelihood.name} likelihood"
        )
