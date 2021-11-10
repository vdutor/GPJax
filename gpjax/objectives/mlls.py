from typing import Callable

from jax import jit
import jax.numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd

from ..gps import ConjugatePosterior
from ..kernels import gram
from ..parameters import constrain_parameters
from ..parameters.priors import evaluate_prior
from ..types import Dataset
from ..utils import I


def marginal_log_likelihood(
    gp: ConjugatePosterior,
    data: Dataset,
    negative: bool = False,
) -> Callable:
    r"""
    Compute :math:`\log p(y | x, \theta) for a conjugate, or exact, Gaussian process.
    Args:
        x: A set of N X M inputs
        y: A set of N X 1 outputs
    Returns: A multivariate normal distribution
    """
    # breakpoint()
    constant = jnp.array(-1.0) if negative else jnp.array(1.0)

    def neg_mll(gp, Dataset):
        x, y, n = Dataset.X, Dataset.y, Dataset.n
        gp = constrain_parameters(gp)
        mu = gp.prior.mean_function(x)
        gram_matrix = gram(gp.prior.kernel, x)
        gram_matrix += gp.likelihood.obs_noise * I(n)
        # gram_matrix += 1e-6 * I(n)
        L = jnp.linalg.cholesky(gram_matrix)
        random_variable = tfd.MultivariateNormalTriL(mu.squeeze(), L)

        log_prior_density = 0.0  # evaluate_prior(params, priors)
        return constant * (random_variable.log_prob(y.squeeze()).sum())

    return neg_mll


# @dispatch(NonConjugatePosterior)
# def marginal_ll(
#     gp: NonConjugatePosterior,
#     transform: Callable,
#     negative: bool = False,
#     jitter: float = 1e-6,
# ) -> Callable:
#     def mll(
#         params: dict,
#         training: Dataset,
#         priors: dict = {"latent": tfd.Normal(loc=0.0, scale=1.0)},
#         static_params: dict = None,
#     ):
#         x, y = training.X, training.y
#         n = training.n
#         params = transform(params)
#         if static_params:
#             params = concat_dictionaries([params, transform(static_params)])
#         link = gp.likelihood.link_function
#         gram_matrix = gram(gp.prior.kernel, x, params)
#         gram_matrix += I(n) * jitter
#         L = jnp.linalg.cholesky(gram_matrix)
#         F = jnp.matmul(L, params["latent"])
#         rv = link(F)
#         ll = jnp.sum(rv.log_prob(y))

#         priors = prior_checks(gp, priors)
#         log_prior_density = evaluate_prior(params, priors)
#         constant = jnp.array(-1.0) if negative else jnp.array(1.0)
#         return constant * (ll + log_prior_density)

#     return mll
