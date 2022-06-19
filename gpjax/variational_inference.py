import abc
from logging import root
from pickle import FALSE
from typing import Callable, Dict, Optional

import jax.numpy as jnp
import jax.scipy as jsp
from chex import dataclass
from jax import vmap

from .config import get_defaults
from .gps import AbstractPosterior
from .kernels import cross_covariance, diagonal, gram
from .likelihoods import Gaussian
from .parameters import transform
from .quadrature import gauss_hermite_quadrature
from .types import Array, Dataset
from .utils import I, concat_dictionaries
from .variational_families import AbstractVariationalFamily, VariationalGaussian

DEFAULT_JITTER = get_defaults()["jitter"]

DEFAULT_JITTER = 1e-4


@dataclass
class AbstractVariationalInference:
    """A base class for inference and training of variational families against an extact posterior"""

    posterior: AbstractPosterior
    variational_family: AbstractVariationalFamily

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood

    @property
    def params(self) -> Dict:
        """Construct the parameter set used within the variational scheme adopted."""
        hyperparams = concat_dictionaries(
            {"likelihood": self.posterior.likelihood.params},
            self.variational_family.params,
        )
        return hyperparams

    @abc.abstractmethod
    def elbo(
        self, train_data: Dataset, transformations: Dict
    ) -> Callable[[Dict], Array]:
        """Placeholder method for computing the evidence lower bound function (ELBO), given a training dataset and a set of transformations that map each parameter onto the entire real line.

        Args:
            train_data (Dataset): The training dataset for which the ELBO is to be computed.
            transformations (Dict): A set of functions that unconstrain each parameter.

        Returns:
            Callable[[Array], Array]: A function that computes the ELBO given a set of parameters.
        """
        raise NotImplementedError


@dataclass
class StochasticVI(AbstractVariationalInference):
    """Stochastic Variational inference training module. The key reference is Hensman et. al., (2013) - Gaussian processes for big data."""

    def __post_init__(self):
        self.prior = self.posterior.prior
        self.likelihood = self.posterior.likelihood
        self.num_inducing = self.variational_family.num_inducing

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate and batch of data for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)

            # KL[q(f(·)) || p(f(·))]
            kl = self.variational_family.prior_kl(params)

            # ∫[log(p(y|f(x))) q(f(x))] df(x)
            var_exp = self.variational_expectation(params, batch)

            # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
            return constant * (jnp.sum(var_exp) * train_data.n / batch.n - kl)

        return elbo_fn

    def variational_expectation(self, params: Dict, batch: Dataset) -> Array:
        """Compute the expectation of our model's log-likelihood under our variational distribution. Batching can be done here to speed up computation.

        Args:
            params (Dict): The set of parameters that induce our variational approximation.
            batch (Dataset): The data batch for which the expectation should be computed for.

        Returns:
            Array: The expectation of the model's log-likelihood under our variational distribution.
        """
        x, y = batch.X, batch.y

        # q(f(x))
        predictive_dist = vmap(self.variational_family.predict(params))(x[:, None])
        mean = predictive_dist.mean().val.reshape(-1, 1)
        variance = predictive_dist.variance().val.reshape(-1, 1)

        # log(p(y|f(x)))
        log_prob = vmap(
            lambda f, y: self.likelihood.link_function(
                f, params["likelihood"]
            ).log_prob(y)
        )

        # ≈ ∫[log(p(y|f(x))) q(f(x))] df(x)
        expectation = gauss_hermite_quadrature(log_prob, mean, variance, y=y)

        return expectation


@dataclass
class OSGPR(AbstractVariationalInference):
    """
    Online Sparse Variational GP regression (collapsed bound).

    Key reference - Bui et al. (2017) Streaming Gaussian process approximations
    """

    posterior: AbstractPosterior
    variational_family: AbstractVariationalFamily
    variational_family_old: AbstractVariationalFamily
    params_old: dict
    jitter: Optional[float] = DEFAULT_JITTER

    def __post_init__(self):
        assert isinstance(self.posterior.likelihood, Gaussian)
        self.variational_family_new = self.variational_family

        # Need to assert variational family is AbsractVariationalGaussian -> once natgrads is merged.

    def elbo(
        self, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        # a is old inducing points, b is new.
        params_old = self.params_old

        # Old and new variational family.
        q_old = self.variational_family_old
        q_new = self.variational_family_new

        # Old and new number of inducing points.
        m_old = q_old.num_inducing
        m_new = q_new.num_inducing

        # Old and new kernel.
        kernel_old = q_old.prior.kernel
        kernel_new = q_new.prior.kernel

        # Old inducing input locations and corresponding gram matrix:
        a = self.params_old["variational_family"]["inducing_inputs"]
        Kaa_old = gram(kernel_old, a, kernel_old.params)
        Kaa_old += I(m_old) * self.jitter
        La_old = jnp.linalg.cholesky(Kaa_old)

        # Obtain variational mean, ma, and covariance, Sa, at old inducing inputs, a.
        if isinstance(q_old, VariationalGaussian):
            ma = params_old["variational_family"]["variational_mean"]
            sqrta = params_old["variational_family"]["variational_root_covariance"]
            Sa = jnp.matmul(sqrta, sqrta.T)

        else:
            ma, Sa = q_old(params_old)
            Sa += I(m_old) * self.jitter
            sqrta = jnp.linalg.cholesky(Sa)

        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        # Compute Sa⁻¹ ma.
        sqrta_inv_ma = jsp.linalg.solve_triangular(sqrta, ma, lower=True)
        Sa_inv_ma = jsp.linalg.solve_triangular(sqrta.T, sqrta_inv_ma, lower=False)

        def elbo_fn(params: Dict, data_new: Dataset) -> Array:
            params = transform(params, transformations)
            x, y, n = data_new.X, data_new.y, data_new.n

            noise = params["likelihood"]["obs_noise"]
            b = params["variational_family"]["inducing_inputs"]

            Kaa = gram(kernel_new, a, params["kernel"])
            Kaa += I(m_old) * self.jitter
            La = jnp.linalg.cholesky(Kaa)

            Kbb = gram(kernel_new, b, params["kernel"])
            Kbb += I(m_new) * self.jitter
            Lb = jnp.linalg.cholesky(Kbb)

            Kbx = cross_covariance(kernel_new, b, x, params["kernel"])
            Kba = cross_covariance(kernel_new, b, a, params["kernel"])

            µx = q_new.prior.mean_function(x, params["mean_function"])

            diff = y - µx

            # c = 1/σ² Kbx y + Kba Sa⁻¹ ma
            c = jnp.matmul(Kbx, y / noise) + jnp.matmul(Kba, Sa_inv_ma)

            # Lb⁻¹ c
            Lb_inv_c = jsp.linalg.solve_triangular(Lb, c, lower=True)

            # Lb⁻¹ Kba
            Lb_inv_Kba = jsp.linalg.solve_triangular(Lb, Kba, lower=True)

            # Lb⁻¹ Kbx
            Lb_inv_Kbx = jsp.linalg.solve_triangular(Lb, Kbx, lower=True)

            # 1/σ² Lb⁻¹ Kbx Kxb Lb⁻ᵀ
            d1 = jnp.matmul(Lb_inv_Kbx, Lb_inv_Kbx.T) / noise

            # sqrta⁻¹ Kab Lb⁻ᵀ
            sqrta_inv_Kab_Lb_invT = jsp.linalg.solve_triangular(
                sqrta, Lb_inv_Kba.T, lower=True
            )
            # Lb⁻¹ Kba Sa⁻¹ Kab Lb⁻ᵀ
            d2 = jnp.matmul(sqrta_inv_Kab_Lb_invT.T, sqrta_inv_Kab_Lb_invT)

            # L'a⁻¹ Kab Lb⁻ᵀ
            La_old_inv_Kab_Lb_invT = jsp.linalg.solve_triangular(
                La_old, Lb_inv_Kba.T, lower=True
            )

            # Lb⁻¹ Kba K'aa⁻¹ Kab Lb⁻ᵀ
            d3 = jnp.matmul(La_old_inv_Kab_Lb_invT.T, La_old_inv_Kab_Lb_invT)

            # D = I + Lb⁻¹[1/σ² Kbx Kxb + Kba Sa⁻¹ Kab - Kba K'aa⁻¹ Kab] Lb⁻ᵀ
            D = I(m_new) + d1 + d2 - d3
            D += I(m_new) * self.jitter

            # LLᵀ = D
            L = jnp.linalg.cholesky(D)

            # L⁻¹ Lb⁻¹ c
            L_inv_Lb_inv_c = jsp.linalg.solve_triangular(L, Lb_inv_c, lower=True)

            Kxx_diag = diagonal(kernel_new, x, params["kernel"])

            # quadratic term

            # -1/2σ² (y - µx)ᵀ(y - µx)
            quad = -0.5 * jnp.sum(jnp.square(diff)) / noise

            # -1/2 maᵀ Sa⁻¹ ma
            quad += -0.5 * jnp.sum(jnp.square(sqrta_inv_ma))

            # 1/2 cᵀ Lb⁻ᵀ D⁻¹ Lb⁻¹
            quad += 0.5 * jnp.sum(jnp.square(L_inv_Lb_inv_c))

            # log |D| = 2 trace(log|L|)
            log_det = 2 * jnp.sum(jnp.log(jnp.diagonal(L)))

            # log probs
            log_prob = -0.5 * n * jnp.log(2.0 * jnp.pi * noise) + quad - 0.5 * log_det

            # ∆2 = -1/2 tr(Qx) with Qx:= Kxx - Kxb Kbb⁻¹ Kbx
            delta2 = -0.5 * (jnp.sum(Kxx_diag) / noise - jnp.sum(jnp.diagonal(d1)))

            # ∆1 = log|K'aa| - log|Sa|    [+ log|Da| ??]
            delta1 = jnp.sum(jnp.log(jnp.diagonal(La_old))) - jnp.sum(
                jnp.log(jnp.diagonal(sqrta))
            )

            # Qa = Kaa - Kab Kbb⁻ᵀ
            Qa = Kaa - jnp.matmul(Lb_inv_Kba.T, Lb_inv_Kba)
            sqrta_inv_Qa = jsp.linalg.solve_triangular(sqrta, Qa, lower=True)

            # Sa⁻¹Qa
            Sa_inv_Qa = jsp.linalg.solve_triangular(sqrta.T, sqrta_inv_Qa, lower=False)

            # La⁻¹Qa
            La_inv_Qa = jsp.linalg.solve_triangular(La, Qa, lower=True)

            # Kaa⁻¹Qa
            Kaa_inv_Qa = jsp.linalg.solve_triangular(La.T, La_inv_Qa, lower=False)

            # tr(Da⁻¹Qa) = tr([Sa⁻¹ - Kaa⁻¹]Qa) = tr(Sa⁻¹Qa) - tr(Kaa⁻¹Qa)
            delta1 += jnp.sum(jnp.diag(Sa_inv_Qa) - jnp.diag(Kaa_inv_Qa))

            return constant * (log_prob + delta1 + delta2).squeeze()

        return elbo_fn


@dataclass
class OSVGP(StochasticVI):
    """
     Online Sparse Variational GP (uncollapsed bound).

    Key reference - Bui et al. (2017) Streaming Gaussian process approximations
    """

    posterior: AbstractPosterior
    variational_family: AbstractVariationalFamily
    variational_family_old: AbstractVariationalFamily
    params_old: dict
    jitter: Optional[float] = DEFAULT_JITTER

    def online_regulariser(self, params: dict) -> Array:
        # a is old inducing points, b is new.
        params_old = self.params_old

        # Old and new variational family.
        q_old = self.variational_family_old
        q_new = self.variational_family

        # Old number of inducing points.
        m_old = q_old.num_inducing

        # Old kernel.
        kernel_old = q_old.prior.kernel

        # Old inducing input locations and corresponding gram matrix:
        a = self.params_old["variational_family"]["inducing_inputs"]
        Kaa_old = gram(kernel_old, a, kernel_old.params)
        Kaa_old += I(m_old) * self.jitter
        La_old = jnp.linalg.cholesky(Kaa_old)

        # Obtain variational mean, ma, and covariance, Sa, at old inducing inputs, a.
        if isinstance(q_old, VariationalGaussian):
            ma = params_old["variational_family"]["variational_mean"]
            sqrta = params_old["variational_family"]["variational_root_covariance"]
            Sa = jnp.matmul(sqrta, sqrta.T)

        else:
            ma, Sa = q_old(params_old)
            Sa += I(m_old) * self.jitter
            sqrta = jnp.linalg.cholesky(Sa)

        # a is old inducing points, b is new
        q_newa = q_new.predict(params)(a)
        mu, Sigma = q_newa.mean(), q_newa.covariance()
        Smm = Sigma + jnp.matmul(mu, mu.T)
        obj = jnp.sum(jnp.log(jnp.diagonal(La_old)))
        obj += -jnp.sum(jnp.log(jnp.diagonal(sqrta)))

        sqrta_inv_ma = jsp.linalg.solve_triangular(sqrta, ma, lower=True)
        Sa_inv_ma = jsp.linalg.solve_triangular(sqrta, sqrta_inv_ma, lower=False)
        obj += -0.5 * jnp.sum(ma * Sa_inv_ma)
        obj += jnp.sum(mu * Sa_inv_ma)

        sqrta_inv_Smm = jsp.linalg.solve_triangular(sqrta, Smm, lower=True)
        Sa_inv_Smm = jsp.linalg.solve_triangular(sqrta, sqrta_inv_Smm, lower=False)
        La_inv_Smm = jsp.linalg.solve_triangular(La_old, Smm, lower=True)
        Ka_inv_Smm = jsp.linalg.solve_triangular(sqrta, La_inv_Smm, lower=False)

        obj += -0.5 * jnp.sum(jnp.diagonal(Sa_inv_Smm) - jnp.diagonal(Ka_inv_Smm))
        return obj.squeeze()

    def elbo(
        self, train_data: Dataset, transformations: Dict, negative: bool = False
    ) -> Callable[[Array], Array]:
        """Compute the evidence lower bound under this model. In short, this requires evaluating the expectation of the model's log-likelihood under the variational approximation. To this, we sum the KL divergence from the variational posterior to the prior. When batching occurs, the result is scaled by the batch size relative to the full dataset size.

        Args:
            train_data (Dataset): The training data for which we should maximise the ELBO with respect to.
            transformations (Dict): The transformation set that unconstrains each parameter.
            negative (bool, optional): Whether or not the resultant elbo function should be negative. For gradient descent where we minimise our objective function this argument should be true as minimisation of the negative corresponds to maximisation of the ELBO. Defaults to False.

        Returns:
            Callable[[Dict, Dataset], Array]: A callable function that accepts a current parameter estimate and batch of data for which gradients should be computed.
        """
        constant = jnp.array(-1.0) if negative else jnp.array(1.0)

        def elbo_fn(params: Dict, batch: Dataset) -> Array:
            params = transform(params, transformations)

            # KL[q(f(·)) || p(f(·))]
            kl = self.variational_family.prior_kl(params)

            # ∫[log(p(y|f(x))) q(f(x))] df(x)
            var_exp = self.variational_expectation(params, batch)

            # explain maths
            online_reg = self.online_regulariser(params)

            # For batch size b, we compute  n/b * Σᵢ[ ∫log(p(y|f(xᵢ))) q(f(xᵢ)) df(xᵢ)] - KL[q(f(·)) || p(f(·))]
            return constant * (
                jnp.sum(var_exp) * train_data.n / batch.n - kl + online_reg
            )

        return elbo_fn
