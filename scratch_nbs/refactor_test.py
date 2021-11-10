# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.0
#   kernelspec:
#     display_name: 'Python 3.8.0 64-bit (''venv'': venv)'
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2

# +
from gpjax.gps import Prior
from gpjax.likelihoods import Gaussian
from gpjax.types import Dataset
from gpjax.mean_functions import Zero
from gpjax.kernels import RBF

import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from time import time

key = jr.PRNGKey(123)

# +
N = 50
noise = 0.2

x = jr.uniform(key=key, minval=-3.0, maxval=3.0, shape=(N,)).sort().reshape(-1, 1)
latent_f = lambda x: jnp.sin(4 * x) + jnp.cos(2 * x) + 2
signal = latent_f(x)
y = signal + jr.normal(key, shape=signal.shape) * noise
xtest = jnp.linspace(-3.5, 3.5, 500).reshape(-1, 1)
ytest = latent_f(xtest)

training = Dataset(X=x, y=y)
testing = Dataset(X=xtest, y=ytest)
# -

kernel = RBF()
prior = Prior(RBF())

lik = Gaussian()
posterior = prior * lik

# +
from gpjax.parameters import unconstrain_parameters, constrain_parameters
from gpjax.objectives.mlls import marginal_log_likelihood
import jax
from jax.experimental import optimizers

nmll = jax.jit(marginal_log_likelihood(posterior, training, negative=True))

opt_init, opt_update, get_params = optimizers.adam(step_size=0.01)
opt_state = opt_init(posterior)


def step(i, opt_state):
    posterior = get_params(opt_state)
    posterior = unconstrain_parameters(posterior)
    v, g = jax.value_and_grad(nmll)(posterior, training)
    return opt_update(i, g, opt_state), v


nits = 500
mlls = []

start = time()
for i in range(nits):
    opt_state, mll_estimate = step(i, opt_state)
    mlls.append(-mll_estimate)
end = time() - start
print(f"Jitted: {end}")
# -


learned_posterior = get_params(opt_state)

mu = learned_posterior.mean(training)(testing)
sigma = jnp.diag(learned_posterior.variance(training)(testing))

plt.plot(training.X, training.y, "o", color="tab:blue")
plt.fill_between(
    testing.X.ravel(),
    mu.ravel() - sigma.ravel(),
    mu.ravel() + sigma.ravel(),
    alpha=0.5,
    color="tab:orange",
)
plt.plot(testing.X, mu, color="tab:orange")
plt.savefig("posterior.png")

# def step(i, opt_state):
#     posterior = get_params(opt_state)
#     # The model's parameters are not being unconstrained
#     posterior = unconstrain_parameters(posterior)
#     v, g = jax.value_and_grad(negative_marginal_ll)(posterior, training)
#     print(v)
#     print("Unconstrained:")
#     print(
#         "lengthscale: ",
#         posterior.prior.kernel.lengthscale,
#         "kernel variance",
#         posterior.prior.kernel.variance,
#         "obs_noise",
#         posterior.likelihood.obs_noise,
#     )
#     print("-" * 5)
#     print("Constrained:")
#     posterior2 = constrain_parameters(posterior)
#     print(
#         "lengthscale: ",
#         posterior2.prior.kernel.lengthscale,
#         "kernel variance",
#         posterior2.prior.kernel.variance,
#         "obs_noise",
#         posterior2.likelihood.obs_noise,
#     )
#     print("-" * 80)
#     return opt_update(i, g, opt_state), v
