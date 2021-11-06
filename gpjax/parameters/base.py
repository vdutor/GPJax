import jax.numpy as jnp
from multipledispatch import dispatch

from ..kernels import Kernel
from ..likelihoods import Likelihood
from ..mean_functions import MeanFunction
from ..utils import concat_dictionaries, merge_dictionaries, sort_dictionary
from typing import Dict

JaxKey = jnp.DeviceArray


def initialise(obj) -> Dict:
    return obj.parameters


#####################################
# Initialise GPs where one or more of the parameter's initialisation is stochastic
#####################################


# @dispatch(JaxKey, (ConjugatePosterior, SpectralPosterior))
# def initialise(key: JaxKey, gp: ConjugatePosterior) -> dict:
#     meanf = initialise(gp.prior.mean_function)
#     kernel = concat_dictionaries(initialise(key, gp.prior.kernel), meanf)
#     all_params = concat_dictionaries(kernel, initialise(gp.likelihood))
#     return sort_dictionary(all_params)


# Helper function for initialising the GP's mean and kernel function
def _initialise_hyperparams(kernel: Kernel, meanf: MeanFunction) -> dict:
    return concat_dictionaries((kernel), initialise(meanf))


##################################################
# Initialise the GP where all of the initialisations are stochastic
##################################################
# @dispatch((ConjugatePosterior, SpectralPosterior))
# def initialise(gp: ConjugatePosterior) -> dict:
#     hyps = _initialise_hyperparams(gp.prior.kernel, gp.prior.mean_function)
#     all_params = concat_dictionaries(hyps, initialise(gp.likelihood))
#     return sort_dictionary(all_params)


# @dispatch((ConjugatePosterior, SpectralPosterior), object)
# def initialise(gp: ConjugatePosterior, n_data):
#     return sort_dictionary(initialise(gp))


# @dispatch(NonConjugatePosterior, int)
# def initialise(gp: NonConjugatePosterior, n_data: int) -> dict:
#     hyperparams = _initialise_hyperparams(gp.prior.kernel, gp.prior.mean_function)
#     likelihood = concat_dictionaries(hyperparams, initialise(gp.likelihood))
#     latent_process = {"latent": jnp.zeros(shape=(n_data, 1))}
#     return sort_dictionary(concat_dictionaries(likelihood, latent_process))


# Helper function to complete a parameter set.
def complete(params: dict, gp, n_data: int = None) -> dict:
    full_param_set = initialise(gp, n_data)
    return sort_dictionary(merge_dictionaries(full_param_set, params))
