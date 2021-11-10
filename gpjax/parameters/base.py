import jax.numpy as jnp

import tensorflow_probability.substrates.jax as tfp
from dataclasses import dataclass
import treeo as to
import jax

JaxKey = jnp.DeviceArray


@dataclass
class Parameter:
    bijector: tfp.bijectors.Bijector = tfp.bijectors.Softplus()

    def constrain(self, x):
        return self.bijector.forward(x)

    def unconstrain(self, x):
        return self.bijector.inverse(x)


@dataclass(repr=False)
class PositiveParameter(Parameter):
    bijector = tfp.bijectors.Softplus()


@dataclass
class ShiftedPositive(Parameter):
    shift = tfp.bijectors.Shift(1e-6)
    splus = tfp.bijectors.Softplus()
    bijector = tfp.bijectors.Chain([shift, splus])


def unconstrain_parameters(model):
    def unconstrain(f):
        # print(f.kind().unconstrain(f.value))
        return f.kind().unconstrain(f.value)

    m1 = to.map(unconstrain, model, Parameter, field_info=True)

    return m1


def constrain_parameters(model):
    def constrain(f):
        return f.kind().constrain(f.value)

    m1 = to.map(constrain, model, Parameter, field_info=True)
    return m1
