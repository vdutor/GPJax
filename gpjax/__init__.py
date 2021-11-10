from jax.config import config

config.update("jax_enable_x64", True)
from .gps import Prior
from .types import Dataset

__version__ = "0.3.9"
