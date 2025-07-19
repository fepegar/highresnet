"""Top-level package for highresnet."""

__author__ = """Fernando Perez-Garcia"""
__version__ = "0.10.2"

from . import cli
from .modules.highresnet import *

__all__ = [
    "__author__",
    "__version__",
    "cli",
]
