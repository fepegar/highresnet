# -*- coding: utf-8 -*-

"""Top-level package for highresnet."""

__author__ = """Fernando Perez-Garcia"""
__email__ = 'fernando.perezgarcia.17@ucl.ac.uk'
__version__ = '0.9.2'

import sys
INSTALL = '\nInstall it by running:\npip install "torch>=1.1"'
try:
    import torch
except ModuleNotFoundError:
    print('torch not found')
    print(INSTALL)
    sys.exit(1)
torch_version = torch.__version__
if torch_version < '1.1':
    message = (
        'Minimum torch version required is 1.1.0'
        ' but you are using {}'.format(torch_version)
    )
    print(message)
    print(INSTALL)
    sys.exit(1)

from . import cli
from .modules.highresnet import *
