import sys
INSTALL = '\nInstall it by running:\npip install "torch>=1.2"'
try:
    import torch
except ModuleNotFoundError:
    print('torch not found')
    print(INSTALL)
    sys.exit(1)
torch_version = torch.__version__
if torch_version < '1.2':
    message = (
        'Minimum torch version required is 1.2.0'
        ' but you are using {}'.format(torch_version)
    )
    print(message)
    print(INSTALL)
    sys.exit(1)

from .modules.highresnet import *
