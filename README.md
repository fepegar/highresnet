# HighRes3DNet

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://badge.fury.io/py/highresnet.svg)](https://badge.fury.io/py/highresnet)
[![DOI](https://zenodo.org/badge/195385893.svg)](https://zenodo.org/badge/latestdoi/195385893)

PyTorch implementation of HighRes3DNet from [Li et al. 2017,
*On the Compactness, Efficiency, and Representation of
3D Convolutional Networks: Brain Parcellation as a Pretext Task*][li].

All the information about how the weights were ported from NiftyNet can be found
in [my submission to the MICCAI Educational Challenge 2019][mec].

A 2D version (`HighRes2DNet`) is also available.

[li]: https://arxiv.org/pdf/1707.01992.pdf
[mec]: https://nbviewer.jupyter.org/github/fepegar/miccai-educational-challenge-2019/blob/master/Combining_the_power_of_PyTorch_and_NiftyNet.ipynb?flush_cache=true

## Installation

### [PyTorch Hub](https://pytorch.org/hub)

If you are using the nightly version of PyTorch, you can import the model
directly from this repository using [PyTorch Hub](https://pytorch.org/hub).

```python
>>> import torch
>>> repo = 'fepegar/highresnet'
>>> model_name = 'highres3dnet'
>>> print(torch.hub.help(repo, model_name))

        "HighRes3DNet by Li et al. 2017 for T1-MRI brain parcellation"
        "pretrained (bool): load parameters from pretrained model"

>>> model = torch.hub.load(repo, model_name, pretrained=True)
```

### [PyPI](https://pypi.org/)

```shell
$ pip install highresnet
```

```python
>>> from highresnet import HighRes3DNet
>>> model = HighRes3DNet(in_channels=1, out_channels=160)
```

## Command line interface

```shell
$ deepgif t1_mri.nii.gz parcellation.nii.gz
Using cache found in /home/fernando/.cache/torch/hub/fepegar_highresnet_master
100%|███████████████████████████████████████████| 36/36 [01:13<00:00,  2.05s/it]
```
