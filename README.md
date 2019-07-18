# HighRes3DNet

PyTorch implementation of HighRes3DNet from [Li et al., 2017,
*On the Compactness, Efficiency, and Representation of
3D Convolutional Networks: Brain Parcellation as a Pretext Task*][li].

A 2D version (HighRes2DNet) is also available.

[li]: https://arxiv.org/pdf/1707.01992.pdf

## Installation

### PyTorch hub

If you are using the nightly version of PyTorch, you can import the model
directly from this repository using [PyTorch hub](https://pytorch.org/hub).

```python
>>> import torch
>>> repo = 'fepegar/highresnet'
>>> model_name = 'highres3dnet'
>>> print(torch.hub.help(repo, model_name))

        "HighRes3DNet by Li et al. 2017 for T1-MRI brain parcellation"
    
>>> model = torch.hub.load(repo, model_name, in_channels=1, out_channels=160)
```

### `pip`

```shell
$ pip install highresnet
```

```python
>>> from highresnet import HighRes3DNet
>>> model = HighRes3DNet(in_channels=1, out_channels=160)
```




