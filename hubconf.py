import torch

dependencies = ['torch']


def highres2dnet(*args, **kwargs):
    """
    HighRes2DNet in the style of
    HighRes3DNet by Li et al. 2017 for T1-MRI brain parcellation
    """
    from highresnet import HighRes2DNet
    model = HighRes2DNet(*args, **kwargs)
    return model


def highres3dnet(*args, pretrained=False, **kwargs):
    """
    HighRes3DNet by Li et al. 2017 for T1-MRI brain parcellation
    pretrained (bool): load parameters from pretrained model
    """
    from highresnet import HighRes3DNet
    if pretrained:
        model = HighRes3DNet(
            *args,
            in_channels=1,
            out_channels=160,
            add_dropout_layer=True,
            **kwargs,
        )
        url = 'https://github.com/fepegar/highresnet/raw/master/highres3dnet_li_parameters-7d297872.pth'
        state_dict = torch.hub.load_state_dict_from_url(url, progress=False, map_location='cpu')
        model.load_state_dict(state_dict)
    else:
        model = HighRes3DNet(*args, **kwargs)
    return model
