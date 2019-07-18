import torch
from highresnet import HighRes3DNet

dependencies = ['torch']


def highres3dnet(*args, pretrained=False, **kwargs):
    model = HighRes3DNet(*args, **kwargs)
    if pretrained:
        state_dict_path = 'highres3dnet_li_parameters.pth'
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
    return model
