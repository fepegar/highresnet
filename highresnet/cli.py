
# -*- coding: utf-8 -*-

"""Console script for staple."""
import sys
import click
import nibabel as nib
from tqdm import tqdm
try:
    import torch
except ModuleNotFoundError:
    print('torch not found. Install it running "pip install torch"')
torch_version = torch.__version__
if torch_version < '1.1.0':
    message = (
        'Minimum torch version required is 1.1.0'
        ' but you are using {}'.format(torch_version)
    )
    raise Exception(message)
from torch.utils.data import DataLoader
from .sampling import GridSampler, GridAggregator
from .preprocessing import preprocess, crop

@click.command()
@click.argument('input_path', nargs=1, type=click.Path(exists=True))
@click.argument('output_path', nargs=1, type=click.Path())
@click.option('--border', '-b', default=2, type=int, show_default=True)
@click.option('--volume-padding', '-p', default=10, type=int, show_default=True)
@click.option('--window-size', '-w', default=128, type=int, show_default=True)
@click.option('--cuda-device', '-c', default=0, type=int, show_default=True)
def main(input_path, output_path, border, volume_padding, window_size, cuda_device):
    nii = nib.load(input_path)
    data = nii.get_fdata()
    preprocessed = preprocess(data, volume_padding)
    labels = run_inference(
        preprocessed,
        get_model(),
        window_size,
        window_border=border,
        batch_size=1,
        cuda_device=cuda_device,
    )
    if volume_padding:
        labels = crop(labels, volume_padding)
    nib.Nifti1Image(labels, nii.affine).to_filename(output_path)


def run_inference(
        data,
        model,
        window_size,
        window_border=0,
        batch_size=2,
        cuda_device=0,
        ):
    success = False
    while not success:
        window_sizes = to_tuple(window_size)
        window_border = to_tuple(window_border)

        sampler = GridSampler(data, window_sizes, window_border)
        aggregator = GridAggregator(data, window_border)
        loader = DataLoader(sampler, batch_size=batch_size)

        device = get_device(cuda_device=cuda_device)
        model.to(device)
        model.eval()

        CHANNELS_DIMENSION = 1

        try:
            with torch.no_grad():
                for batch in tqdm(loader):
                    input_tensor = batch['image'].to(device)
                    locations = batch['location']
                    logits = model(input_tensor)
                    labels = logits.argmax(dim=CHANNELS_DIMENSION, keepdim=True)
                    outputs = labels
                    aggregator.add_batch(outputs, locations)
            success = True
        except RuntimeError as e:
            print(e)
            window_size = int(window_size * 0.75)
            print('Trying with window size', window_size)

    return aggregator.output_array


def get_device(cuda_device=0):
    return torch.device(f'cuda:{cuda_device}' if torch.cuda.is_available() else 'cpu')


def to_tuple(value):
    try:
        iter(value)
    except TypeError:
        value = 3 * (value,)
    return value


def get_model():
    """
    Using PyTorch Hub as I haven't been able to install the .pth file
    within the pip package
    """
    repo = 'fepegar/highresnet'
    model_name = 'highres3dnet'
    model = torch.hub.load(repo, model_name, pretrained=True)
    return model


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
