from pathlib import Path

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
import torchio as tio
from torch.utils.data import DataLoader
from .sampling import GridSampler, GridAggregator
from .preprocessing import (
    crop,
    preprocess,
    resample_ras_1mm_iso,
    resample_to_reference,
    mean_plus,
    LI_LANDMARKS,
)

def infer(
        input_path,
        output_path,
        batch_size,
        window_cropping,
        volume_padding,
        window_size,
        cuda_device,
        use_niftynet_hist_std=False,
        ):
    # Read image
    nii = nib.load(str(input_path))
    needs_resampling = check_header(nii)
    if needs_resampling:
        nii = resample_ras_1mm_iso(nii)
    data = nii.get_fdata()

    # Preprocessing
    hist_masking_function = mean_plus if use_niftynet_hist_std else None
    preprocessed = preprocess(
        data,
        volume_padding,
        hist_masking_function=hist_masking_function,
    )

    # Inference
    labels = run_inference(
        preprocessed,
        get_model(),
        window_size,
        window_border=window_cropping,
        batch_size=batch_size,
        cuda_device=cuda_device,
    )

    # Postprocessing
    if volume_padding:
        labels = crop(labels, volume_padding)
    nib.Nifti1Image(labels, nii.affine).to_filename(str(output_path))

    # Resample parcellation to original dimensions
    if needs_resampling:
        resample_to_reference(
            reference_path=input_path,
            floating_path=output_path,
            result_path=output_path,
        )


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
            print('Window size', window_size, 'is too large.')
            window_size = int(window_size * 0.75)
            print('Trying with smaller window size', window_size)

    return aggregator.output_array


def check_header(nifti_image):
    orientation = ''.join(nib.aff2axcodes(nifti_image.affine))
    spacing = nifti_image.header.get_zooms()
    one_iso = 1, 1, 1
    is_ras = orientation == 'RAS'
    if not is_ras:
        print(f'Detected orientation: {orientation}. Reorienting to RAS...')
    is_1_iso = np.allclose(spacing, one_iso)
    if not is_1_iso:
        print(f'Detected spacing: {spacing}. Resampling to 1 mm iso...')
    needs_resampling = not is_ras or not is_1_iso
    return needs_resampling


def get_device(cuda_device=0):
    return torch.device(
        'cuda:{}'.format(cuda_device) if torch.cuda.is_available() else 'cpu')


def to_tuple(value):
    try:
        iter(value)
    except TypeError:
        value = 3 * (value,)
    return value


def get_model(classifier=True):
    """
    Using PyTorch Hub as I haven't been able to install the .pth file
    within the pip package
    """
    repo = 'fepegar/highresnet'
    model_name = 'highres3dnet'
    model = torch.hub.load(repo, model_name, pretrained=True)
    if not classifier:
        model.block[6] = torch.nn.Identity()
    return model


def extract_features(input_path, output_path, pooling):
    device = get_device()
    model = get_model(classifier=False).to(device).eval()
    torch.set_grad_enabled(False)
    image = tio.ScalarImage(input_path)
    subject = tio.Subject(t1=image)
    transform = tio.HistogramStandardization(landmarks=dict(t1=LI_LANDMARKS))
    transformed = transform(subject)
    x = transformed.t1.data[None].to(device)  # None adds batch dimension
    logits = model(x).cpu()
    if pooling:
        global_average_pooled = logits.mean(dim=(-3, -2, -1))  # spatial dims
        features = global_average_pooled.squeeze().tolist()
        torch.save(features, output_path)
    else:
        features = logits.squeeze()
        output_dir = Path(output_path)
        image_name = image.path.name
        stem = image['stem']
        for i, feature_volume in enumerate(features):
            feature_image = tio.ScalarImage(
                tensor=feature_volume[None],
                affine=image.affine,
            )
            feature_name = image_name.replace(stem, f'{stem}_feature_{i}')
            feature_path = output_dir / feature_name
            feature_image.save(feature_path)
