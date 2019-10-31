import tempfile
from pathlib import Path
import numpy as np
import nibabel as nib
import SimpleITK as sitk
from .histogram import normalize

# From NiftyNet model zoo
LI_LANDMARKS = "4.4408920985e-16 8.06305571158 15.5085721044 18.7007018006 21.5032879029 26.1413278906 29.9862059045 33.8384058795 38.1891334787 40.7217966068 44.0109152758 58.3906435207 100.0"
LI_LANDMARKS = np.array([float(n) for n in LI_LANDMARKS.split()])


def preprocess(data, padding, hist_masking_function=None):
    # data = pad(data, padding)
    data = standardize(data, masking_function=hist_masking_function)
    data = whiten(data)
    data = data.astype(np.float32)
    data = pad(data, padding)  # should I pad at the beginning instead?
    return data


def pad(data, padding):
    # Should I use this value for padding?
    value = data[0, 0, 0]
    return np.pad(data, padding, mode='constant', constant_values=value)


def crop(data, padding):
    p = padding
    return data[p:-p, p:-p, p:-p]


def standardize(data, landmarks=LI_LANDMARKS, masking_function=None):
    return normalize(data, landmarks, masking_function=masking_function)


def whiten(data, masking_function=None):
    if masking_function is None:
        masking_function = mean_plus
    mask_data = masking_function(data)
    values = data[mask_data]
    mean, std = values.mean(), values.std()
    data -= mean
    data /= std
    return data


def mean_plus(data):
    return data > data.mean()


def resample_spacing(nifti, output_spacing, interpolation):
    output_spacing = tuple(output_spacing)
    temp_dir = Path(tempfile.gettempdir()) / '.deepgif'
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / 'deepgif_resampled.nii'
    temp_path = str(temp_path)

    nifti.to_filename(temp_path)
    image = sitk.ReadImage(temp_path)

    output_spacing = np.array(output_spacing).astype(float)
    output_spacing = tuple(output_spacing)

    reference_spacing = np.array(image.GetSpacing())
    reference_size = np.array(image.GetSize())

    output_size = reference_spacing / output_spacing * reference_size
    output_size = np.round(output_size).astype(np.uint32)
    # tuple(output_size) does not work, see
    # https://github.com/Radiomics/pyradiomics/issues/204
    output_size = output_size.tolist()

    identity = sitk.Transform(3, sitk.sitkIdentity)

    resample = sitk.ResampleImageFilter()
    resample.SetInterpolator(interpolation)
    resample.SetOutputDirection(image.GetDirection())
    resample.SetOutputOrigin(image.GetOrigin())  # TODO: double-check that this is correct
    resample.SetOutputPixelType(image.GetPixelID())
    resample.SetOutputSpacing(output_spacing)
    resample.SetSize(output_size)
    resample.SetTransform(identity)
    resampled = resample.Execute(image)
    sitk.WriteImage(resampled, temp_path)
    nifti_resampled = nib.load(temp_path)
    return nifti_resampled


def resample_ras_1mm_iso(nifti, interpolation=None):
    if interpolation is None:
        interpolation = sitk.sitkLinear
    nii_ras = nib.as_closest_canonical(nifti)
    spacing = nii_ras.header.get_zooms()
    one_iso = 1, 1, 1
    if np.allclose(spacing, one_iso):
        return nii_ras
    nii_resampled = resample_spacing(
        nii_ras,
        output_spacing=one_iso,
        interpolation=interpolation,
    )
    return nii_resampled


def resample_to_reference(
        reference_path,
        floating_path,
        result_path,
        interpolation=None,
        default_value=0.0,
        ):
    if interpolation is None:
        interpolation = sitk.sitkNearestNeighbor
    reference = sitk.ReadImage(str(reference_path))
    floating = sitk.ReadImage(str(floating_path))
    transform = sitk.Transform(3, sitk.sitkIdentity)
    resampled = sitk.Resample(
        floating,
        reference,
        transform,
        interpolation,
        default_value,
        floating.GetPixelID(),
    )
    sitk.WriteImage(resampled, str(result_path))
