
# -*- coding: utf-8 -*-
"""Console script for deepgif."""

import sys
import click
import pathlib


@click.command()
@click.argument('input_path', nargs=1, type=click.Path(exists=True))
@click.option('--output_path', '-o',
    type=click.Path(),
)
@click.option(
    '--batch-size', '-b',
    default=1, type=int,
    show_default=True,
)
@click.option(
    '--window-cropping', '-c',
    default=2, type=int,
    show_default=True,
    help='cropping size of each output window',
)
@click.option(
    '--volume-padding', '-p',
    default=10, type=int,
    show_default=True,
    help='padding size of the input volume'
)
@click.option(
    '--window-size', '-w',
    default=128, type=int,
    show_default=True,
    help='size of the sliding window',
)
@click.option(
    '--cuda-device', '-d',
    default=0, type=int,
    show_default=True,
    help='select which CUDA device (GPU) to use'
)
@click.option(
    '--hist-niftynet/--hist-normal', '-h',
    default=False,
    show_default=True,
    help='use the volume mean as threshold for the histogram standardization'
)
def main(
        input_path,
        output_path,
        batch_size,
        window_cropping,
        volume_padding,
        window_size,
        cuda_device,
        hist_niftynet,
        ):
    """
    Parcellation of T1-weighted brain MRI using HighRes3DNet

    \b
    Larger batch sizes take more memory but inference is faster.
    Smaller window cropping values run faster but results are poorer.
    Larger volume padding values run slower but produce better results near the
    borders of the volume.
    Larger window size values take more memory but produces better results.
    """
    from highresnet.inference import infer
    if output_path is None:
        input_path = pathlib.Path(input_path)
        input_name = input_path.name
        output_name = input_name.replace('.nii', '_seg.nii')
        output_path = input_path.parent / output_name
    infer(
        input_path,
        output_path,
        batch_size,
        window_cropping,
        volume_padding,
        window_size,
        cuda_device,
        use_niftynet_hist_std=hist_niftynet,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
