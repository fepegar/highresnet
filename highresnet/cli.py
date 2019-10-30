
# -*- coding: utf-8 -*-
"""Console script for deepgif."""

import sys
import click


@click.command()
@click.argument('input_path', nargs=1, type=click.Path(exists=True))
@click.argument('output_path', nargs=1, type=click.Path())
@click.option(
    '--batch-size', '-b',
    default=1, type=int,
    show_default=True,
)
@click.option(
    '--window-cropping', '-c',
    default=16, type=int,
    show_default=True,
    help='cropping size of each output window',
)
@click.option(
    '--volume-padding', '-p',
    default=16, type=int,
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
)
def main(
        input_path,
        output_path,
        batch_size,
        window_cropping,
        volume_padding,
        window_size,
        cuda_device,
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
    infer(
        input_path,
        output_path,
        batch_size,
        window_cropping,
        volume_padding,
        window_size,
        cuda_device,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
