
# -*- coding: utf-8 -*-
"""Console script for deepgif."""

import sys
import click
import pathlib


@click.command()
@click.argument('input-path', type=click.Path(exists=True))
@click.argument('output-path', type=click.Path())
@click.option(
    '--pooling/--no-pooling', '-p',
    default=True,
    show_default=True,
    help='perform global average pooling to generate a feature vector'
)
def main(
        input_path,
        output_path,
        pooling,
        ):
    """
    Extract deep features from T1-weighted brain MRI using HighRes3DNet
    """
    from highresnet.inference import extract_features
    extract_features(
        input_path,
        output_path,
        pooling,
    )


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
