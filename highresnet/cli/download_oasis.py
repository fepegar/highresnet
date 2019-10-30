
# -*- coding: utf-8 -*-
"""Console script for deepgif."""

import os
import sys
import shutil
import urllib
import tarfile

import click


@click.command()
def main():
    """
    http://blog.ppkt.eu/2014/06/python-urllib-and-tarfile/
    """
    url = 'https://github.com/NifTK/NiftyNetModelZoo/raw/5-reorganising-with-lfs/highres3dnet_brain_parcellation/data.tar.gz'
    file_tmp = urllib.request.urlretrieve(url, filename=None)[0]
    base_name = os.path.basename(url)
    dir_name = base_name.replace('.tar.gz', '')
    tar = tarfile.open(file_tmp)
    tar.extractall(dir_name)
    nii_filename = 'OAS1_0145_MR2_mpr_n4_anon_sbj_111.nii.gz'
    nii_filepath = os.path.join(dir_name, nii_filename)
    os.rename(nii_filepath, nii_filename)
    shutil.rmtree(dir_name)
    print(nii_filename)


if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    sys.exit(main())  # pragma: no cover
