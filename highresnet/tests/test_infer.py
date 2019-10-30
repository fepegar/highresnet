import tarfile
import tempfile
import urllib.request
from pathlib import Path
from configparser import ConfigParser

def get_data_url_from_model_zoo():
    url = 'https://raw.githubusercontent.com/NifTK/NiftyNetModelZoo/5-reorganising-with-lfs/highres3dnet_brain_parcellation/main.ini'
    with urllib.request.urlopen(url) as response:
        config_string = response.read().decode()
    config = ConfigParser()
    config.read_string(config_string)
    data_url = config['data']['url']
    return data_url


def download_data(data_url):
    tempdir = Path(tempfile.gettempdir())
    download_dir = tempdir / 'downloaded_data'
    download_dir.mkdir(exist_ok=True)
    data_path = download_dir / Path(data_url).name
    print(data_path)
    if not data_path.is_file():
        urllib.request.urlretrieve(data_url, data_path)
    with tarfile.open(data_path, 'r') as tar:
        tar.extractall(download_dir)
    nifti_files = download_dir.glob('**/*.nii.gz')
    return list(nifti_files)[0]


def test_infer():
    image_path = download_data(get_data_url_from_model_zoo())
    # TODO
