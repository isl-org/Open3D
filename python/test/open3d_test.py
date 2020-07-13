import os
import sys
import urllib.request
import zipfile

# Avoid pathlib to be compatible with Python 3.5+.
__pwd = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(__pwd, os.pardir, os.pardir, "examples",
                             "test_data")


def download_fountain_dataset():
    fountain_path = os.path.join(test_data_dir, "fountain_small")
    fountain_zip_path = os.path.join(test_data_dir, "fountain.zip")
    if not os.path.exists(fountain_path):
        print("Downloading fountain dataset")
        url = "https://storage.googleapis.com/isl-datasets/open3d-dev/fountain.zip"
        urllib.request.urlretrieve(url, fountain_zip_path)
        print("Extracting fountain dataset")
        with zipfile.ZipFile(fountain_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(fountain_path))
        os.remove(fountain_zip_path)
    return fountain_path
