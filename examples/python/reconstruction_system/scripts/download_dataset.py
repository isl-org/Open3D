# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
"""
Cross-platform Python script to download and extract datasets. All common
archive formats are supported.
python download_dataset.py dataset_name ... [--cache download-folder]
[--extract-to local-folder]
"""
import urllib.request
import shutil
import os
import argparse

dataset_dict = {
    "L515_test":
        "https://storage.googleapis.com/isl-datasets/open3d-dev/realsense/L515_test.bag",
    "L515_paper_lantern":
        "https://storage.googleapis.com/isl-datasets/open3d-dev/realsense/L515_paper_lantern.bag",
    "L515_JackJack":
        "https://storage.googleapis.com/isl-datasets/open3d-dev/realsense/L515_JackJack.bag"
}


def download(dataset_name, download_folder, extract_folder):
    """
    Download dataset 'dataset_name' to 'download_folder'. If it is an archive,
    extract it to 'extract_folder', else move it there.
    """
    cache_path = os.path.join(download_folder,
                              os.path.basename(dataset_dict[dataset_name]))
    extract_path = os.path.join(extract_folder, dataset_name)
    os.makedirs(os.path.realpath(extract_path), exist_ok=True)
    with urllib.request.urlopen(dataset_dict[dataset_name]) as response:
        with open(cache_path, 'wb') as cp_obj:
            shutil.copyfileobj(response, cp_obj)
            try:
                shutil.unpack_archive(cache_path, extract_path)
            except (ValueError, shutil.ReadError):  # Not an archive
                shutil.move(cache_path, extract_path)
    print(f"Downloaded dataset {dataset_name} to {extract_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description='Download and extract datasets.')
    parser.add_argument('dataset_name',
                        nargs='*',
                        help='Names of datasets to download.')
    parser.add_argument('-c',
                        '--cache',
                        default='.',
                        help='Download folder for caching archives.')
    parser.add_argument('-e',
                        '--extract-to',
                        default='../datasets/',
                        help='Extract downloaded datasets here.')
    parser.add_argument('-l',
                        '--list-datasets',
                        action='store_true',
                        help='List datasets')

    args = parser.parse_args()
    if args.list_datasets:
        print('Available datasets:\n\t' + '\n\t'.join(dataset_dict.keys()))
    for ds in args.dataset_name:
        download(ds, args.cache, args.extract_to)
