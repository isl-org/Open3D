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

from pathlib import Path
import urllib.request
import concurrent.futures
import json
import hashlib
import io
import time
import sys

# Typically "Open3D/examples/test_data", the test data dir.
_test_data_dir = Path(__file__).parent.absolute().resolve()

# Typically "Open3D/examples/test_data/open3d_downloads", the test data download dir.
_download_dir = _test_data_dir / "open3d_downloads"

sys.path.append(str(_test_data_dir))
from download_file_list import map_url_to_relative_path


def _download_file(url, save_path, overwrite):
    """
    Args:
        url: Direct download URL.
        save_path: Absolute full path to save the downloaded file.
        overwrite: If true, overwrite the existing file.
    """
    save_path = Path(save_path)

    # The saved file must be inside _test_data_dir.
    if _download_dir not in save_path.parents:
        raise AssertionError(f"{save_path} must be inside {_download_dir}.")

    # Supports sub directory inside _test_data_dir, e.g.
    # Open3D/examples/test_data/open3d_downloads/foo/bar/my_file.txt
    save_path.parent.mkdir(parents=True, exist_ok=True)

    if save_path.exists() and not overwrite:
        print(f"[open3d_downloads] {str(save_path)} already exists, skipped.")
    else:
        try:
            urllib.request.urlretrieve(url, save_path)
            print(f"Downloaded {url}\n        to {str(save_path)}")
        except Exception as e:
            print(f"Failed to download {url}: {str(e)}")


def download_all_files(overwrite=False):
    for url, relative_path in map_url_to_relative_path.items():
        save_path = _download_dir / relative_path
        _download_file(url, save_path, overwrite)
