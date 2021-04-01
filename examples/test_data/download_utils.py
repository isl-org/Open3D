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

import concurrent.futures
import hashlib
import io
import json
import time
import urllib.request
from pathlib import Path

# Typically "Open3D/examples/test_data", the test data dir.
_test_data_dir = Path(__file__).parent.absolute().resolve()

# Typically "Open3D/examples/test_data/open3d_downloads", the download dir.
_download_dir = _test_data_dir / "open3d_downloads"


def _compute_sha256(path):
    """
    Returns sha256 checksum as string.
    """
    # http://stackoverflow.com/a/17782753 with fixed block size
    algo = hashlib.sha256()
    with io.open(str(path), 'br') as f:
        for chunk in iter(lambda: f.read(4096), b''):
            algo.update(chunk)
    return algo.hexdigest()


def _download_file(url, path, sha256, max_retry=3):
    if max_retry == 0:
        raise OSError(f"max_retry reached, cannot download {url}.")

    full_path = _download_dir / Path(path)

    # The saved file must be inside _test_data_dir.
    if _download_dir not in full_path.parents:
        raise AssertionError(f"{full_path} must be inside {_download_dir}.")

    # Supports sub directory inside _test_data_dir, e.g.
    # Open3D/examples/test_data/open3d_downloads/foo/bar/my_file.txt
    full_path.parent.mkdir(parents=True, exist_ok=True)

    if full_path.exists() and _compute_sha256(full_path) == sha256:
        print(f"[download_utils.py] {str(full_path)} already exists, skipped.")
        return

    try:
        urllib.request.urlretrieve(url, full_path)
        print(
            f"[download_utils.py] Downloaded {url}\n        to {str(full_path)}"
        )
        if _compute_sha256(full_path) != sha256:
            raise ValueError(f"{path}'s SHA256 checksum incorrect:\n"
                             f"- Expected: {sha256}\n"
                             f"- Actual  : {_compute_sha256(full_path)}")
    except Exception as e:
        sleep_time = 5
        print(f"[download_utils.py] Failed to download {url}: {str(e)}")
        print(f"[download_utils.py] Retrying in {sleep_time}s")
        time.sleep(sleep_time)
        _download_file(url, path, sha256, max_retry=max_retry - 1)


def download_all_files():
    with open(_test_data_dir / "download_file_list.json") as f:
        datasets = json.load(f)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        for name, dataset in datasets.items():
            executor.submit(_download_file, dataset["url"], dataset["path"],
                            dataset["sha256"])
