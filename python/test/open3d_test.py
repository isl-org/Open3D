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

import os
import sys
import urllib.request
import zipfile

import numpy as np
import pytest

# Avoid pathlib to be compatible with Python 3.5+.
_pwd = os.path.dirname(os.path.realpath(__file__))
test_data_dir = os.path.join(_pwd, os.pardir, os.pardir, "examples",
                             "test_data")

# Whenever you import open3d_test, the test data will be downloaded
# automatically to Open3D/examples/test_data/open3d_downloads. Therefore, make
# sure to import open3d_test or anything inside open3d_test before running
# unit tests. See https://github.com/isl-org/open3d_downloads for details on
# how to manage the test data files.
sys.path.append(test_data_dir)
from download_utils import download_all_files as _download_all_files

_download_all_files()


def torch_available():
    try:
        import torch
        import torch.utils.dlpack
    except ImportError:
        return False
    return True


def list_devices():
    """
    If Open3D is built with CUDA support:
    - If cuda device is available, returns [Device("CPU:0"), Device("CUDA:0")].
    - If cuda device is not available, returns [Device("CPU:0")].

    If Open3D is built without CUDA support:
    - returns [Device("CPU:0")].
    """
    import open3d as o3d
    if o3d.core.cuda.device_count() > 0:
        return [o3d.core.Device("CPU:0"), o3d.core.Device("CUDA:0")]
    else:
        return [o3d.core.Device("CPU:0")]


def list_devices_with_torch():
    """
    Similar to list_devices(), but take PyTorch available devices into account.
    The returned devices are compatible on both PyTorch and Open3D.

    If PyTorch is not available at all, empty list will be returned, thus the
    test is effectively skipped.
    """
    if torch_available():
        import open3d as o3d
        import torch
        if (o3d.core.cuda.device_count() > 0 and torch.cuda.is_available() and
                torch.cuda.device_count() > 0):
            return [o3d.core.Device("CPU:0"), o3d.core.Device("CUDA:0")]
        else:
            return [o3d.core.Device("CPU:0")]
    else:
        return []


def download_fountain_dataset():
    fountain_path = os.path.join(test_data_dir, "fountain_small")
    fountain_zip_path = os.path.join(test_data_dir, "fountain.zip")
    if not os.path.exists(fountain_path):
        print("Downloading fountain dataset")
        url = "https://github.com/isl-org/open3d_downloads/releases/download/open3d_tutorial/fountain.zip"
        urllib.request.urlretrieve(url, fountain_zip_path)
        print("Extracting fountain dataset")
        with zipfile.ZipFile(fountain_zip_path, "r") as zip_ref:
            zip_ref.extractall(os.path.dirname(fountain_path))
        os.remove(fountain_zip_path)
    return fountain_path
