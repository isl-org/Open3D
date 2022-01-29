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

import open3d as o3d
import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from pathlib import Path


def test_dataset_base():
    default_data_root = os.path.join(Path.home(), "open3d_data")

    ds = o3d.data.Dataset("some_prefix")
    assert ds.data_root == default_data_root

    ds_custom = o3d.data.Dataset("some_prefix", "/my/custom/data_root")
    assert ds_custom.data_root == "/my/custom/data_root"
    assert ds_custom.prefix == "some_prefix"
    assert ds_custom.download_dir == "/my/custom/data_root/download/some_prefix"
    assert ds_custom.extract_dir == "/my/custom/data_root/extract/some_prefix"


# def test_simple_dataset_base():
#     prefix = "O3DTestSimpleDataset"
#     data_root = os.path.join(Path.home(), "open3d_data")
#     download_dir = os.path.join(data_root, "download", prefix)
#     extract_dir = os.path.join(data_root, "extract", prefix)

#     # delete if files already exists.
#     o3d.data.Dataset("/my")
#     dtype = o3c.int32
#     assert dtype.byte_size() == 4
#     assert "{}".format(dtype) == "Int32"

# def test_demo_icp_pointclouds():
