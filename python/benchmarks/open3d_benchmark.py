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

import numpy as np
import open3d as o3d
import open3d.core as o3c


def list_tensor_sizes():
    return [
        100000000,
    ]


def list_non_bool_dtypes():
    return [
        o3c.int8,
        o3c.uint8,
        o3c.int16,
        o3c.uint16,
        o3c.int32,
        o3c.uint32,
        o3c.int64,
        o3c.uint64,
        o3c.float32,
        o3c.float64,
    ]


def list_float_dtypes():
    return [
        o3c.float32,
        o3c.float64,
    ]


def to_numpy_dtype(dtype: o3c.Dtype):
    conversions = {
        o3c.bool8: np.bool8,  # np.bool deprecated
        o3c.bool: np.bool8,  # o3c.bool is an alias for o3c.bool8
        o3c.int8: np.int8,
        o3c.uint8: np.uint8,
        o3c.int16: np.int16,
        o3c.uint16: np.uint16,
        o3c.int32: np.int32,
        o3c.uint32: np.uint32,
        o3c.int64: np.int64,
        o3c.uint64: np.uint64,
        o3c.float32: np.float32,
        o3c.float64: np.float64,
    }
    return conversions[dtype]


def list_devices():
    devices = [o3c.Device("CPU:0")]
    if o3c.cuda.is_available():
        devices.append(o3c.Device("CUDA:0"))
    return devices
