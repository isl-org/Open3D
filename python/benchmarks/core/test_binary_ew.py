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
import open3d.core as o3c
import numpy as np
import pytest
import operator

import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_benchmark import list_tensor_sizes, list_non_bool_dtypes, to_numpy_dtype


class BinaryEWOps:

    @staticmethod
    def logical_and(lhs, rhs):
        return lhs.logical_and(rhs)

    @staticmethod
    def logical_or(lhs, rhs):
        return lhs.logical_or(rhs)

    @staticmethod
    def logical_xor(lhs, rhs):
        return lhs.logical_xor(rhs)


def list_binary_ops():
    return [
        operator.add,
        operator.sub,
        operator.mul,
        operator.truediv,
        BinaryEWOps.logical_and,
        BinaryEWOps.logical_or,
        BinaryEWOps.logical_xor,
        operator.gt,
        operator.lt,
        operator.ge,
        operator.le,
        operator.eq,
        operator.ne,
    ]


def to_numpy_binary_op(op):
    conversions = {
        operator.add: operator.add,
        operator.sub: operator.sub,
        operator.mul: operator.mul,
        operator.truediv: operator.truediv,
        BinaryEWOps.logical_and: np.logical_and,
        BinaryEWOps.logical_or: np.logical_or,
        BinaryEWOps.logical_xor: np.logical_xor,
        operator.gt: operator.gt,
        operator.lt: operator.lt,
        operator.ge: operator.ge,
        operator.le: operator.le,
        operator.eq: operator.eq,
        operator.ne: operator.ne,
    }
    return conversions[op]


@pytest.mark.parametrize("size", list_tensor_sizes())
@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("op", list_binary_ops())
def test_binary_ew_ops(benchmark, size, dtype, op):
    np_a = np.array(np.random.uniform(1, 127, size),
                    dtype=to_numpy_dtype(dtype))
    np_b = np.array(np.random.uniform(1, 127, size),
                    dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=o3c.Device("CPU:0"))
    b = o3c.Tensor(np_b, dtype=dtype, device=o3c.Device("CPU:0"))
    benchmark(op, a, b)


@pytest.mark.parametrize("size", list_tensor_sizes())
@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("op", list_binary_ops())
def test_binary_ew_ops_numpy(benchmark, size, dtype, op):
    np_a = np.array(np.random.uniform(1, 127, size),
                    dtype=to_numpy_dtype(dtype))
    np_b = np.array(np.random.uniform(1, 127, size),
                    dtype=to_numpy_dtype(dtype))
    benchmark(to_numpy_binary_op(op), np_a, np_b)
