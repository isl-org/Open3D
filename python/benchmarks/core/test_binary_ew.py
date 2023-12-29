# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
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
