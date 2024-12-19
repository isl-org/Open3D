# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
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
from open3d_benchmark import list_tensor_sizes, list_non_bool_dtypes, list_float_dtypes, to_numpy_dtype


class UnaryEWOps:

    @staticmethod
    def sqrt(a):
        return a.sqrt()

    @staticmethod
    def sin(a):
        return a.sin()

    @staticmethod
    def cos(a):
        return a.cos()

    @staticmethod
    def neg(a):
        return a.neg()

    @staticmethod
    def exp(a):
        return a.exp()

    @staticmethod
    def abs(a):
        return a.abs()

    @staticmethod
    def isnan(a):
        return a.isnan()

    @staticmethod
    def isinf(a):
        return a.isinf()

    @staticmethod
    def isfinite(a):
        return a.isfinite()

    @staticmethod
    def floor(a):
        return a.floor()

    @staticmethod
    def ceil(a):
        return a.ceil()

    @staticmethod
    def round(a):
        return a.round()

    @staticmethod
    def trunc(a):
        return a.trunc()

    @staticmethod
    def logical_not(a):
        return a.logical_not()


def list_unary_ops():
    return [
        UnaryEWOps.neg,
        UnaryEWOps.abs,
        UnaryEWOps.isnan,
        UnaryEWOps.isinf,
        UnaryEWOps.isfinite,
        UnaryEWOps.floor,
        UnaryEWOps.ceil,
        UnaryEWOps.round,
        UnaryEWOps.trunc,
        UnaryEWOps.logical_not,
    ]


def list_float_unary_ops():
    return [
        UnaryEWOps.sqrt,
        UnaryEWOps.sin,
        UnaryEWOps.cos,
        UnaryEWOps.exp,
    ]


def to_numpy_unary_op(op):
    conversions = {
        UnaryEWOps.sqrt: np.sqrt,
        UnaryEWOps.sin: np.sin,
        UnaryEWOps.cos: np.cos,
        UnaryEWOps.neg: operator.neg,
        UnaryEWOps.exp: np.exp,
        UnaryEWOps.abs: np.abs,
        UnaryEWOps.isnan: np.isnan,
        UnaryEWOps.isinf: np.isinf,
        UnaryEWOps.isfinite: np.isfinite,
        UnaryEWOps.floor: np.floor,
        UnaryEWOps.ceil: np.ceil,
        UnaryEWOps.round: np.round,
        UnaryEWOps.trunc: np.trunc,
        UnaryEWOps.logical_not: np.logical_not,
    }
    return conversions[op]


@pytest.mark.parametrize("size", list_tensor_sizes())
@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("op", list_unary_ops())
def test_unary_ew_ops(benchmark, size, dtype, op):
    # Set upper bound to 88 to avoid overflow for exp() op.
    np_a = np.array(np.random.uniform(1, 88, size), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=o3c.Device("CPU:0"))
    benchmark(op, a)


@pytest.mark.parametrize("size", list_tensor_sizes())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("op", list_float_unary_ops())
def test_float_unary_ew_ops(benchmark, size, dtype, op):
    # Set upper bound to 88 to avoid overflow for exp() op.
    np_a = np.array(np.random.uniform(1, 88, size), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=o3c.Device("CPU:0"))
    benchmark(op, a)


@pytest.mark.parametrize("size", list_tensor_sizes())
@pytest.mark.parametrize("dtype", list_non_bool_dtypes())
@pytest.mark.parametrize("op", list_unary_ops())
def test_unary_ew_ops_numpy(benchmark, size, dtype, op):
    # Set upper bound to 88 to avoid overflow for exp() op.
    np_a = np.array(np.random.uniform(1, 88, size), dtype=to_numpy_dtype(dtype))
    benchmark(to_numpy_unary_op(op), np_a)


@pytest.mark.parametrize("size", list_tensor_sizes())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("op", list_float_unary_ops())
def test_float_unary_ew_ops_numpy(benchmark, size, dtype, op):
    # Set upper bound to 88 to avoid overflow for exp() op.
    np_a = np.array(np.random.uniform(1, 88, size), dtype=to_numpy_dtype(dtype))
    benchmark(to_numpy_unary_op(op), np_a)
