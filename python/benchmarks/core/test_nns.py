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

import itertools
import operator
import os
import sys

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
sys.path.append(
    os.path.dirname(os.path.realpath(__file__)) +
    "/../../../examples/python/utility")
from open3d_benchmark import list_devices, list_float_dtypes, to_numpy_dtype


class NNSOps:

    @staticmethod
    def knn_setup(datasets, nns_opt):
        index = o3c.nns.NearestNeighborSearch(datasets)
        index.knn_index()
        return index

    @staticmethod
    def radius_setup(datasets, nns_opt):
        radius = nns_opt["radius"]
        index = o3c.nns.NearestNeighborSearch(datasets)
        index.fixed_radius_index(radius)
        return index

    @staticmethod
    def hybrid_setup(datasets, nns_opt):
        radius = nns_opt["radius"]
        index = o3c.nns.NearestNeighborSearch(datasets)
        index.hybrid_index(radius)
        return index

    @staticmethod
    def knn_search(index, queries, nns_opt):
        knn = nns_opt["knn"]
        result = index.knn_search(queries, knn)
        return result

    @staticmethod
    def radius_search(index, queries, nns_opt):
        radius = nns_opt["radius"]
        result = index.fixed_radius_search(queries, radius)
        return result

    @staticmethod
    def hybrid_search(index, queries, nns_opt):
        radius, knn = nns_opt["radius"], nns_opt["knn"]
        result = index.hybrid_search(queries, radius, knn)
        return result


def list_sizes():
    num_points = (10000,)
    return num_points


def list_dimensions():
    dimensions = (3, 8, 16, 32)
    return dimensions


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("dim", list_dimensions())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_knn_setup(benchmark, size, dim, dtype, device):
    nns_opt = dict(knn=1, radius=0.01)
    np_a = np.array(np.random.rand(size, dim), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=device)
    benchmark(NNSOps.knn_setup, a, nns_opt)


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("dim", list_dimensions())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_knn_search(benchmark, size, dim, dtype, device):
    nns_opt = dict(knn=1, radius=0.01)
    np_a = np.array(np.random.rand(size, dim), dtype=to_numpy_dtype(dtype))
    np_b = np.array(np.random.rand(size, dim), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=device)
    b = o3c.Tensor(np_b, dtype=dtype, device=device)
    index = NNSOps.knn_setup(a, nns_opt)
    benchmark(NNSOps.knn_search, index, b, nns_opt)


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_radius_setup(benchmark, size, dtype, device):
    nns_opt = dict(knn=1, radius=0.01)
    np_a = np.array(np.random.rand(size, 3), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=device)
    benchmark(NNSOps.radius_setup, a, nns_opt)


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_radius_search(benchmark, size, dtype, device):
    nns_opt = dict(knn=1, radius=0.01)
    np_a = np.array(np.random.rand(size, 3), dtype=to_numpy_dtype(dtype))
    np_b = np.array(np.random.rand(size, 3), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=device)
    b = o3c.Tensor(np_b, dtype=dtype, device=device)
    index = NNSOps.radius_setup(a, nns_opt)
    benchmark(NNSOps.radius_search, index, b, nns_opt)


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_hybrid_setup(benchmark, size, dtype, device):
    nns_opt = dict(knn=1, radius=0.01)
    np_a = np.array(np.random.rand(size, 3), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=device)
    benchmark(NNSOps.hybrid_setup, a, nns_opt)


@pytest.mark.parametrize("size", list_sizes())
@pytest.mark.parametrize("dtype", list_float_dtypes())
@pytest.mark.parametrize("device", list_devices())
def test_hybrid_search(benchmark, size, dtype, device):
    nns_opt = dict(knn=1, radius=0.01)
    np_a = np.array(np.random.rand(size, 3), dtype=to_numpy_dtype(dtype))
    np_b = np.array(np.random.rand(size, 3), dtype=to_numpy_dtype(dtype))
    a = o3c.Tensor(np_a, dtype=dtype, device=device)
    b = o3c.Tensor(np_b, dtype=dtype, device=device)
    index = NNSOps.hybrid_setup(a, nns_opt)
    benchmark(NNSOps.hybrid_search, index, b, nns_opt)
