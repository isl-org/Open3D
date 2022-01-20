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

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")
from open3d_test import list_devices

np.random.seed(0)


@pytest.mark.parametrize("device", list_devices())
def test_knn_index(device):
    dtype = o3c.float32

    t = o3c.Tensor.zeros((10, 3), dtype, device=device)
    nns = o3c.nns.NearestNeighborSearch(t)
    assert nns.knn_index()
    assert nns.fixed_radius_index(0.1)
    assert nns.hybrid_index(0.1)

    # Multi radii search is only supported on CPU.
    if device.get_type() == o3c.Device.DeviceType.CPU:
        assert nns.multi_radius_index()


@pytest.mark.parametrize("device", list_devices())
def test_knn_search(device):
    dtype = o3c.float32

    dataset_points = o3c.Tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=dtype,
        device=device)
    nns = o3c.nns.NearestNeighborSearch(dataset_points)
    nns.knn_index()

    # Single query point.
    query_points = o3c.Tensor([[0.064705, 0.043921, 0.087843]],
                              dtype=dtype,
                              device=device)
    indices, distances = nns.knn_search(query_points, 3)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([[1, 4, 9]], dtype=np.int64))
    np.testing.assert_allclose(distances.cpu().numpy(),
                               np.array([[0.00626358, 0.00747938, 0.0108912]],
                                        dtype=np.float64),
                               rtol=1e-5,
                               atol=0)

    # Multiple query points.
    query_points = o3c.Tensor(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843]],
        dtype=dtype,
        device=device)
    indices, distances = nns.knn_search(query_points, 3)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([[1, 4, 9], [1, 4, 9]], dtype=np.int64))
    np.testing.assert_allclose(distances.cpu().numpy(),
                               np.array([[0.00626358, 0.00747938, 0.0108912],
                                         [0.00626358, 0.00747938, 0.0108912]],
                                        dtype=np.float64),
                               rtol=1e-5,
                               atol=0)


@pytest.mark.parametrize("device", list_devices())
@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_fixed_radius_search(device, dtype):
    dataset_points = o3c.Tensor(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.1], [0.0, 0.0, 0.2], [0.0, 0.1, 0.0],
         [0.0, 0.1, 0.1], [0.0, 0.1, 0.2], [0.0, 0.2, 0.0], [0.0, 0.2, 0.1],
         [0.0, 0.2, 0.2], [0.1, 0.0, 0.0]],
        dtype=dtype,
        device=device)
    nns = o3c.nns.NearestNeighborSearch(dataset_points)
    nns.fixed_radius_index(0.1)

    # Single query point.
    query_points = o3c.Tensor([[0.064705, 0.043921, 0.087843]],
                              dtype=dtype,
                              device=device)
    indices, distances, neighbors_row_splits = nns.fixed_radius_search(
        query_points, 0.1)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([1, 4], dtype=np.int64))
    np.testing.assert_allclose(distances.cpu().numpy(),
                               np.array([0.00626358, 0.00747938],
                                        dtype=np.float64),
                               rtol=1e-5,
                               atol=0)
    np.testing.assert_equal(neighbors_row_splits.cpu().numpy(),
                            np.array([0, 2], dtype=np.int64))

    # Multiple query points.
    query_points = o3c.Tensor(
        [[0.064705, 0.043921, 0.087843], [0.064705, 0.043921, 0.087843]],
        dtype=dtype,
        device=device)
    indices, distances, neighbors_row_splits = nns.fixed_radius_search(
        query_points, 0.1)
    np.testing.assert_equal(indices.cpu().numpy(),
                            np.array([1, 4, 1, 4], dtype=np.int64))
    np.testing.assert_allclose(
        distances.cpu().numpy(),
        np.array([0.00626358, 0.00747938, 0.00626358, 0.00747938],
                 dtype=np.float64),
        rtol=1e-5,
        atol=0)
    np.testing.assert_equal(neighbors_row_splits.cpu().numpy(),
                            np.array([0, 2, 4], dtype=np.int64))


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_hybrid_search_random(dtype):
    if o3c.cuda.device_count() > 0:
        dataset_size, query_size = 1000, 100
        radius, k = 0.1, 10

        dataset_np = np.random.rand(dataset_size, 3)

        dataset_points = o3c.Tensor(dataset_np, dtype=dtype)
        dataset_points_cuda = dataset_points.cuda()

        nns = o3c.nns.NearestNeighborSearch(dataset_points)
        nns_cuda = o3c.nns.NearestNeighborSearch(dataset_points_cuda)

        for _ in range(10):
            query_np = np.random.rand(query_size, 3)
            query_points = o3c.Tensor(query_np, dtype=dtype)
            query_points_cuda = query_points.cuda()

            nns.hybrid_index(radius)
            indices, distances, counts = nns.hybrid_search(
                query_points, radius, k)

            nns_cuda.hybrid_index(radius)
            indices_cuda, distances_cuda, counts_cuda = nns_cuda.hybrid_search(
                query_points_cuda, radius, k)

            np.testing.assert_allclose(distances.numpy(),
                                       distances_cuda.cpu().numpy(),
                                       rtol=1e-5,
                                       atol=0)
            np.testing.assert_equal(indices.numpy(), indices_cuda.cpu().numpy())
            np.testing.assert_equal(counts.numpy(), counts_cuda.cpu().numpy())


@pytest.mark.parametrize("dtype", [o3c.float32, o3c.float64])
def test_fixed_radius_search_random(dtype):
    if o3c.cuda.device_count() > 0:
        dataset_size, query_size = 1000, 100
        radius = 0.1

        dataset_np = np.random.rand(dataset_size, 3)

        dataset_points = o3c.Tensor(dataset_np, dtype=dtype)
        dataset_points_cuda = dataset_points.cuda()

        nns = o3c.nns.NearestNeighborSearch(dataset_points)
        nns_cuda = o3c.nns.NearestNeighborSearch(dataset_points_cuda)

        for _ in range(10):
            query_np = np.random.rand(query_size, 3)
            query_points = o3c.Tensor(query_np, dtype=dtype)
            query_points_cuda = query_points.cuda()

            nns.fixed_radius_index(radius)
            indices, distances, neighbors_row_splits = nns.fixed_radius_search(
                query_points, radius)

            nns_cuda.fixed_radius_index(radius)
            indices_cuda, distances_cuda, neighbors_row_splits_cuda = nns_cuda.fixed_radius_search(
                query_points_cuda, radius)

            np.testing.assert_equal(neighbors_row_splits.numpy(),
                                    neighbors_row_splits_cuda.cpu().numpy())
            np.testing.assert_allclose(distances.numpy(),
                                       distances_cuda.cpu().numpy(),
                                       rtol=1e-5,
                                       atol=0)
            np.testing.assert_equal(indices.numpy(), indices_cuda.cpu().numpy())
