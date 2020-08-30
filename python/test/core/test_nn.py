# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
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
import numpy as np
import pytest


@pytest.mark.parametrize("device", [o3d.core.Device("CPU:0")])
def test_nn_index(device):
    t = o3d.core.Tensor.zeros((2, 3), o3d.core.Dtype.Float64, device=device)
    nn = o3d.core.nns.NearestNeighborSearch(t)
    assert nn.knn_index() == True
    assert nn.multi_radius_index() == True
    assert nn.fixed_radius_index() == True
    assert nn.hybrid_index() == True


@pytest.mark.parametrize("device", [["CPU", 0]])
def test_knn_search_single(device):
    np.random.seed(0)
    data = np.random.rand(100, 3)
    query = np.array([0.5, 0.5, 0.5])
    query.shape = (1, 3)
    data_t = o3d.core.Tensor(data)
    query_t = o3d.core.Tensor(query)

    if device[0] == "CUDA":
        data_t = data_t.cuda(device[1])
        query_t = query_t.cuda(device[1])

    nn = o3d.core.nns.NearestNeighborSearch(data_t)
    nn.knn_index()
    idx, dist = nn.knn_search(query_t, 3)

    expect_idx = np.array([35, 1, 96])
    expect_idx.shape = (1, 3)
    expect_dist = np.array([0.019492, 0.0291282, 0.0367294])
    expect_dist.shape = (1, 3)
    np.testing.assert_almost_equal(idx.numpy(), expect_idx, 7)
    np.testing.assert_almost_equal(dist.numpy(), expect_dist, 7)


@pytest.mark.parametrize("device", [["CPU", 0]])
def test_knn_search_multiple(device):
    np.random.seed(0)
    data = np.random.rand(100, 3)
    query = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    data_t = o3d.core.Tensor(data)
    query_t = o3d.core.Tensor(query)

    if device[0] == "CUDA":
        data_t = data_t.cuda(device[1])
        query_t = query_t.cuda(device[1])

    nn = o3d.core.nns.NearestNeighborSearch(data_t)
    nn.knn_index()
    idx, dist = nn.knn_search(query_t, 3)

    expect_idx = np.array([[35, 1, 96], [35, 45, 0]])
    expect_dist = np.array([[0.019492, 0.0291282, 0.0367294],
                            [0.00140176, 0.00357283, 0.0158963]])
    np.testing.assert_almost_equal(idx.numpy(), expect_idx, 7)
    np.testing.assert_almost_equal(dist.numpy(), expect_dist, 7)


@pytest.mark.parametrize("device", [["CPU", 0]])
def test_fixed_radius_search_single(device):
    np.random.seed(0)
    data = np.random.rand(100, 3)
    query = np.array([[0.5, 0.5, 0.5]])
    data_t = o3d.core.Tensor(data)
    query_t = o3d.core.Tensor(query)

    if device[0] == "CUDA":
        data_t = data_t.cuda(device[1])
        query_t = query_t.cuda(device[1])

    nn = o3d.core.nns.NearestNeighborSearch(data_t)
    nn.fixed_radius_index()
    idx, dist, lims = nn.fixed_radius_search(query_t, 0.2)

    expect_idx = np.array([35, 1, 96, 45, 41])
    expect_dist = np.array(
        [0.019492, 0.0291282, 0.0367294, 0.0372526, 0.0378507])
    expect_lims = np.array([5])
    np.testing.assert_almost_equal(idx.numpy(), expect_idx, 7)
    np.testing.assert_almost_equal(dist.numpy(), expect_dist, 7)
    np.testing.assert_almost_equal(lims.numpy(), expect_lims, 7)


@pytest.mark.parametrize("device", [["CPU", 0]])
def test_fixed_radius_search_multiple(device):
    np.random.seed(0)
    data = np.random.rand(100, 3)
    query = np.array([[0.5, 0.5, 0.5], [0.6, 0.6, 0.6]])
    data_t = o3d.core.Tensor(data)
    query_t = o3d.core.Tensor(query)

    if device[0] == "CUDA":
        data_t = data_t.cuda(device[1])
        query_t = query_t.cuda(device[1])

    nn = o3d.core.nns.NearestNeighborSearch(data_t)
    nn.fixed_radius_index()
    idx, dist, lims = nn.fixed_radius_search(query_t, 0.2)

    expect_idx = np.array([35, 1, 96, 45, 41, 35, 45, 0, 62, 41, 1])
    expect_dist = np.array([
        0.019492, 0.0291282, 0.0367294, 0.0372526, 0.0378507, 0.00140176,
        0.00357283, 0.0158963, 0.0211767, 0.0330031, 0.0362418
    ])
    expect_lims = np.array([5, 6])
    np.testing.assert_almost_equal(idx.numpy(), expect_idx, 7)
    np.testing.assert_almost_equal(dist.numpy(), expect_dist, 7)
    np.testing.assert_almost_equal(lims.numpy(), expect_lims, 7)
