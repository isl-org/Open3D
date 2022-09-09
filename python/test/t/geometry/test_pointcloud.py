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
import pickle
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_constructor_and_accessors(device):
    dtype = o3c.float32

    # Constructor.
    pcd = o3d.t.geometry.PointCloud(device)
    assert "positions" not in pcd.point
    assert "colors" not in pcd.point
    assert isinstance(pcd.point, o3d.t.geometry.TensorMap)

    # Assignment.
    pcd.point.positions = o3c.Tensor.ones((0, 3), dtype, device)
    pcd.point.colors = o3c.Tensor.ones((0, 3), dtype, device)
    assert len(pcd.point.positions) == 0
    assert len(pcd.point.colors) == 0

    pcd.point.positions = o3c.Tensor.ones((1, 3), dtype, device)
    pcd.point.colors = o3c.Tensor.ones((1, 3), dtype, device)
    assert len(pcd.point.positions) == 1
    assert len(pcd.point.colors) == 1

    # Edit and access values.
    points = pcd.point.positions
    points[0] = o3c.Tensor([1, 2, 3], dtype, device)
    assert pcd.point.positions.allclose(o3c.Tensor([[1, 2, 3]], dtype, device))


@pytest.mark.parametrize("device", list_devices())
def test_from_legacy(device):
    dtype = o3c.float32

    legacy_pcd = o3d.geometry.PointCloud()
    legacy_pcd.points = o3d.utility.Vector3dVector(
        np.array([[0, 1, 2], [3, 4, 5]]))
    legacy_pcd.colors = o3d.utility.Vector3dVector(
        np.array([[6, 7, 8], [9, 10, 11]]))

    pcd = o3d.t.geometry.PointCloud.from_legacy(legacy_pcd, dtype, device)
    assert pcd.point.positions.allclose(
        o3c.Tensor([[0, 1, 2], [3, 4, 5]], dtype, device))
    assert pcd.point.colors.allclose(
        o3c.Tensor([[6, 7, 8], [9, 10, 11]], dtype, device))


@pytest.mark.parametrize("device", list_devices())
def test_to_legacy(device):
    dtype = o3c.float32

    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor([[0, 1, 2], [3, 4, 5]], dtype, device)
    pcd.point.colors = o3c.Tensor([[6, 7, 8], [9, 10, 11]], dtype, device)

    legacy_pcd = pcd.to_legacy()
    np.testing.assert_allclose(np.asarray(legacy_pcd.points),
                               np.array([[0, 1, 2], [3, 4, 5]]))
    np.testing.assert_allclose(np.asarray(legacy_pcd.colors),
                               np.array([[6, 7, 8], [9, 10, 11]]))


@pytest.mark.parametrize("device", list_devices())
def test_member_functions(device):
    dtype = o3c.float32

    # get_min_bound, get_max_bound, get_center.
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor([[1, 10, 20], [30, 2, 40], [50, 60, 3]],
                                     dtype, device)
    assert pcd.get_min_bound().allclose(o3c.Tensor([1, 2, 3], dtype, device))
    assert pcd.get_max_bound().allclose(o3c.Tensor([50, 60, 40], dtype, device))
    assert pcd.get_center().allclose(o3c.Tensor([27, 24, 21], dtype, device))

    # append.
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor.ones((2, 3), dtype, device)
    pcd.point.normals = o3c.Tensor.ones((2, 3), dtype, device)

    pcd2 = o3d.t.geometry.PointCloud(device)
    pcd2.point.positions = o3c.Tensor.ones((2, 3), dtype, device)
    pcd2.point.normals = o3c.Tensor.ones((2, 3), dtype, device)
    pcd2.point.labels = o3c.Tensor.ones((2, 3), dtype, device)

    pcd3 = o3d.t.geometry.PointCloud(device)
    pcd3 = pcd + pcd2

    assert pcd3.point.positions.allclose(o3c.Tensor.ones((4, 3), dtype, device))
    assert pcd3.point.normals.allclose(o3c.Tensor.ones((4, 3), dtype, device))

    with pytest.raises(RuntimeError) as excinfo:
        pcd3 = pcd2 + pcd
        assert 'The pointcloud is missing attribute' in str(excinfo.value)

    # transform.
    pcd = o3d.t.geometry.PointCloud(device)
    transform_t = o3c.Tensor(
        [[1, 1, 0, 1], [0, 1, 1, 1], [0, 1, 0, 1], [0, 0, 0, 1]], dtype, device)
    pcd.point.positions = o3c.Tensor([[1, 1, 1]], dtype, device)
    pcd.point.normals = o3c.Tensor([[1, 1, 1]], dtype, device)
    pcd.transform(transform_t)
    assert pcd.point.positions.allclose(o3c.Tensor([[3, 3, 2]], dtype, device))
    assert pcd.point.normals.allclose(o3c.Tensor([[2, 2, 1]], dtype, device))

    # translate.
    pcd = o3d.t.geometry.PointCloud(device)
    transloation = o3c.Tensor([10, 20, 30], dtype, device)

    pcd.point.positions = o3c.Tensor([[0, 1, 2], [6, 7, 8]], dtype, device)
    pcd.translate(transloation, True)
    assert pcd.point.positions.allclose(
        o3c.Tensor([[10, 21, 32], [16, 27, 38]], dtype, device))

    pcd.point.positions = o3c.Tensor([[0, 1, 2], [6, 7, 8]], dtype, device)
    pcd.translate(transloation, False)
    assert pcd.point.positions.allclose(
        o3c.Tensor([[7, 17, 27], [13, 23, 33]], dtype, device))

    # scale
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype,
                                     device)
    center = o3c.Tensor([1, 1, 1], dtype, device)
    pcd.scale(4, center)
    assert pcd.point.positions.allclose(
        o3c.Tensor([[-3, -3, -3], [1, 1, 1], [5, 5, 5]], dtype, device))

    # rotate.
    pcd = o3d.t.geometry.PointCloud(device)
    rotation = o3c.Tensor([[1, 1, 0], [0, 1, 1], [0, 1, 0]], dtype, device)
    center = o3c.Tensor([1, 1, 1], dtype, device)
    pcd.point.positions = o3c.Tensor([[2, 2, 2]], dtype, device)
    pcd.point.normals = o3c.Tensor([[1, 1, 1]], dtype, device)
    pcd.rotate(rotation, center)
    assert pcd.point.positions.allclose(o3c.Tensor([[3, 3, 2]], dtype, device))
    assert pcd.point.normals.allclose(o3c.Tensor([[2, 2, 1]], dtype, device))

    # voxel_down_sample
    pcd = o3d.t.geometry.PointCloud(device)
    pcd.point.positions = o3c.Tensor(
        [[0.1, 0.3, 0.9], [0.9, 0.2, 0.4], [0.3, 0.6, 0.8], [0.2, 0.4, 0.2]],
        dtype, device)

    pcd_small_down = pcd.voxel_down_sample(1)
    assert pcd_small_down.point.positions.allclose(
        o3c.Tensor([[0, 0, 0]], dtype, device))


def test_extrude_rotation():
    pcd = o3d.t.geometry.PointCloud([[1.0, 0, 0]])
    ans = pcd.extrude_rotation(3 * 360, [0, 1, 0],
                               resolution=3 * 16,
                               translation=2)
    assert ans.point.positions.shape == (49, 3)
    assert ans.line.indices.shape == (48, 2)


def test_extrude_linear():
    pcd = o3d.t.geometry.PointCloud([[1.0, 0, 0]])
    ans = pcd.extrude_linear([0, 0, 1])
    assert ans.point.positions.shape == (2, 3)
    assert ans.line.indices.shape == (1, 2)


@pytest.mark.parametrize("device", list_devices())
def test_pickle(device):
    pcd = o3d.t.geometry.PointCloud(device)
    with tempfile.TemporaryDirectory() as temp_dir:
        file_name = f"{temp_dir}/pcd.pkl"
        pcd.point.positions = o3c.Tensor.ones((10, 3),
                                              o3c.float32,
                                              device=device)
        pickle.dump(pcd, open(file_name, "wb"))
        pcd_load = pickle.load(open(file_name, "rb"))
        assert pcd_load.point.positions.device == device and pcd_load.point.positions.dtype == o3c.float32
        np.testing.assert_equal(pcd.point.positions.cpu().numpy(),
                                pcd_load.point.positions.cpu().numpy())
