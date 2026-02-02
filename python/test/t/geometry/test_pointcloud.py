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
import pickle
import tempfile

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
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


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
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


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
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
        o3c.Tensor([[0.375, 0.375, 0.575]], dtype, device))


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


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
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


def test_metrics():

    from open3d.t.geometry import TriangleMesh, PointCloud, Metric, MetricParameters
    # box is a cube with one vertex at the origin and a side length 1
    pos = TriangleMesh.create_box().vertex.positions
    pcd1 = PointCloud(pos.clone())
    pcd2 = PointCloud(pos * 1.1)

    # (1, 3, 3, 1) vertices are shifted by (0, 0.1, 0.1*sqrt(2), 0.1*sqrt(3))
    # respectively
    metric_params = MetricParameters(fscore_radius=(0.01, 0.11, 0.15, 0.18))
    metrics = pcd1.compute_metrics(
        pcd2, (Metric.ChamferDistance, Metric.HausdorffDistance, Metric.FScore),
        metric_params)

    np.testing.assert_allclose(
        metrics.cpu().numpy(),
        (0.22436734, np.sqrt(3) / 10, 100. / 8, 400. / 8, 700. / 8, 100.),
        rtol=1e-6)


@pytest.mark.parametrize("device", list_devices())
def test_project_to_depth_image(device):
    """Project point cloud to depth image; check shape and non-empty depth."""
    dtype = o3c.float32
    width, height = 8, 8
    # Points in front of camera (z > 0): (0, 0, 1) and (0.1, 0.1, 1) project
    # with intrinsics fx=fy=10, cx=cy=4 to pixel (4,4) and (5,5) approx.
    positions = o3c.Tensor([[0.0, 0.0, 1.0], [0.1, 0.1, 1.0]], dtype, device)
    pcd = o3d.t.geometry.PointCloud(positions)
    intrinsics = o3c.Tensor([[10.0, 0, 4.0], [0, 10.0, 4.0], [0, 0, 1.0]],
                            o3c.float64)
    extrinsics = o3c.Tensor(np.eye(4), o3c.float64)

    depth_img = pcd.project_to_depth_image(
        width, height, intrinsics, extrinsics,
        depth_scale=1.0, depth_max=10.0)

    depth_tensor = depth_img.as_tensor()
    assert depth_tensor.shape == (height, width, 1)
    depth_np = depth_tensor.cpu().numpy()
    assert depth_np.shape == (height, width, 1)
    # At least one pixel should have depth (points project into image)
    assert np.any(depth_np > 0)


@pytest.mark.parametrize("device", list_devices())
def test_project_to_rgbd_image(device):
    """Project colored point cloud to RGBD image; check shapes and content."""
    dtype = o3c.float32
    width, height = 8, 8
    positions = o3c.Tensor(
        [[0.0, 0.0, 1.0], [0.1, 0.0, 1.0], [0.0, 0.1, 1.0]], dtype, device)
    colors = o3c.Tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], dtype, device)
    pcd = o3d.t.geometry.PointCloud(positions)
    pcd.point.colors = colors
    pcd = pcd.to(device)
    intrinsics = o3c.Tensor([[10.0, 0, 4.0], [0, 10.0, 4.0], [0, 0, 1.0]],
                            o3c.float64)
    extrinsics = o3c.Tensor(np.eye(4), o3c.float64)

    rgbd = pcd.project_to_rgbd_image(
        width, height, intrinsics, extrinsics,
        depth_scale=1.0, depth_max=10.0)

    assert rgbd.depth.as_tensor().shape == (height, width, 1)
    assert rgbd.color.as_tensor().shape == (height, width, 3)
    depth_np = rgbd.depth.as_tensor().cpu().numpy()
    color_np = rgbd.color.as_tensor().cpu().numpy()
    assert np.any(depth_np > 0), "depth should have at least one hit"
    # Where depth > 0, color should not be all zeros (no black-artifact pixels)
    hit_mask = (depth_np.squeeze(-1) > 0).astype(bool)
    hit_colors = color_np[hit_mask]
    assert hit_colors.size > 0
    assert np.any(hit_colors > 0), "projected pixels should have non-zero color"


@pytest.mark.parametrize("device", list_devices())
def test_project_to_rgbd_image_cpu_cuda_consistent(device):
    """When both CPU and CUDA are available, RGBD projection should match."""
    if o3c.cuda.device_count() == 0:
        pytest.skip("CUDA not available")
    dtype = o3c.float32
    width, height = 16, 16
    np.random.seed(42)
    n = 50
    positions_np = np.random.randn(n, 3).astype(np.float32) * 0.2
    positions_np[:, 2] = 1.0 + np.abs(positions_np[:, 2])  # z in [1, ~2]
    colors_np = np.random.rand(n, 3).astype(np.float32)
    intrinsics = o3c.Tensor([[20.0, 0, 8.0], [0, 20.0, 8.0], [0, 0, 1.0]],
                            o3c.float64)
    extrinsics = o3c.Tensor(np.eye(4), o3c.float64)

    pcd_cpu = o3d.t.geometry.PointCloud(o3c.Tensor(positions_np))
    pcd_cpu.point.colors = o3c.Tensor(colors_np)
    rgbd_cpu = pcd_cpu.project_to_rgbd_image(
        width, height, intrinsics, extrinsics,
        depth_scale=1.0, depth_max=5.0)

    pcd_cuda = pcd_cpu.to(o3c.Device("CUDA:0"))
    rgbd_cuda = pcd_cuda.project_to_rgbd_image(
        width, height, intrinsics, extrinsics,
        depth_scale=1.0, depth_max=5.0)

    np.testing.assert_allclose(
        rgbd_cpu.depth.as_tensor().cpu().numpy(),
        rgbd_cuda.depth.as_tensor().cpu().numpy(),
        rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(
        rgbd_cpu.color.as_tensor().cpu().numpy(),
        rgbd_cuda.color.as_tensor().cpu().numpy(),
        rtol=1e-5, atol=1e-5)
