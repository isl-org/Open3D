# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import sys

import numpy as np
import open3d as o3d
import open3d.core as o3c
import pytest

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


def _sycl_gpu_available():
    return len(o3c.sycl.get_available_devices()) > 1


def _integrate_redwood_frames(device,
                              num_frames: int,
                              block_resolution: int = 8):
    """Integrate the first ``num_frames`` Redwood RGB-D frames on ``device``."""
    dataset = o3d.data.SampleRedwoodRGBDImages()
    intrinsic = o3c.Tensor(
        [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
        dtype=o3c.float64,
        device=o3c.Device("CPU:0"),
    )
    trajectory = o3d.io.read_pinhole_camera_trajectory(
        dataset.odometry_log_path)
    depth_scale = 1000.0
    depth_max = 3.0
    voxel_size = 3.0 / 512

    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=["tsdf", "weight", "color"],
        attr_dtypes=[o3c.float32, o3c.float32, o3c.float32],
        attr_channels=[[1], [1], [3]],
        voxel_size=voxel_size,
        block_resolution=block_resolution,
        block_count=10000,
        device=device,
    )

    for i in range(num_frames):
        depth = o3d.t.io.read_image(dataset.depth_paths[i]).to(device)
        color = o3d.t.io.read_image(dataset.color_paths[i]).to(device)
        extrinsic = o3c.Tensor(
            trajectory.parameters[i].extrinsic,
            dtype=o3c.float64,
            device=o3c.Device("CPU:0"),
        )
        frustum = vbg.compute_unique_block_coordinates(
            depth,
            intrinsic,
            extrinsic,
            depth_scale,
            depth_max,
            trunc_voxel_multiplier=4.0,
        )
        vbg.integrate(
            frustum,
            depth,
            color,
            intrinsic,
            extrinsic,
            depth_scale,
            depth_max,
            trunc_voxel_multiplier=block_resolution * 0.5,
        )
    return vbg


@pytest.mark.parametrize("device", list_devices(enable_sycl=True))
def test_voxel_block_grid_construct(device):
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=["tsdf", "weight", "color"],
        attr_dtypes=[o3c.float32, o3c.uint16, o3c.uint8],
        attr_channels=[[1], [1], [3]],
        voxel_size=3.0 / 512,
        block_resolution=8,
        block_count=10,
        device=device,
    )
    assert vbg.hashmap().capacity() == 10


@pytest.mark.skipif(not _sycl_gpu_available(), reason="SYCL GPU required.")
def test_voxel_block_grid_sycl_integrate_extract_matches_cpu():
    """Short Redwood integration: SYCL extraction vs CPU oracle."""
    cpu = o3c.Device("CPU:0")
    sycl = o3c.Device("SYCL:0")

    vbg_cpu = _integrate_redwood_frames(cpu, num_frames=2)
    vbg_sycl = _integrate_redwood_frames(sycl, num_frames=2)

    pcd_cpu = vbg_cpu.extract_point_cloud()
    pcd_sycl = vbg_sycl.extract_point_cloud()
    assert pcd_cpu.point.positions.shape[0] == pcd_sycl.point.positions.shape[0]
    np.testing.assert_allclose(
        pcd_sycl.point.positions.cpu().numpy(),
        pcd_cpu.point.positions.cpu().numpy(),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        pcd_sycl.point.normals.cpu().numpy(),
        pcd_cpu.point.normals.cpu().numpy(),
        rtol=1e-2,
        atol=1e-2,
    )

    mesh_cpu = vbg_cpu.extract_triangle_mesh()
    mesh_sycl = vbg_sycl.extract_triangle_mesh()
    assert mesh_cpu.vertex.positions.shape[
        0] == mesh_sycl.vertex.positions.shape[0]
    assert mesh_cpu.triangle.indices.shape[
        0] == mesh_sycl.triangle.indices.shape[0]


@pytest.mark.skipif(not _sycl_gpu_available(), reason="SYCL GPU required.")
def test_voxel_block_grid_sycl_raycast_runs():
    """Smoke test: ray cast with depth/vertex/normal outputs on SYCL."""
    sycl = o3c.Device("SYCL:0")
    vbg = _integrate_redwood_frames(sycl, num_frames=2)
    dataset = o3d.data.SampleRedwoodRGBDImages()
    intrinsic = o3c.Tensor(
        [[525.0, 0.0, 319.5], [0.0, 525.0, 239.5], [0.0, 0.0, 1.0]],
        dtype=o3c.float64,
        device=o3c.Device("CPU:0"),
    )
    trajectory = o3d.io.read_pinhole_camera_trajectory(
        dataset.odometry_log_path)
    i = len(trajectory.parameters) - 1
    depth = o3d.t.io.read_image(dataset.depth_paths[i]).to(sycl)
    extrinsic = o3c.Tensor(
        trajectory.parameters[i].extrinsic,
        dtype=o3c.float64,
        device=o3c.Device("CPU:0"),
    )
    frustum = vbg.compute_unique_block_coordinates(depth, intrinsic, extrinsic,
                                                   1000.0, 3.0)
    out = vbg.ray_cast(
        frustum,
        intrinsic,
        extrinsic,
        depth.columns,
        depth.rows,
        ["vertex", "normal", "depth"],
        depth_scale=1000.0,
        depth_min=0.1,
        depth_max=3.0,
        weight_threshold=1.0,
    )
    assert "vertex" in out
    assert "normal" in out
    assert "depth" in out
