# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def _rgbd(depth, depth_scale, depth_trunc=3.0):
    color = np.full((24, 32, 3), 128, dtype=np.uint8)
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        o3d.geometry.Image(color),
        o3d.geometry.Image(depth),
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)


def _reconstruct(rgbd, stride, voxel_length=0.05):
    intrinsic = o3d.camera.PinholeCameraIntrinsic(32, 24, 28.0, 28.0, 15.5,
                                                  11.5)
    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=2.0 * voxel_length,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
        depth_sampling_stride=stride)
    volume.integrate(rgbd, intrinsic, np.eye(4))
    return volume.extract_point_cloud(), volume.extract_triangle_mesh()


def _assert_plane(pointcloud, mesh, depth, voxel_length):
    assert len(mesh.triangles) > 0
    for points in (np.asarray(pointcloud.points), np.asarray(mesh.vertices)):
        assert len(points) > 0
        assert np.isfinite(points).all()
        # TSDF zero crossings interpolate neighboring voxel centers.
        np.testing.assert_allclose(points[:, 2],
                                   depth,
                                   rtol=0,
                                   atol=voxel_length)


@pytest.mark.parametrize("stride", [1, 4])
def test_scalable_tsdf_metric_depth(stride):
    meter_rgbd = _rgbd(np.full((24, 32), 2.0, dtype=np.float32), 1.0)
    millimeter_rgbd = _rgbd(np.full((24, 32), 2000, dtype=np.uint16), 1000.0)
    meter_points, meter_mesh = _reconstruct(meter_rgbd, stride)
    millimeter_points, millimeter_mesh = _reconstruct(millimeter_rgbd, stride)
    _assert_plane(meter_points, meter_mesh, 2.0, 0.05)
    _assert_plane(millimeter_points, millimeter_mesh, 2.0, 0.05)
    for meter, millimeter in ((meter_points.points, millimeter_points.points),
                              (meter_mesh.vertices, millimeter_mesh.vertices)):
        meter = np.asarray(meter)
        millimeter = np.asarray(millimeter)
        meter = meter[np.lexsort(meter.T[::-1])]
        millimeter = millimeter[np.lexsort(millimeter.T[::-1])]
        np.testing.assert_allclose(meter, millimeter, rtol=0, atol=1e-6)
    assert len(meter_mesh.triangles) == len(millimeter_mesh.triangles)


def test_scalable_tsdf_preserves_rgbd_truncation():
    # Scale the scene, preserving its voxel count, beyond the point-cloud
    # factory's default truncation. RGBD already applied the chosen cutoff.
    rgbd = _rgbd(np.full((24, 32), 2000.0, dtype=np.float32), 1.0, 5000.0)
    points, mesh = _reconstruct(rgbd, 4, voxel_length=50.0)
    _assert_plane(points, mesh, 2000.0, 50.0)


def test_scalable_tsdf_depth_sampling_stride():
    depth = np.full((24, 32), 2.0, dtype=np.float32)
    depth[::4, ::4] = 0.0
    rgbd = _rgbd(depth, 1.0)
    dense_points, dense_mesh = _reconstruct(rgbd, 1)
    _assert_plane(dense_points, dense_mesh, 2.0, 0.05)
    sampled_points, sampled_mesh = _reconstruct(rgbd, 4)
    assert len(sampled_points.points) == 0
    assert len(sampled_mesh.vertices) == 0
    assert len(sampled_mesh.triangles) == 0
