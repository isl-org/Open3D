# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np

import open3d as o3d


def test_gaussian_splat_spz_round_trip(tmp_path):
    """SPZ preserves the Gaussian-splat schema through quantized I/O."""
    pointcloud = o3d.t.geometry.PointCloud()
    pointcloud.point.positions = o3d.core.Tensor(
        np.array([[0.0, 0.0, 1.0], [1.0, -0.5, 2.0]], dtype=np.float32))
    pointcloud.point["scale"] = o3d.core.Tensor(
        np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32))
    pointcloud.point["rot"] = o3d.core.Tensor(
        np.array([[1.0, 0.0, 0.0, 0.0], [0.9239, 0.0, 0.3827, 0.0]],
                 dtype=np.float32))
    pointcloud.point["opacity"] = o3d.core.Tensor(
        np.array([[0.0], [-1.0]], dtype=np.float32))
    pointcloud.point["f_dc"] = o3d.core.Tensor(
        np.array([[0.1, 0.2, 0.3], [-0.1, 0.0, 0.2]], dtype=np.float32))
    pointcloud.point["f_rest"] = o3d.core.Tensor(
        np.zeros((2, 3, 3), dtype=np.float32))

    filename = tmp_path / "round_trip.spz"
    assert o3d.t.io.write_point_cloud(str(filename), pointcloud)
    loaded = o3d.t.io.read_point_cloud(str(filename))

    assert loaded.point.positions.shape == (2, 3)
    assert loaded.point["f_rest"].shape == (2, 3, 3)
    np.testing.assert_allclose(loaded.point.positions.numpy(),
                               pointcloud.point.positions.numpy(),
                               rtol=0.1,
                               atol=0.1)
    np.testing.assert_allclose(loaded.point["scale"].numpy(),
                               pointcloud.point["scale"].numpy(),
                               rtol=0.1,
                               atol=0.1)
    original_rot = pointcloud.point["rot"].numpy()
    loaded_rot = loaded.point["rot"].numpy()
    dots = np.sum(original_rot * loaded_rot, axis=1)
    original_norms = np.linalg.norm(original_rot, axis=1)
    loaded_norms = np.linalg.norm(loaded_rot, axis=1)
    np.testing.assert_allclose(np.abs(dots) / (original_norms * loaded_norms),
                               np.ones(2),
                               rtol=0.02,
                               atol=0.02)


def test_gaussian_splat_spz_antialias_write(tmp_path):
    """gaussian_splat_antialias sets the SPZ header flag (read logs a hint)."""
    pointcloud = o3d.t.geometry.PointCloud()
    pointcloud.point.positions = o3d.core.Tensor(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32))
    pointcloud.point["scale"] = o3d.core.Tensor(
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32))
    pointcloud.point["rot"] = o3d.core.Tensor(
        np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32))
    pointcloud.point["opacity"] = o3d.core.Tensor(
        np.array([[0.0]], dtype=np.float32))
    pointcloud.point["f_dc"] = o3d.core.Tensor(
        np.array([[0.1, 0.2, 0.3]], dtype=np.float32))

    filename = tmp_path / "antialias.spz"
    assert o3d.t.io.write_point_cloud(str(filename),
                                      pointcloud,
                                      gaussian_splat_antialias=True)
    loaded = o3d.t.io.read_point_cloud(str(filename))
    assert loaded.point.positions.shape == (1, 3)
