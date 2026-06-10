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

import sys
import os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")
from open3d_test import list_devices


@pytest.mark.parametrize("device", list_devices())
def test_pointcloud_get_oriented_bounding_ellipsoid(device):
    """GetOrientedBoundingEllipsoid on t::PointCloud returns valid ellipsoid."""
    pcd = o3d.t.geometry.PointCloud(device)
    # Use a set of points spread across three axes so the ellipsoid is
    # non-degenerate (Khachiyan requires at least 4 points in 3D).
    pts = np.array([
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 2.0, 0.0],
        [0.0, -2.0, 0.0],
        [0.0, 0.0, 3.0],
        [0.0, 0.0, -3.0],
    ],
                   dtype=np.float32)
    pcd.point["positions"] = o3c.Tensor(pts, dtype=o3c.float32, device=device)

    obe = pcd.get_oriented_bounding_ellipsoid()

    assert isinstance(obe, o3d.t.geometry.OrientedBoundingEllipsoid)
    np.testing.assert_allclose(obe.volume(), 25.13274123, atol=1e-5)
    # Center should be near the origin for this symmetric point set.
    np.testing.assert_allclose(obe.center.cpu().numpy(), 0.0, atol=1e-3)


@pytest.mark.parametrize("device", list_devices())
def test_trianglemesh_get_oriented_bounding_ellipsoid(device):
    """GetOrientedBoundingEllipsoid on t::TriangleMesh returns valid ellipsoid."""
    # Use the built-in sphere as a simple non-degenerate mesh.
    mesh = o3d.t.geometry.TriangleMesh.create_sphere(radius=1.0, device=device)

    obe = mesh.get_oriented_bounding_ellipsoid()

    assert isinstance(obe, o3d.t.geometry.OrientedBoundingEllipsoid)
    np.testing.assert_allclose(obe.volume(), 4.18879, atol=1e-5)
    # Sphere centered at origin — ellipsoid center should be near origin.
    np.testing.assert_allclose(obe.center.cpu().numpy(), 0.0, atol=1e-3)
    # All radii should be positive.
    np.testing.assert_array_less(0.0, obe.radii.cpu().numpy())
