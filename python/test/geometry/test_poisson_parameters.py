# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest


def _create_point_cloud(num_points=100):
    """Helper to create a point cloud with normals."""
    np.random.seed(42)  # Fixed seed for reproducible tests
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.random.rand(num_points, 3) - 0.5)
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(num_points, 3) - 0.5)
    pcd.normalize_normals()
    return pcd


def _assert_valid_mesh(mesh, densities):
    """Helper to validate mesh and densities output."""
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    assert len(densities) == len(mesh.vertices)


@pytest.fixture
def sample_point_cloud():
    """Fixture that returns a simple point cloud for testing."""
    return _create_point_cloud()


def test_poisson_default_parameters(sample_point_cloud):
    """Test Poisson reconstruction with default parameters."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, depth=6)
    _assert_valid_mesh(mesh, densities)


@pytest.mark.parametrize("params", [
    {
        "depth": 6,
        "full_depth": 4,
        "samples_per_node": 2.0,
        "point_weight": 5.0
    },
    {
        "depth": 6,
        "full_depth": 3
    },
    {
        "depth": 6,
        "full_depth": 5
    },
    {
        "depth": 5,
        "samples_per_node": 1.0
    },
    {
        "depth": 5,
        "samples_per_node": 3.0
    },
    {
        "depth": 5,
        "point_weight": 4.0
    },
    {
        "depth": 5,
        "point_weight": 10.0
    },
])
def test_poisson_with_various_parameters(sample_point_cloud, params):
    """Test Poisson reconstruction with various parameter combinations."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, **params)
    _assert_valid_mesh(mesh, densities)


def test_poisson_backward_compatibility():
    """Test that old API calls still work (backward compatibility)."""
    pcd = _create_point_cloud(num_points=50)

    # Old-style call without new parameters
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, scale=1.1, linear_fit=False)
    _assert_valid_mesh(mesh, densities)
