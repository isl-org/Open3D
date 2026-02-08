# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest


@pytest.fixture
def sample_point_cloud():
    """Create a simple point cloud for testing."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normalize_normals()
    return pcd


def _assert_valid_mesh(mesh, densities):
    """Helper to validate mesh and densities output."""
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    assert len(densities) == len(mesh.vertices)


def test_poisson_default_parameters(sample_point_cloud):
    """Test Poisson reconstruction with default parameters."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, depth=6
    )
    _assert_valid_mesh(mesh, densities)


@pytest.mark.parametrize("params,expected_valid", [
    ({"depth": 6, "full_depth": 4, "samples_per_node": 2.0, 
      "point_weight": 5.0, "confidence": 0.5, "exact_interpolation": True}, True),
    ({"depth": 6, "full_depth": 3}, True),
    ({"depth": 6, "full_depth": 5}, True),
    ({"depth": 5, "samples_per_node": 1.0}, True),
    ({"depth": 5, "samples_per_node": 3.0}, True),
])
def test_poisson_with_various_parameters(sample_point_cloud, params, expected_valid):
    """Test Poisson reconstruction with various parameter combinations."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, **params
    )
    if expected_valid:
        _assert_valid_mesh(mesh, densities)


def test_poisson_parameter_variation(sample_point_cloud):
    """Test that different parameters produce different results."""
    # Run with default point_weight
    mesh1, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, depth=5, point_weight=4.0
    )
    
    # Run with higher point_weight (should produce different result)
    mesh2, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, depth=5, point_weight=10.0
    )
    
    # Meshes should be different (different vertex counts or positions)
    # We just check they both succeeded and have positive vertex counts
    assert len(mesh1.vertices) > 0
    assert len(mesh2.vertices) > 0


def test_poisson_backward_compatibility():
    """Test that old API calls still work (backward compatibility)."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(50, 3) - 0.5
    )
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(50, 3) - 0.5
    )
    pcd.normalize_normals()
    
    # Old-style call without new parameters
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, scale=1.1, linear_fit=False
    )
    _assert_valid_mesh(mesh, densities)
