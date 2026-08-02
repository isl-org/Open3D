# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import pytest


def _create_point_cloud():
    """Point cloud with reliable normals for Poisson reconstruction."""
    dataset = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(dataset.paths[0])
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1,
                                                            max_nn=30))
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
def test_poisson_various_parameters(sample_point_cloud, params):
    """Smoke test: verify Poisson reconstruction succeeds with various
    parameter combinations without crashing or producing empty output."""
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        sample_point_cloud, **params)
    _assert_valid_mesh(mesh, densities)


def test_poisson_backward_compatibility():
    """Test that old API calls still work (backward compatibility)."""
    pcd = _create_point_cloud()

    # Old-style call without new parameters
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, scale=1.1, linear_fit=False)
    _assert_valid_mesh(mesh, densities)
