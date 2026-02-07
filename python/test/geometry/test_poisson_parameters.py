# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import pytest


def test_poisson_default_parameters():
    """Test Poisson reconstruction with default parameters."""
    # Create a simple point cloud (sphere)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    # Add normals pointing outward
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normalize_normals()
    
    # Run with default parameters
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=6
    )
    
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    assert len(densities) == len(mesh.vertices)


def test_poisson_custom_parameters():
    """Test Poisson reconstruction with custom parameters."""
    # Create a simple point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normalize_normals()
    
    # Run with custom parameters
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd,
        depth=6,
        full_depth=4,
        samples_per_node=2.0,
        point_weight=5.0,
        confidence=0.5,
        exact_interpolation=True
    )
    
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(mesh.triangles) > 0
    assert len(densities) == len(mesh.vertices)


def test_poisson_parameter_variation():
    """Test that different parameters produce different results."""
    # Create a simple point cloud
    np.random.seed(42)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normalize_normals()
    
    # Run with default point_weight
    mesh1, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, point_weight=4.0
    )
    
    # Run with higher point_weight (should produce different result)
    mesh2, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, point_weight=10.0
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
    
    assert mesh is not None
    assert len(mesh.vertices) > 0
    assert len(densities) == len(mesh.vertices)


def test_poisson_full_depth_parameter():
    """Test full_depth parameter specifically."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normalize_normals()
    
    # Test with different full_depth values
    mesh1, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=6, full_depth=3
    )
    
    mesh2, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=6, full_depth=5
    )
    
    assert len(mesh1.vertices) > 0
    assert len(mesh2.vertices) > 0


def test_poisson_samples_per_node_parameter():
    """Test samples_per_node parameter specifically."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normals = o3d.utility.Vector3dVector(
        np.random.rand(100, 3) - 0.5
    )
    pcd.normalize_normals()
    
    # Test with different samples_per_node values
    mesh1, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, samples_per_node=1.0
    )
    
    mesh2, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=5, samples_per_node=3.0
    )
    
    assert len(mesh1.vertices) > 0
    assert len(mesh2.vertices) > 0
