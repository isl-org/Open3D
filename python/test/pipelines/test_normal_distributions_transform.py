# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d
import pytest


def make_structured_point_cloud():
    points = []
    centers = [
        np.array([-1.2, -0.8, -0.4]),
        np.array([-0.2, 0.7, 0.3]),
        np.array([0.9, -0.1, 0.8]),
        np.array([1.5, 1.0, -0.2]),
        np.array([-1.5, 1.1, 0.9]),
    ]
    for cluster, center in enumerate(centers):
        angle = 0.35 * cluster
        c = np.cos(angle)
        s = np.sin(angle)
        rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        for i in range(160):
            a = ((i % 16) - 7.5) / 7.5
            b = (((i // 16) % 10) - 4.5) / 4.5
            h = np.sin(0.37 * i + cluster)
            offset = np.array([
                0.20 * a + 0.04 * b,
                0.13 * b + 0.03 * h,
                0.08 * h + 0.025 * a * b,
            ])
            points.append(center + rotation @ offset)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points))
    return pcd


def make_transformation():
    transformation = np.eye(4)
    angle = 0.08
    c = np.cos(angle)
    s = np.sin(angle)
    transformation[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0],
                                       [0.0, 0.0, 1.0]])
    transformation[:3, 3] = np.array([0.24, -0.17, 0.11])
    return transformation


def test_normal_distributions_transform_option():
    option = o3d.pipelines.registration.NormalDistributionsTransformOption(
        voxel_size=0.8,
        min_points_per_voxel=4,
        covariance_regularization=1e-3,
        transformation_epsilon=0.01,
        relative_objective=1e-7,
        max_iteration=40,
        outlier_threshold=9.0,
        neighbor_search_type=1)

    assert option.voxel_size == 0.8
    assert option.min_points_per_voxel == 4
    assert option.relative_objective == 1e-7
    assert option.max_iteration == 40
    assert "NormalDistributionsTransformOption" in repr(option)


def test_transformation_estimation_for_ndt_is_not_exposed():
    assert not hasattr(o3d.pipelines.registration,
                       "TransformationEstimationForNDT")


def test_registration_ndt_recovers_known_transform():
    source = make_structured_point_cloud()
    target = o3d.geometry.PointCloud(source)
    expected = make_transformation()
    target.transform(expected)

    initial = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.35)
    result = o3d.pipelines.registration.registration_ndt(
        source, target,
        o3d.pipelines.registration.NormalDistributionsTransformOption(
            voxel_size=0.8,
            min_points_per_voxel=4,
            covariance_regularization=1e-3,
            transformation_epsilon=0.01,
            relative_objective=1e-7,
            max_iteration=40,
            outlier_threshold=9.0,
            neighbor_search_type=1), np.eye(4))
    refined = o3d.pipelines.registration.evaluate_registration(
        source, target, 0.35, result.transformation)

    assert refined.fitness > initial.fitness
    assert refined.inlier_rmse < initial.inlier_rmse
    assert result.fitness > 0.80
    assert result.inlier_rmse < 0.16
    np.testing.assert_allclose(result.transformation, expected, atol=5e-2)

    source_transformed = o3d.geometry.PointCloud(source)
    source_transformed.transform(result.transformation)
    correspondences = np.asarray(result.correspondence_set)
    source_points = np.asarray(source_transformed.points)
    target_points = np.asarray(target.points)
    errors = (source_points[correspondences[:, 0]] -
              target_points[correspondences[:, 1]])
    expected_rmse = np.sqrt(np.mean(np.sum(errors * errors, axis=1)))
    np.testing.assert_allclose(result.inlier_rmse, expected_rmse)


def test_registration_ndt_rejects_icp_convergence_criteria():
    source = make_structured_point_cloud()
    target = o3d.geometry.PointCloud(source)

    with pytest.raises(TypeError):
        o3d.pipelines.registration.registration_ndt(
            source, target,
            o3d.pipelines.registration.NormalDistributionsTransformOption(),
            np.eye(4), o3d.pipelines.registration.ICPConvergenceCriteria())


def test_registration_ndt_rejects_invalid_options():
    with pytest.raises(RuntimeError):
        o3d.pipelines.registration.NormalDistributionsTransformOption(
            voxel_size=-1.0)
    with pytest.raises(RuntimeError):
        o3d.pipelines.registration.NormalDistributionsTransformOption(
            voxel_size=1.0, outlier_threshold=0.0)
    with pytest.raises(RuntimeError):
        o3d.pipelines.registration.NormalDistributionsTransformOption(
            voxel_size=1.0, neighbor_search_type=2)
    with pytest.raises(RuntimeError):
        o3d.pipelines.registration.NormalDistributionsTransformOption(
            voxel_size=1.0, relative_objective=0.0)


def test_registration_ndt_rejects_mutated_invalid_option():
    source = make_structured_point_cloud()
    target = o3d.geometry.PointCloud(source)
    option = o3d.pipelines.registration.NormalDistributionsTransformOption()
    option.voxel_size = 0.0

    with pytest.raises(RuntimeError):
        o3d.pipelines.registration.registration_ndt(source, target, option)

    option = o3d.pipelines.registration.NormalDistributionsTransformOption()
    option.relative_objective = np.nan
    with pytest.raises(RuntimeError):
        o3d.pipelines.registration.registration_ndt(source, target, option)
