# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import copy

import numpy as np
import open3d as o3d


def _make_box_clouds():
    points = []
    normals = []

    def add(point, normal):
        points.append(point)
        normals.append(normal)

    for y in [0.0, 1.0]:
        for z in [0.0, 1.0]:
            add([0.0, y, z], [-1.0, 0.0, 0.0])
            add([1.0, y, z], [1.0, 0.0, 0.0])
    for x in [0.0, 1.0]:
        for z in [0.0, 1.0]:
            add([x, 0.0, z], [0.0, -1.0, 0.0])
            add([x, 1.0, z], [0.0, 1.0, 0.0])
    for x in [0.0, 1.0]:
        for y in [0.0, 1.0]:
            add([x, y, 0.0], [0.0, 0.0, -1.0])
            add([x, y, 1.0], [0.0, 0.0, 1.0])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(points))
    target.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    source_to_target = np.eye(4)
    source_to_target[:3, 3] = [0.03, -0.02, 0.04]
    source = copy.deepcopy(target)
    source.transform(np.linalg.inv(source_to_target))
    return source, target


def _make_asymmetric_box_clouds():
    points = []
    normals = []
    xs = [0.0, 0.23, 0.51, 0.88, 1.30]
    ys = [0.0, 0.17, 0.39, 0.64, 0.80]
    zs = [0.0, 0.19, 0.46, 0.73, 1.10]

    def add(point, normal):
        points.append(point)
        normals.append(normal)

    for y in ys:
        for z in zs:
            add([0.0, y, z], [-1.0, 0.0, 0.0])
            add([1.30, y, z], [1.0, 0.0, 0.0])
    for x in xs:
        for z in zs:
            add([x, 0.0, z], [0.0, -1.0, 0.0])
            add([x, 0.80, z], [0.0, 1.0, 0.0])
    for x in xs:
        for y in ys:
            add([x, y, 0.0], [0.0, 0.0, -1.0])
            add([x, y, 1.10], [0.0, 0.0, 1.0])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(points))
    target.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    source_to_target = np.eye(4)
    source_to_target[:3, :3] = o3d.geometry.get_rotation_matrix_from_xyz(
        [0.02, -0.015, 0.01])
    source_to_target[:3, 3] = [0.035, -0.025, 0.03]
    source = copy.deepcopy(target)
    source.transform(np.linalg.inv(source_to_target))
    return source, target, source_to_target


def _make_cylinder_side_clouds():
    points = []
    normals = []
    for theta in np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False):
        normal = [np.cos(theta), np.sin(theta), 0.0]
        for z in np.linspace(-0.6, 0.6, 7):
            points.append([normal[0], normal[1], z])
            normals.append(normal)

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(points))
    target.normals = o3d.utility.Vector3dVector(np.asarray(normals))

    source = copy.deepcopy(target)
    source.translate([-0.12, 0.05, -0.35])
    return source, target


def _make_offset_plane_clouds():
    points = []
    for x in range(5):
        for y in range(5):
            points.append([0.2 * x + 0.5, 0.2 * y + 0.5, 1.0])

    target = o3d.geometry.PointCloud()
    target.points = o3d.utility.Vector3dVector(np.asarray(points))

    source = copy.deepcopy(target)
    source.translate([0.0, 0.0, -0.1])
    return source, target


def _identity_correspondences(size):
    return o3d.utility.Vector2iVector(
        np.asarray([[i, i] for i in range(size)], dtype=np.int32))


def test_dcreg_option_constructor_and_repr():
    option = o3d.pipelines.registration.DCRegOption()
    assert option.degeneracy_condition_threshold == 10.0
    assert option.kappa_target == 10.0
    assert option.pcg_tolerance == 1e-6
    assert option.pcg_max_iteration == 10
    assert option.local_plane_knn == 5
    assert option.local_plane_max_thickness == 0.2
    assert option.local_plane_weight_slope == 0.9
    assert option.local_plane_min_weight == 0.1
    assert option.local_plane_use_weight_derivative
    assert option.local_frame_convergence_rotation == 1e-5
    assert option.local_frame_convergence_translation == 1e-3

    custom = o3d.pipelines.registration.DCRegOption(20.0, 15.0, 1e-8, 6, 7,
                                                    0.15, 0.7, 0.05, False,
                                                    1e-4, 2e-3)
    assert "DCRegOption" in repr(custom)
    assert custom.degeneracy_condition_threshold == 20.0
    assert custom.kappa_target == 15.0
    assert custom.pcg_tolerance == 1e-8
    assert custom.pcg_max_iteration == 6
    assert custom.local_plane_knn == 7
    assert custom.local_plane_max_thickness == 0.15
    assert custom.local_plane_weight_slope == 0.7
    assert custom.local_plane_min_weight == 0.05
    assert not custom.local_plane_use_weight_derivative
    assert custom.local_frame_convergence_rotation == 1e-4
    assert custom.local_frame_convergence_translation == 2e-3


def test_dcreg_compute_transformation_matches_point_to_plane():
    source, target = _make_box_clouds()
    corres = _identity_correspondences(len(source.points))

    baseline = o3d.pipelines.registration.TransformationEstimationPointToPlane(
    ).compute_transformation(source, target, corres)
    dcreg = o3d.pipelines.registration.TransformationEstimationPointToPlaneDCReg(
    ).compute_transformation(source, target, corres)

    assert "TransformationEstimationPointToPlaneDCReg" in repr(
        o3d.pipelines.registration.TransformationEstimationPointToPlaneDCReg())
    np.testing.assert_allclose(dcreg, baseline, rtol=1e-8, atol=1e-8)


def test_dcreg_degeneracy_analysis_on_normal_geometry():
    source, target, _ = _make_asymmetric_box_clouds()
    corres = _identity_correspondences(len(source.points))

    analysis = o3d.pipelines.registration.compute_dcreg_degeneracy_analysis(
        source, target, corres)

    assert "DCRegDegeneracyAnalysis" in repr(analysis)
    assert analysis.has_correspondence
    assert analysis.has_target_normals
    assert not analysis.is_rank_deficient
    assert analysis.schur_factorization_ok
    assert np.isfinite(analysis.condition_number_full)
    assert np.isfinite(analysis.condition_number_rotation)
    assert np.isfinite(analysis.condition_number_translation)
    assert np.isfinite(analysis.schur_eigenvalues_rotation).all()
    assert np.isfinite(analysis.schur_eigenvalues_translation).all()
    assert "target/world" in analysis.coordinate_frame
    assert "left-multiplied SE(3)" in analysis.coordinate_frame
    assert "target/world" in analysis.degeneracy_description
    assert analysis.solver_type != "invalid"


def test_dcreg_cylinder_degeneracy_is_stable():
    source, target = _make_cylinder_side_clouds()
    corres = _identity_correspondences(len(source.points))

    estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
    )
    initial_rmse = estimation.compute_rmse(source, target, corres)
    dcreg = o3d.pipelines.registration.TransformationEstimationPointToPlaneDCReg(
    ).compute_transformation(source, target, corres)
    dcreg_aligned = copy.deepcopy(source)
    dcreg_aligned.transform(dcreg)

    assert np.isfinite(dcreg).all()
    assert initial_rmse > 0.05
    np.testing.assert_allclose(estimation.compute_rmse(dcreg_aligned, target,
                                                       corres),
                               0.0,
                               atol=1e-8)
    np.testing.assert_allclose(dcreg[:2, 3], [0.12, -0.05], atol=1e-8)
    np.testing.assert_allclose(dcreg[2, 3], 0.0, atol=1e-8)
    np.testing.assert_allclose(dcreg[:3, :3], np.eye(3), atol=1e-8)


def test_dcreg_degeneracy_analysis_on_cylinder():
    source, target = _make_cylinder_side_clouds()
    corres = _identity_correspondences(len(source.points))

    analysis = o3d.pipelines.registration.compute_dcreg_degeneracy_analysis(
        source, target, corres)

    assert analysis.has_correspondence
    assert analysis.has_target_normals
    assert analysis.is_rank_deficient
    assert analysis.is_degenerate
    assert not analysis.schur_factorization_ok
    assert np.isfinite(analysis.schur_eigenvalues_rotation).all()
    assert np.isfinite(analysis.schur_eigenvalues_translation).all()
    assert np.isfinite(analysis.axis_aligned_eigenvalues_rotation).all()
    assert np.isfinite(analysis.axis_aligned_eigenvalues_translation).all()
    assert np.isfinite(analysis.condition_number_full)
    assert np.isfinite(analysis.condition_number_rotation)
    assert np.isfinite(analysis.condition_number_translation)
    np.testing.assert_array_equal(analysis.weak_rotation_axes, [0, 0, 1])
    np.testing.assert_array_equal(analysis.weak_translation_axes, [0, 0, 1])
    np.testing.assert_allclose(analysis.axis_aligned_eigenvalues_translation[2],
                               0.0,
                               atol=1e-10)
    assert analysis.weak_rotation_axes_description == "z"
    assert analysis.weak_translation_axes_description == "z"
    assert "target/world" in analysis.degeneracy_description
    assert "left-multiplied SE(3)" in analysis.coordinate_frame
    assert analysis.solver_type == "minimum_norm"


def test_dcreg_local_registration_uses_local_plane_without_normals():
    source, target = _make_offset_plane_clouds()
    option = o3d.pipelines.registration.DCRegOption(
        local_frame_convergence_rotation=1e-8,
        local_frame_convergence_translation=1e-8)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-9, relative_rmse=1e-9, max_iteration=10)

    result = o3d.pipelines.registration.registration_icp_dcreg_local(
        source, target, 0.5, np.eye(4), option, criteria)

    assert np.isfinite(result.transformation).all()
    np.testing.assert_allclose(result.transformation[:3, 3], [0.0, 0.0, 0.1],
                               atol=1e-8)
    np.testing.assert_allclose(result.fitness, 1.0, atol=1e-12)
    np.testing.assert_allclose(result.inlier_rmse, 0.0, atol=1e-12)


def test_dcreg_local_registration_then_prints_degeneracy_description(capsys):
    source, target = _make_offset_plane_clouds()
    option = o3d.pipelines.registration.DCRegOption(
        local_frame_convergence_rotation=1e-8,
        local_frame_convergence_translation=1e-8)
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-9, relative_rmse=1e-9, max_iteration=10)

    result = o3d.pipelines.registration.registration_icp_dcreg_local(
        source, target, 0.5, np.eye(4), option, criteria)
    analysis = o3d.pipelines.registration.compute_dcreg_local_degeneracy_analysis(
        source, target, 0.5, result.transformation, option)

    print("DCReg degeneracy description:")
    print(analysis.degeneracy_description)
    print("weak rotation axes:", analysis.weak_rotation_axes_description)
    print("weak translation axes:", analysis.weak_translation_axes_description)
    print("condition number rotation:", analysis.condition_number_rotation)
    print("condition number translation:",
          analysis.condition_number_translation)
    print("coordinate frame:", analysis.coordinate_frame)

    captured = capsys.readouterr()
    assert "DCReg degeneracy description" in captured.out
    assert "local-plane" in captured.out
    assert "local body frame" in captured.out
    assert "condition number translation" in captured.out


def test_dcreg_local_degeneracy_analysis_uses_local_frame():
    source, target = _make_offset_plane_clouds()

    analysis = o3d.pipelines.registration.compute_dcreg_local_degeneracy_analysis(
        source, target, 0.5, np.eye(4))

    assert analysis.has_correspondence
    assert analysis.has_target_normals
    assert analysis.is_degenerate
    assert analysis.weak_rotation_axes_description == "x, y, z"
    assert analysis.weak_translation_axes_description == "x, y, z"
    assert "local body frame" in analysis.coordinate_frame
    assert "local-plane" in analysis.degeneracy_description
    assert analysis.solver_type == "qr_fallback"


def test_dcreg_registration_icp_matches_point_to_plane_on_normal_geometry():
    source, target, expected = _make_asymmetric_box_clouds()
    threshold = 0.2
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=1e-9, relative_rmse=1e-9, max_iteration=20)

    baseline = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria)
    dcreg = o3d.pipelines.registration.registration_icp(
        source, target, threshold, np.eye(4),
        o3d.pipelines.registration.TransformationEstimationPointToPlaneDCReg(),
        criteria)

    assert np.isfinite(dcreg.transformation).all()
    np.testing.assert_allclose(dcreg.fitness, baseline.fitness, atol=1e-12)
    np.testing.assert_allclose(dcreg.inlier_rmse,
                               baseline.inlier_rmse,
                               atol=1e-12)
    np.testing.assert_allclose(dcreg.transformation,
                               baseline.transformation,
                               rtol=1e-8,
                               atol=1e-8)
    np.testing.assert_allclose(dcreg.transformation,
                               expected,
                               rtol=1e-8,
                               atol=1e-8)
