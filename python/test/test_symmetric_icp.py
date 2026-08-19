# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import pytest

import open3d as o3d


def test_symmetric_icp_binding_contract():
    kernel = o3d.pipelines.registration.HuberLoss(0.25)
    estimation = o3d.pipelines.registration.TransformationEstimationSymmetric(
        kernel)
    assert isinstance(estimation.kernel, o3d.pipelines.registration.HuberLoss)
    assert estimation.kernel.k == pytest.approx(0.25)

    source = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))
    target = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))

    with pytest.raises(RuntimeError,
                       match="requires both source and target to have normals"):
        o3d.pipelines.registration.registration_symmetric_icp(
            source, target, 0.1, estimation_method=estimation)
