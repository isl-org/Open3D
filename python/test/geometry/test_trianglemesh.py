# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import math

import numpy as np
import open3d as o3d


def test_self_intersection_issue_5117():
    vertices = np.array([[0.0, 0.13918686, 1.0], [0.0, 0.0, 1.1270161],
                         [1.0, 0.0, 1.0284119], [1.0, 1.1269569, 0.0],
                         [1.0, 0.03113556, 1.0], [2.0, 1.0189056, 0.0]])
    triangles = np.array([[0, 1, 2], [3, 4, 5]])
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices),
                                     o3d.utility.Vector3iVector(triangles))

    # Classification of the separated issue #5117 triangles must remain false
    # before and after a rigid rotation.
    assert not mesh.is_self_intersecting()
    assert len(mesh.get_self_intersecting_triangles()) == 0

    sqrt_half = math.sqrt(0.5)
    rotation = np.array([[1.0, 0.0, 0.0], [0.0, sqrt_half, sqrt_half],
                         [0.0, -sqrt_half, sqrt_half]])
    mesh.rotate(rotation)
    assert not mesh.is_self_intersecting()
    assert len(mesh.get_self_intersecting_triangles()) == 0
