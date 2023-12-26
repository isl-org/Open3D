# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import copy


def translate():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    pcd_tx = copy.deepcopy(pcd).translate((150, 0, 0))
    pcd_ty = copy.deepcopy(pcd).translate((0, 200, 0))
    print('Displaying original and translated geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Translated (in X) Geometry",
        "geometry": pcd_tx
    }, {
        "name": "Translated (in Y) Geometry",
        "geometry": pcd_ty
    }],
                           show_ui=True)


def rotate():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    pcd_r = copy.deepcopy(pcd).translate((200, 0, 0))
    R = pcd.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    pcd_r.rotate(R)
    print('Displaying original and rotated geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Rotated Geometry",
        "geometry": pcd_r
    }],
                           show_ui=True)


def scale():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    pcd_s = copy.deepcopy(pcd).translate((200, 0, 0))
    pcd_s.scale(0.5, center=pcd_s.get_center())
    print('Displaying original and scaled geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Scaled Geometry",
        "geometry": pcd_s
    }],
                           show_ui=True)


def transform():
    armadillo_data = o3d.data.ArmadilloMesh()
    pcd = o3d.io.read_triangle_mesh(
        armadillo_data.path).sample_points_poisson_disk(5000)
    T = np.eye(4)
    T[:3, :3] = pcd.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[0, 3] = 150
    T[1, 3] = 200
    print(T)
    pcd_t = copy.deepcopy(pcd).transform(T)
    print('Displaying original and transformed geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": pcd
    }, {
        "name": "Transformed Geometry",
        "geometry": pcd_t
    }],
                           show_ui=True)


if __name__ == "__main__":

    translate()
    rotate()
    scale()
    transform()
