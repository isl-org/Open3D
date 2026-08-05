# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np
import copy


def translate():
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_tx = copy.deepcopy(mesh).translate((1.3, 0, 0))
    mesh_ty = copy.deepcopy(mesh).translate((0, 1.3, 0))
    print('Displaying original and translated geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": mesh
    }, {
        "name": "Translated (in X) Geometry",
        "geometry": mesh_tx
    }, {
        "name": "Translated (in Y) Geometry",
        "geometry": mesh_ty
    }],
                           show_ui=True)


def rotate():
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_r = copy.deepcopy(mesh)
    R = mesh.get_rotation_matrix_from_xyz((np.pi / 2, 0, np.pi / 4))
    mesh_r.rotate(R, center=(0, 0, 0))
    print('Displaying original and rotated geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": mesh
    }, {
        "name": "Rotated Geometry",
        "geometry": mesh_r
    }],
                           show_ui=True)


def scale():
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    mesh_s = copy.deepcopy(mesh).translate((2, 0, 0))
    mesh_s.scale(0.5, center=mesh_s.get_center())
    print('Displaying original and scaled geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": mesh
    }, {
        "name": "Scaled Geometry",
        "geometry": mesh_s
    }],
                           show_ui=True)


def transform():
    mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    T = np.eye(4)
    T[:3, :3] = mesh.get_rotation_matrix_from_xyz((0, np.pi / 3, np.pi / 2))
    T[0, 3] = 1
    T[1, 3] = 1.3
    print(T)
    mesh_t = copy.deepcopy(mesh).transform(T)
    print('Displaying original and transformed geometries ...')
    o3d.visualization.draw([{
        "name": "Original Geometry",
        "geometry": mesh
    }, {
        "name": "Transformed Geometry",
        "geometry": mesh_t
    }],
                           show_ui=True)


if __name__ == "__main__":

    translate()
    rotate()
    scale()
    transform()
