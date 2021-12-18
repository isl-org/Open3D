# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
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
