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
