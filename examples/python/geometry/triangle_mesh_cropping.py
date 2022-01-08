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
import os
import sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

import open3d_example as o3dex

if __name__ == "__main__":
    mesh = o3dex.get_knot_mesh()
    print("Displaying original mesh ...")
    o3d.visualization.draw([mesh])

    print("Displaying mesh of only the first half triangles ...")
    mesh_cropped = copy.deepcopy(mesh)
    mesh_cropped.triangles = o3d.utility.Vector3iVector(
        np.asarray(mesh_cropped.triangles)[:len(mesh_cropped.triangles) //
                                           2, :])
    mesh_cropped.triangle_normals = o3d.utility.Vector3dVector(
        np.asarray(mesh_cropped.triangle_normals)
        [:len(mesh_cropped.triangle_normals) // 2, :])
    print(mesh_cropped.triangles)
    o3d.visualization.draw([mesh_cropped])
