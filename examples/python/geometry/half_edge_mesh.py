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
import os
import sys

dir_path = os.path.dirname(os.path.abspath(__file__))
sys.path.append(dir_path + "/..")
import open3d_example as o3dex

if __name__ == "__main__":
    # Initialize a HalfEdgeTriangleMesh from TriangleMesh
    path_to_mesh = dir_path + "/../../test_data/sphere.ply"
    mesh = o3d.io.read_triangle_mesh(path_to_mesh)
    bbox = o3d.geometry.AxisAlignedBoundingBox()
    bbox.min_bound = [-1, -1, -1]
    bbox.max_bound = [1, 0.6, 1]
    mesh = mesh.crop(bbox)
    het_mesh = o3d.geometry.HalfEdgeTriangleMesh.create_from_triangle_mesh(mesh)
    o3d.visualization.draw_geometries([het_mesh], mesh_show_back_face=True)

    # Colorize boundary vertices to red
    vertex_colors = 0.75 * np.ones((len(het_mesh.vertices), 3))
    for boundary in het_mesh.get_boundaries():
        for vertex_id in boundary:
            vertex_colors[vertex_id] = [1, 0, 0]
    het_mesh.vertex_colors = o3d.utility.Vector3dVector(vertex_colors)
    o3d.visualization.draw_geometries([het_mesh], mesh_show_back_face=True)
