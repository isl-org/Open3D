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

if __name__ == "__main__":
    # Read a mesh and get its data as numpy arrays.
    knot_mesh = o3d.data.KnotMesh()
    mesh = o3d.io.read_triangle_mesh(knot_mesh.path)
    mesh.paint_uniform_color([0.5, 0.1, 0.3])
    print('Vertices:')
    print(np.asarray(mesh.vertices))
    print('Vertex Colors:')
    print(np.asarray(mesh.vertex_colors))
    print('Vertex Normals:')
    print(np.asarray(mesh.vertex_normals))
    print('Triangles:')
    print(np.asarray(mesh.triangles))
    print('Triangle Normals:')
    print(np.asarray(mesh.triangle_normals))
    print("Displaying mesh ...")
    print(mesh)
    o3d.visualization.draw([mesh])

    # Create a mesh using numpy arrays with random colors.
    N = 5
    vertices = o3d.utility.Vector3dVector(
        np.array([[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1], [0.5, 0.5, 0.5]]))
    triangles = o3d.utility.Vector3iVector(
        np.array([[0, 1, 2], [0, 2, 3], [0, 4, 1], [1, 4, 2], [2, 4, 3],
                  [3, 4, 0]]))
    mesh_np = o3d.geometry.TriangleMesh(vertices, triangles)
    mesh_np.vertex_colors = o3d.utility.Vector3dVector(
        np.random.uniform(0, 1, size=(N, 3)))
    mesh_np.compute_vertex_normals()
    print(np.asarray(mesh_np.triangle_normals))
    print("Displaying mesh made using numpy ...")
    o3d.visualization.draw_geometries([mesh_np])
