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

if __name__ == "__main__":
    bunny = o3d.data.BunnyMesh()
    mesh_in = o3d.io.read_triangle_mesh(bunny.path)
    mesh_in.compute_vertex_normals()

    print("Before Simplification: ", mesh_in)
    o3d.visualization.draw_geometries([mesh_in])

    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 32
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print("After Simplification with voxel size =", voxel_size, ":\n", mesh_smp)
    o3d.visualization.draw_geometries([mesh_smp])

    voxel_size = max(mesh_in.get_max_bound() - mesh_in.get_min_bound()) / 16
    mesh_smp = mesh_in.simplify_vertex_clustering(
        voxel_size=voxel_size,
        contraction=o3d.geometry.SimplificationContraction.Average)
    print("After Simplification with voxel size =", voxel_size, ":\n", mesh_smp)
    o3d.visualization.draw_geometries([mesh_smp])
