# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/surface_reconstruction_poisson.py

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import os

import sys

sys.path.append(
    os.path.join(os.path.dirname(os.path.realpath(__file__)), "../Misc"))
import meshes

if __name__ == "__main__":
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    pcd = meshes.eagle()
    print(pcd)
    o3d.visualization.draw_geometries([pcd])

    print('run Poisson surface reconstruction')
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=8)
    print(mesh)
    o3d.visualization.draw_geometries([mesh])

    print('visualize densities')
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    o3d.visualization.draw_geometries([density_mesh])

    print('remove low density vertices')
    vertices_to_remove = densities < np.quantile(densities, 0.1)
    mesh.remove_vertices_by_mask(vertices_to_remove)
    print(mesh)
    o3d.visualization.draw_geometries([mesh])
