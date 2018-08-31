# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Basic/visualization.py

import numpy as np
from open3d import *

if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("../../TestData/fragment.ply")
    draw_geometries([pcd])

    print("Let\'s draw some primitives")
    mesh_sphere = create_mesh_sphere(radius = 1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = create_mesh_cylinder(radius = 0.3, height = 4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = create_mesh_coordinate_frame(size = 0.6, origin = [-2, -2, -2])

    print("We draw a few primitives using collection.")
    draw_geometries([mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    draw_geometries([mesh_sphere + mesh_cylinder + mesh_frame])

    print("Let\'s draw a cubic that consists of 8 points and 12 lines")
    points = [[0,0,0],[1,0,0],[0,1,0],[1,1,0],
              [0,0,1],[1,0,1],[0,1,1],[1,1,1]]
    lines = [[0,1],[0,2],[1,3],[2,3],
             [4,5],[4,6],[5,7],[6,7],
             [0,4],[1,5],[2,6],[3,7]]
    colors = [[1, 0, 0] for i in range(len(lines))]
    line_set = LineSet()
    line_set.points = Vector3dVector(points)
    line_set.lines = Vector2iVector(lines)
    line_set.colors = Vector3dVector(colors)
    draw_geometries([line_set])
