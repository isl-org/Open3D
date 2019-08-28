# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Advanced/surface_reconstruction_ball_pivoting.py

import open3d as o3d
import numpy as np
import sys

sys.path.append("../Basic")
import meshes


def problem_generator():
    o3d.utility.set_verbosity_level(o3d.utility.Debug)

    points = []
    normals = []
    for _ in range(4):
        for _ in range(4):
            pt = (np.random.uniform(-2, 2), np.random.uniform(-2, 2), 0)
            points.append(pt)
            normals.append((0, 0, 1))
    points = np.array(points, dtype=np.float64)
    normals = np.array(normals, dtype=np.float64)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.normals = o3d.utility.Vector3dVector(normals)
    radii = [1, 2]
    yield pcd, radii

    o3d.utility.set_verbosity_level(o3d.utility.Info)

    gt_mesh = o3d.geometry.TriangleMesh.create_sphere()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(100)
    radii = [0.5, 1, 2]
    yield pcd, radii

    gt_mesh = meshes.bunny()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(2000)
    radii = [0.005, 0.01, 0.02, 0.04]
    yield pcd, radii

    gt_mesh = meshes.armadillo()
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(2000)
    radii = [5, 10]
    yield pcd, radii


if __name__ == "__main__":
    for pcd, radii in problem_generator():
        o3d.visualization.draw_geometries([pcd])
        rec_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        o3d.visualization.draw_geometries([pcd, rec_mesh])
