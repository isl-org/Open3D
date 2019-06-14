# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Utility/visualization.py

import numpy as np
import open3d as o3d
import time


def geometry_generator():
    mesh = o3d.geometry.TriangleMesh.create_sphere()
    verts = np.asarray(mesh.vertices)
    colors = np.random.uniform(0, 1, size=verts.shape)
    mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    mesh.compute_vertex_normals()

    pcl = o3d.geometry.PointCloud()
    pcl.points = mesh.vertices
    pcl.colors = mesh.vertex_colors
    pcl.normals = mesh.vertex_normals
    yield pcl

    yield o3d.geometry.LineSet.create_from_triangle_mesh(mesh)

    yield mesh


def animate(geom):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    geom.rotate(np.array((0.75, 0.5, 0)))
    vis.add_geometry(geom)

    scales = [0.9 for _ in range(30)] + [1 / 0.9 for _ in range(30)]
    axisangles = [(0.2 / np.sqrt(2), 0.2 / np.sqrt(2), 0) for _ in range(60)]
    ts = [(0.1, 0.1, -0.1) for _ in range(30)
         ] + [(-0.1, -0.1, 0.1) for _ in range(30)]

    for scale, aa in zip(scales, axisangles):
        geom.scale(scale).rotate(aa,
                                 center=False,
                                 type=o3d.geometry.RotationType.AxisAngle)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    for t in ts:
        geom.translate(t)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    for scale, aa, t in zip(scales, axisangles, ts):
        geom.scale(scale).translate(t).rotate(
            aa, center=True, type=o3d.geometry.RotationType.AxisAngle)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)


if __name__ == "__main__":
    for geom in geometry_generator():
        animate(geom)
