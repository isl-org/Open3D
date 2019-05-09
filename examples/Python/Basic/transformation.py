# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Utility/visualization.py

import numpy as np
from open3d import *
import time

def geometry_generator():
    mesh = create_mesh_sphere()
    verts = np.asarray(mesh.vertices)
    colors = np.random.uniform(0,1, size=verts.shape)
    mesh.vertex_colors = Vector3dVector(colors)
    mesh.compute_vertex_normals()
    yield mesh

    pcl = PointCloud()
    pcl.points = mesh.vertices
    pcl.colors = mesh.vertex_colors
    pcl.normals = mesh.vertex_normals
    yield pcl

    ls = LineSet()
    ls.points = Vector3dVector(np.array([(0,0,0), (1,0,0), (1,0,1), (0,0,1),
        (0,1,0), (1,1,0), (1,1,1), (0,1,1)], dtype=np.float64))
    ls.lines = Vector2iVector(np.array([(0,1), (0,4), (0,3), (2,3), (2,1),
        (2,6), (5,1), (5,4), (5,6), (7,3), (7,6), (7,4)]))
    yield ls

def animate(geom):
    vis = Visualizer()
    vis.create_window()

    geom.rotate(np.array((0.75, 0.5, 0)))
    vis.add_geometry(geom)

    scales = [0.9 for _ in range(30)] + [1/0.9 for _ in range(30)]
    phis = [(np.pi/30, 0, np.pi/30) for _ in range(60)]
    ts = [(0.1, 0.1, -0.1) for _ in range(30)] + [(-0.1, -0.1, 0.1)  for _ in range(30)]

    for scale, phi, t in zip(scales, phis, ts):
        geom.scale(scale).rotate(phi)
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

if __name__ == "__main__":
    for geom in geometry_generator():
        animate(geom)
