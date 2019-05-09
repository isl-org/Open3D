# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Utility/visualization.py

import numpy as np
from open3d import *
import time

def geometry_generator():
    # mesh = create_mesh_sphere()
    # verts = np.asarray(mesh.vertices)
    # mesh.vertex_colors = Vector3dVector(np.random.uniform(0,1, size=verts.shape))
    # mesh.compute_vertex_normals()
    # yield mesh

    # pcl = PointCloud()
    # pcl.points = mesh.vertices
    # pcl.colors = mesh.vertex_colors
    # pcl.normals = mesh.vertex_normals
    # yield pcl

    # TODO line set
    ls = LineSet()
    yield ls

def animate(geom):
    vis = Visualizer()
    vis.create_window()

    geom.rotate(np.array((0.75, 0.5, 0)))
    vis.add_geometry(geom)

    scales = [0.9 for _ in range(30)] + [1/0.9 for _ in range(30)]
    for scale in scales:
        geom.scale(scale)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    phis = [(0, 0.1, 0.1) for _ in range(60)]
    for phi in phis:
        geom.rotate(phi)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

    ts = [(0.1, 0.1, -0.1) for _ in range(30)] + [(-0.1, -0.1, 0.1)  for _ in range(30)]
    for t in ts:
        geom.translate(t)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.05)

if __name__ == "__main__":
    for geom in geometry_generator():
        animate(geom)
