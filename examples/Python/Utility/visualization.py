# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Utility/visualization.py

import copy
from open3d import *
import time

def geometry_generator():
    mesh = read_triangle_mesh("../../TestData/bathtub_0154.ply")
    yield mesh

    # TODO point set
    # TODO line set

def animate(geom):
    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(geom)

    scales = [s for np.linspace(2, 0.1, 10)] + [s for np.linspace(0.1, 2, 10)]
    print(geom)
    print(scales)
    for scale in scales:
        geom.scale(scale)
        vis.update_geometry()
        vis.poll_events()
        vis.update_renderer()
        time.sleep(50)

if __name__ == "__main__":
    for geom in geometry_generator():
        animate(geom)
