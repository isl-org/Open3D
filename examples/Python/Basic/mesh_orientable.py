# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
from open3d import *

if __name__ == "__main__":
    print('start')
    mesh = create_mesh_moebius()
    print(mesh)
    pcl = PointCloud()
    pcl.points = mesh.vertices
    print(pcl)
    draw_geometries([mesh, pcl])
