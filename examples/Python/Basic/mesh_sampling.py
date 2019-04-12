# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/mesh_sampling.py

import numpy as np
from open3d import *

if __name__ == "__main__":
  mesh = read_triangle_mesh('bathtub_0154.ply')
  draw_geometries([mesh])

  pcd = mesh.sample_points_uniformly(5000)
  draw_geometries(pcd)
