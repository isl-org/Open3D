# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":
	mesh = read_triangle_mesh("../../TestData/knot.ply")
	mesh.compute_vertex_normals()
	draw_geometries([mesh])
