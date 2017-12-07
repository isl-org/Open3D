import sys
import numpy as np
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	print("Testing mesh in py3d ...")
	mesh = read_triangle_mesh("../../TestData/knot.ply")
	draw_geometries([mesh])
	mesh.compute_vertex_normals()
	draw_geometries([mesh])
	print(mesh)
	print(np.asarray(mesh.vertices))
	print(np.asarray(mesh.triangles))
	print("")
