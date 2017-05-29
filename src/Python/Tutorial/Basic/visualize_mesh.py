import sys
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":
	mesh = ReadTriangleMesh("../../TestData/knot.ply")
	mesh.ComputeVertexNormals()
	DrawGeometries([mesh])
