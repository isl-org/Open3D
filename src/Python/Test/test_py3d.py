# enable this magic when you are using Jupyter (IPython) notebook
# %matplotlib inline

from py3d import *
import numpy as np
import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def test_py3d_eigen():
    print("Test eigen in py3d")

def test_py3d_mesh():
    print("Test mesh in py3d")
    mesh = CreateMeshFromFile("TestData/knot.ply")
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    print("Try to render a mesh with normals " + str(mesh.HasVertexNormals()))
    DrawGeometries([mesh])

if __name__ == "__main__":
    if len(sys.argv) == 1 or "eigen" in sys.argv:
        test_py3d_eigen()
    if len(sys.argv) == 1 or "mesh" in sys.argv:
        test_py3d_mesh()
