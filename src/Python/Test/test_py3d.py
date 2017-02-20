# enable this magic when you are using Jupyter (IPython) notebook
# %matplotlib inline

from py3d import *
import numpy as np
import sys, copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def test_py3d_eigen():
    print("Testing eigen in py3d ...")

    print("")
    print("Testing IntVector ...")
    vi = IntVector([1, 2, 3, 4, 5])
    vi1 = IntVector(vi) # valid copy
    vi2 = copy.copy(vi) # valid copy
    vi3 = copy.deepcopy(vi) # valid copy
    vi4 = vi[:] # valid copy
    print(vi)
    print(np.asarray(vi))
    vi[0] = 10
    np.asarray(vi)[1] = 22
    vi1[0] *= 5
    vi2[0] += 1
    vi3[0:2] = IntVector([40, 50])
    print(vi)
    print(vi1)
    print(vi2)
    print(vi3)
    print(vi4)

    print("")
    print("Testing DoubleVector ...")
    vd = DoubleVector([1, 2, 3])
    vd1 = DoubleVector([1.1, 1.2])
    vd2 = DoubleVector(np.asarray([0.1, 0.2]))
    print(vd)
    print(vd1)
    print(vd2)
    vd1.append(1.3)
    vd1.extend(vd2)
    print(vd1)

    print("")
    print("Testing Vector3dVector ...")
    vv3d = Vector3dVector([[1, 2, 3], [0.1, 0.2, 0.3]])
    vv3d1 = Vector3dVector(vv3d)
    vv3d2 = Vector3dVector(np.asarray(vv3d))
    vv3d3 = copy.deepcopy(vv3d)
    print(vv3d)
    print(np.asarray(vv3d))
    vv3d[0] = Vector3d([4, 5, 6])
    print(np.asarray(vv3d))
    # bad practice, the second [] will not support slice
    vv3d[0][0] = -1
    print(np.asarray(vv3d))
    # good practice, use [] after converting to numpy.array
    np.asarray(vv3d)[0][0] = 0
    print(np.asarray(vv3d))
    np.asarray(vv3d1)[:2, :2] = [[10, 11], [12, 13]]
    print(np.asarray(vv3d1))
    vv3d2.append(Vector3d([30, 31, 32]))
    print(np.asarray(vv3d2))
    vv3d3.extend(vv3d)
    print(np.asarray(vv3d3))

    print("")
    print("Testing Vector3iVector ...")
    vv3i = Vector3iVector([[1, 2, 3], [4, 5, 6]])
    print(vv3i)
    print(np.asarray(vv3i))

    print("")

def test_py3d_pointcloud():
    print("Testing pointcloud in py3d ...")
    print("")

def test_py3d_mesh():
    print("Testing mesh in py3d ...")
    mesh = CreateMeshFromFile("TestData/knot.ply")
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    print("")

def test_py3d_image():
    print("Testing image in py3d ...")
    print("")

def test_py3d_kdtree():
    print("Testing kdtree in py3d ...")
    print("")

def test_py3d_visualization():
    print("Testing visualization in py3d ...")
    mesh = CreateMeshFromFile("TestData/knot.ply")
    print("Try to render a mesh with normals " + str(mesh.HasVertexNormals()) + " and colors " + str(mesh.HasVertexColors()))
    DrawGeometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")
    mesh.ComputeVertexNormals()
    mesh.PaintUniformColor(Vector3d([0.1, 0.1, 0.7]))
    print(np.asarray(mesh.triangle_normals))
    print("We paint the mesh and render it.")
    DrawGeometries([mesh])
    print("We make a partial mesh of only the first half triangles.")
    mesh1 = copy.deepcopy(mesh)
    print(mesh1.triangles)
    mesh1.triangles = Vector3iVector(np.asarray(mesh1.triangles)[:len(mesh1.triangles)/2, :])
    mesh1.triangle_normals = Vector3dVector(np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals)/2, :])
    print(mesh1.triangles)
    DrawGeometries([mesh1])

    # let's draw some primitives
    mesh_sphere = CreateMeshSphere(radius = 1.0)
    mesh_sphere.ComputeVertexNormals()
    mesh_sphere.PaintUniformColor(Vector3d([0.1, 0.1, 0.7]))
    mesh_cylinder = CreateMeshCylinder(radius = 0.3, height = 4.0)
    mesh_cylinder.ComputeVertexNormals()
    mesh_cylinder.PaintUniformColor(Vector3d([0.1, 0.9, 0.1]))
    mesh_frame = CreateMeshCoordinateFrame(size = 0.6, origin = Vector3d([-2, -2, -2]))
    print("We draw a few primitives using collection.")
    DrawGeometries([mesh_sphere, mesh_cylinder, mesh_frame])
    print("We draw a few primitives using + operator of mesh.")
    DrawGeometries([mesh_sphere + mesh_cylinder + mesh_frame])

    print("")

if __name__ == "__main__":
    if len(sys.argv) == 1 or "eigen" in sys.argv:
        test_py3d_eigen()
    if len(sys.argv) == 1 or "pointcloud" in sys.argv:
        test_py3d_pointcloud()
    if len(sys.argv) == 1 or "mesh" in sys.argv:
        test_py3d_mesh()
    if len(sys.argv) == 1 or "image" in sys.argv:
        test_py3d_image()
    if len(sys.argv) == 1 or "kdtree" in sys.argv:
        test_py3d_kdtree()
    if len(sys.argv) == 1 or "visualization" in sys.argv:
        test_py3d_visualization()
