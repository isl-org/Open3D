# enable this magic when you are using Jupyter (IPython) notebook
# %matplotlib inline

from py3d import *
import numpy as np
import sys, copy
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from trajectory_io import *

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
    vv3d[0] = [4, 5, 6]
    print(np.asarray(vv3d))
    # bad practice, the second [] will not support slice
    vv3d[0][0] = -1
    print(np.asarray(vv3d))
    # good practice, use [] after converting to numpy.array
    np.asarray(vv3d)[0][0] = 0
    print(np.asarray(vv3d))
    np.asarray(vv3d1)[:2, :2] = [[10, 11], [12, 13]]
    print(np.asarray(vv3d1))
    vv3d2.append([30, 31, 32])
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
    print("Testing point cloud in py3d ...")
    print("Load a point cloud, print it, and render it")
    pcd = ReadPointCloud("TestData/fragment.ply")
    print(pcd)
    print(np.asarray(pcd.points))
    DrawGeometries([pcd])
    print("Downsample the point cloud with a voxel of 0.05")
    downpcd = VoxelDownSample(pcd, voxel_size = 0.05)
    DrawGeometries([downpcd])
    print("Recompute the normal of the downsampled point cloud")
    EstimateNormals(downpcd, search_param = KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
    DrawGeometries([downpcd])
    print("")
    print("We load a polygon volume and use it to crop the original point cloud")
    vol = ReadSelectionPolygonVolume("TestData/Crop/cropped.json")
    chair = vol.CropPointCloud(pcd)
    DrawGeometries([chair])
    print("")

def test_py3d_mesh():
    print("Testing mesh in py3d ...")
    mesh = ReadTriangleMesh("TestData/knot.ply")
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    print("")

def test_py3d_image():
    print("Testing image in py3d ...")
    print("Convert an image to numpy and draw it with matplotlib.")
    x = ReadImage("TestData/image.PNG")
    print(x)
    plt.imshow(np.asarray(x))
    plt.show()

    print("Convet a numpy image to Image and show it with DrawGeomtries().")
    y = mpimg.imread("TestData/lena_color.jpg")
    print(y.shape)
    yy = Image(y)
    print(yy)
    DrawGeometries([yy])

    print("Render a channel of the previous image.")
    z = np.array(y[:,:,1])
    print(z.shape)
    print(z.strides)
    zz = Image(z)
    print(zz)
    DrawGeometries([zz])

    print("Write the previous image to file then load it with matplotlib.")
    WriteImage("test.jpg", zz, quality = 100)
    zzz = mpimg.imread("test.jpg")
    plt.imshow(zzz)
    plt.show()

    print("Final test: load an RGB-D image pair and convert to pointcloud.")
    im1 = ReadImage("TestData/RGBD/depth/00000.png")
    im2 = ReadImage("TestData/RGBD/color/00000.jpg")
    plt.figure(figsize=(12,8))
    plt.subplot(1, 2, 1)
    plt.imshow(np.asarray(im1, dtype=np.float64) / 1000.0)
    plt.subplot(1, 2, 2)
    plt.imshow(im2)
    plt.show()
    pcd = CreatePointCloudFromRGBDImage(im1, im2, PinholeCameraIntrinsic.PrimeSenseDefault)
    # Flip it, otherwise the pointcloud will be upside down
    pcd.Transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    DrawGeometries([pcd])

    print("")

def test_py3d_kdtree():
    print("Testing kdtree in py3d ...")
    print("Load a point cloud and paint it black.")
    pcd = ReadPointCloud('TestData/Feature/cloud_bin_0.pcd')
    pcd.PaintUniformColor([0, 0, 0])
    pcd_tree = KDTreeFlann(pcd)
    print("Paint the 1500th point red.")
    pcd.colors[1500] = [1, 0, 0]
    print("Find its 200 nearest neighbors, paint blue.")
    [k, idx, _] = pcd_tree.SearchKNNVector3D(pcd.points[1500], 200)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
    print("Find its neighbors with distance less than 0.2, paint green.")
    [k, idx, _] = pcd_tree.SearchRadiusVector3D(pcd.points[1500], 0.2)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
    print("Visualize the point cloud.")
    DrawGeometries([pcd])
    print("")

    print("Load two aligned point clouds.")
    pcd0 = ReadPointCloud('TestData/Feature/cloud_bin_0.pcd')
    pcd1 = ReadPointCloud('TestData/Feature/cloud_bin_1.pcd')
    pcd0.PaintUniformColor([1, 0.706, 0])
    pcd1.PaintUniformColor([0, 0.651, 0.929])
    DrawGeometries([pcd0, pcd1])
    print("Load their FPFH feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = ReadFeature('TestData/Feature/cloud_bin_0.fpfh.bin')
    feature1 = ReadFeature('TestData/Feature/cloud_bin_1.fpfh.bin')
    fpfh_tree = KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.SearchKNNVectorXD(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    DrawGeometries([pcd0])
    print("")

    print("Load their L32D feature and evaluate.")
    print("Black : matching distance > 0.2")
    print("White : matching distance = 0")
    feature0 = ReadFeature('TestData/Feature/cloud_bin_0.d32.bin')
    feature1 = ReadFeature('TestData/Feature/cloud_bin_1.d32.bin')
    fpfh_tree = KDTreeFlann(feature1)
    for i in range(len(pcd0.points)):
        [_, idx, _] = fpfh_tree.SearchKNNVectorXD(feature0.data[:, i], 1)
        dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
        c = (0.2 - np.fmin(dis, 0.2)) / 0.2
        pcd0.colors[i] = [c, c, c]
    DrawGeometries([pcd0])
    print("")

def test_py3d_camera():
    print("Testing camera in py3d ...")
    print(PinholeCameraIntrinsic.PrimeSenseDefault)
    print(PinholeCameraIntrinsic.PrimeSenseDefault.intrinsic_matrix)
    print(PinholeCameraIntrinsic())
    x = PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    print(x)
    print(x.intrinsic_matrix)
    WritePinholeCameraIntrinsic("test.json", x)
    y = ReadPinholeCameraIntrinsic("test.json")
    print(y)
    print(np.asarray(y.intrinsic_matrix))

    print("Final test, read a trajectory and combine all the RGB-D images.")
    pcds = [];
    trajectory = ReadPinholeCameraTrajectory("TestData/RGBD/trajectory.log")
    WritePinholeCameraTrajectory("test.json", trajectory)
    print(trajectory)
    print(trajectory.extrinsic)
    print(np.asarray(trajectory.extrinsic))
    for i in range(5):
        im1 = ReadImage("TestData/RGBD/depth/{:05d}.png".format(i))
        im2 = ReadImage("TestData/RGBD/color/{:05d}.jpg".format(i))
        pcd = CreatePointCloudFromRGBDImage(im1, im2, trajectory.intrinsic)
        pcd.Transform(trajectory.extrinsic[i])
        pcds.append(pcd)
    DrawGeometries(pcds)
    print("")

def test_py3d_visualization():
    print("Testing visualization in py3d ...")
    mesh = ReadTriangleMesh("TestData/knot.ply")
    print("Try to render a mesh with normals " + str(mesh.HasVertexNormals()) + " and colors " + str(mesh.HasVertexColors()))
    DrawGeometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")
    mesh.ComputeVertexNormals()
    mesh.PaintUniformColor([0.1, 0.1, 0.7])
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
    mesh_sphere.PaintUniformColor([0.1, 0.1, 0.7])
    mesh_cylinder = CreateMeshCylinder(radius = 0.3, height = 4.0)
    mesh_cylinder.ComputeVertexNormals()
    mesh_cylinder.PaintUniformColor([0.1, 0.9, 0.1])
    mesh_frame = CreateMeshCoordinateFrame(size = 0.6, origin = [-2, -2, -2])
    print("We draw a few primitives using collection.")
    DrawGeometries([mesh_sphere, mesh_cylinder, mesh_frame])
    print("We draw a few primitives using + operator of mesh.")
    DrawGeometries([mesh_sphere + mesh_cylinder + mesh_frame])

    print("")

def test_py3d_icp():
    traj = read_trajectory("TestData/ICP/init.log")
    pcds = []
    threshold = 0.02
    for i in range(3):
        pcds.append(ReadPointCloud("TestData/ICP/cloud_bin_{:d}.pcd".format(i)))

    for reg in traj:
        target = pcds[reg.metadata[0]]
        source = pcds[reg.metadata[1]]
        trans = reg.pose
        evaluation_init = EvaluateRegistration(source, target, threshold, trans)
        print(evaluation_init)

        print("Apply point-to-point ICP")
        reg_p2p = RegistrationICP(source, target, threshold, trans, TransformationEstimationPointToPoint())
        print(reg_p2p)
        print("Transformation is:")
        print(reg_p2p.transformation)

        print("Apply point-to-plane ICP")
        reg_p2l = RegistrationICP(source, target, threshold, trans, TransformationEstimationPointToPlane())
        print(reg_p2l)
        print("Transformation is:")
        print(reg_p2l.transformation)
        print("")

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
    if len(sys.argv) == 1 or "camera" in sys.argv:
        test_py3d_camera()
    if len(sys.argv) == 1 or "visualization" in sys.argv:
        test_py3d_visualization()
    if len(sys.argv) == 1 or "icp" in sys.argv:
        test_py3d_icp()
