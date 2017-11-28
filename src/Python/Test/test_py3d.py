# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# enable this magic when you are using Jupyter (IPython) notebook
# %matplotlib inline
import sys
sys.path.append("..")

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
	pcd = read_point_cloud("../TestData/fragment.ply")
	print(pcd)
	print(np.asarray(pcd.points))
	draw_geometries([pcd])
	print("Downsample the point cloud with a voxel of 0.05")
	downpcd = voxel_down_sample(pcd, voxel_size = 0.05)
	draw_geometries([downpcd])
	print("Recompute the normal of the downsampled point cloud")
	estimate_normals(downpcd, search_param = KDTreeSearchParamHybrid(radius = 0.1, max_nn = 30))
	draw_geometries([downpcd])
	print("")
	print("We load a polygon volume and use it to crop the original point cloud")
	vol = read_selection_polygon_volume("../TestData/Crop/cropped.json")
	chair = vol.crop_point_cloud(pcd)
	draw_geometries([chair])
	print("")

def test_py3d_mesh():
	print("Testing mesh in py3d ...")
	mesh = read_triangle_mesh("../TestData/knot.ply")
	print(mesh)
	print(np.asarray(mesh.vertices))
	print(np.asarray(mesh.triangles))
	print("")

def test_py3d_image():
	print("Testing image in py3d ...")
	print("Convert an image to numpy and draw it with matplotlib.")
	x = read_image("../TestData/image.PNG")
	print(x)
	plt.imshow(np.asarray(x))
	plt.show()

	print("Convet a numpy image to Image and show it with DrawGeomtries().")
	y = mpimg.imread("../TestData/lena_color.jpg")
	print(y.shape)
	yy = Image(y)
	print(yy)
	draw_geometries([yy])

	print("Render a channel of the previous image.")
	z = np.array(y[:,:,1])
	print(z.shape)
	print(z.strides)
	zz = Image(z)
	print(zz)
	draw_geometries([zz])

	print("Write the previous image to file then load it with matplotlib.")
	write_image("test.jpg", zz, quality = 100)
	zzz = mpimg.imread("test.jpg")
	plt.imshow(zzz)
	plt.show()

	print("Testing basic image processing module.")
	im_raw = mpimg.imread("../TestData/lena_color.jpg")
	im = Image(im_raw)
	im_g3 = filter_image(im, ImageFilterType.Gaussian3)
	im_g5 = filter_image(im, ImageFilterType.Gaussian5)
	im_g7 = filter_image(im, ImageFilterType.Gaussian7)
	im_gaussian = [im, im_g3, im_g5, im_g7]
	pyramid_levels = 4
	pyramid_with_gaussian_filter = True
	im_pyramid = create_image_pyramid(im, pyramid_levels,
            pyramid_with_gaussian_filter)
	im_dx = filter_image(im, ImageFilterType.Sobel3dx)
	im_dx_pyramid = filter_image_pyramid(im_pyramid, ImageFilterType.Sobel3dx)
	im_dy = filter_image(im, ImageFilterType.Sobel3dy)
	im_dy_pyramid = filter_image_pyramid(im_pyramid, ImageFilterType.Sobel3dy)
	switcher = {
		0: im_gaussian,
		1: im_pyramid,
		2: im_dx_pyramid,
		3: im_dy_pyramid,
	}
	for i in range(4):
		for j in range(pyramid_levels):
			plt.subplot(4, pyramid_levels, i*4+j+1)
			plt.imshow(switcher.get(i)[j])
	plt.show()

	print("Final test: load an RGB-D image pair and convert to pointcloud.")
	im1 = read_image("../TestData/RGBD/depth/00000.png")
	im2 = read_image("../TestData/RGBD/color/00000.jpg")
	im = create_rgbd_image_from_color_and_depth(im2, im1, 1000.0, 5.0, False)
	plt.figure(figsize=(12,8))
	plt.subplot(1, 2, 1)
	plt.imshow(im.depth)
	plt.subplot(1, 2, 2)
	plt.imshow(im.color)
	plt.show()
	pcd = create_point_cloud_from_rgbd_image(im, PinholeCameraIntrinsic.PrimeSenseDefault)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])

	print("")

def test_py3d_kdtree():
	print("Testing kdtree in py3d ...")
	print("Load a point cloud and paint it black.")
	pcd = read_point_cloud("../TestData/Feature/cloud_bin_0.pcd")
	pcd.paint_uniform_color([0, 0, 0])
	pcd_tree = KDTreeFlann(pcd)
	print("Paint the 1500th point red.")
	pcd.colors[1500] = [1, 0, 0]
	print("Find its 200 nearest neighbors, paint blue.")
	[k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
	np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]
	print("Find its neighbors with distance less than 0.2, paint green.")
	[k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
	np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]
	print("Visualize the point cloud.")
	draw_geometries([pcd])
	print("")

	print("Load two aligned point clouds.")
	pcd0 = read_point_cloud("../TestData/Feature/cloud_bin_0.pcd")
	pcd1 = read_point_cloud("../TestData/Feature/cloud_bin_1.pcd")
	pcd0.paint_uniform_color([1, 0.706, 0])
	pcd1.paint_uniform_color([0, 0.651, 0.929])
	draw_geometries([pcd0, pcd1])
	print("Load their FPFH feature and evaluate.")
	print("Black : matching distance > 0.2")
	print("White : matching distance = 0")
	feature0 = read_feature("../TestData/Feature/cloud_bin_0.fpfh.bin")
	feature1 = read_feature("../TestData/Feature/cloud_bin_1.fpfh.bin")
	fpfh_tree = KDTreeFlann(feature1)
	for i in range(len(pcd0.points)):
		[_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
		dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
		c = (0.2 - np.fmin(dis, 0.2)) / 0.2
		pcd0.colors[i] = [c, c, c]
	draw_geometries([pcd0])
	print("")

	print("Load their L32D feature and evaluate.")
	print("Black : matching distance > 0.2")
	print("White : matching distance = 0")
	feature0 = read_feature("../TestData/Feature/cloud_bin_0.d32.bin")
	feature1 = read_feature("../TestData/Feature/cloud_bin_1.d32.bin")
	fpfh_tree = KDTreeFlann(feature1)
	for i in range(len(pcd0.points)):
		[_, idx, _] = fpfh_tree.search_knn_vector_xd(feature0.data[:, i], 1)
		dis = np.linalg.norm(pcd0.points[i] - pcd1.points[idx[0]])
		c = (0.2 - np.fmin(dis, 0.2)) / 0.2
		pcd0.colors[i] = [c, c, c]
	draw_geometries([pcd0])
	print("")

def test_py3d_posegraph():
	print("Testing PoseGraph in py3d ...")
	pose_graph = read_pose_graph("../TestData/test_pose_graph.json")
	print(pose_graph)
	write_pose_graph("../TestData/test_pose_graph_copy.json", pose_graph)
	print("")

def test_py3d_camera():
	print("Testing camera in py3d ...")
	print(PinholeCameraIntrinsic.PrimeSenseDefault)
	print(PinholeCameraIntrinsic.PrimeSenseDefault.intrinsic_matrix)
	print(PinholeCameraIntrinsic())
	x = PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
	print(x)
	print(x.intrinsic_matrix)
	write_pinhole_camera_intrinsic("test.json", x)
	y = read_pinhole_camera_intrinsic("test.json")
	print(y)
	print(np.asarray(y.intrinsic_matrix))

	print("Final test, read a trajectory and combine all the RGB-D images.")
	pcds = [];
	trajectory = read_pinhole_camera_trajectory("../TestData/RGBD/trajectory.log")
	write_pinhole_camera_trajectory("test.json", trajectory)
	print(trajectory)
	print(trajectory.extrinsic)
	print(np.asarray(trajectory.extrinsic))
	for i in range(5):
		im1 = read_image("../TestData/RGBD/depth/{:05d}.png".format(i))
		im2 = read_image("../TestData/RGBD/color/{:05d}.jpg".format(i))
		im = create_rgbd_image_from_color_and_depth(im2, im1, 1000.0, 5.0, False)
		pcd = create_point_cloud_from_rgbd_image(im, trajectory.intrinsic, trajectory.extrinsic[i])
		pcds.append(pcd)
	draw_geometries(pcds)
	print("")

def test_py3d_visualization():
	print("Testing visualization in py3d ...")
	mesh = read_triangle_mesh("../TestData/knot.ply")
	print("Try to render a mesh with normals " + str(mesh.has_vertex_normals()) + " and colors " + str(mesh.has_vertex_colors()))
	draw_geometries([mesh])
	print("A mesh with no normals and no colors does not seem good.")
	mesh.compute_vertex_normals()
	mesh.paint_uniform_color([0.1, 0.1, 0.7])
	print(np.asarray(mesh.triangle_normals))
	print("We paint the mesh and render it.")
	draw_geometries([mesh])
	print("We make a partial mesh of only the first half triangles.")
	mesh1 = copy.deepcopy(mesh)
	print(mesh1.triangles)
	mesh1.triangles = Vector3iVector(np.asarray(mesh1.triangles)[:len(mesh1.triangles)/2, :])
	mesh1.triangle_normals = Vector3dVector(np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals)/2, :])
	print(mesh1.triangles)
	draw_geometries([mesh1])

	# let's draw some primitives
	mesh_sphere = create_mesh_sphere(radius = 1.0)
	mesh_sphere.compute_vertex_normals()
	mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
	mesh_cylinder = create_mesh_cylinder(radius = 0.3, height = 4.0)
	mesh_cylinder.compute_vertex_normals()
	mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
	mesh_frame = create_mesh_coordinate_frame(size = 0.6, origin = [-2, -2, -2])
	print("We draw a few primitives using collection.")
	draw_geometries([mesh_sphere, mesh_cylinder, mesh_frame])
	print("We draw a few primitives using + operator of mesh.")
	draw_geometries([mesh_sphere + mesh_cylinder + mesh_frame])

	print("")

def test_py3d_icp():
	traj = read_trajectory("../TestData/ICP/init.log")
	pcds = []
	threshold = 0.02
	for i in range(3):
		pcds.append(read_point_cloud("../TestData/ICP/cloud_bin_{:d}.pcd".format(i)))

	for reg in traj:
		target = pcds[reg.metadata[0]]
		source = pcds[reg.metadata[1]]
		trans = reg.pose
		evaluation_init = evaluate_registration(source, target, threshold, trans)
		print(evaluation_init)

		print("Apply point-to-point ICP")
		reg_p2p = registration_icp(source, target, threshold, trans, TransformationEstimationPointToPoint())
		print(reg_p2p)
		print("Transformation is:")
		print(reg_p2p.transformation)

		print("Apply point-to-plane ICP")
		reg_p2l = registration_icp(source, target, threshold, trans, TransformationEstimationPointToPlane())
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
	if len(sys.argv) == 1 or "posegraph" in sys.argv:
		test_py3d_posegraph()
	if len(sys.argv) == 1 or "visualization" in sys.argv:
		test_py3d_visualization()
	if len(sys.argv) == 1 or "icp" in sys.argv:
		test_py3d_icp()
