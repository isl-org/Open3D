import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import math
import sys
sys.path.append("../..")
from py3d import *
#from joblib import Parallel, delayed

# some global parameters
path_dataset = '/Users/jaesikpa/Dropbox/Intel/fragment/011/'
path_fragment = path_dataset + 'fragments/'
frames_per_fragment = 20
keyframes_per_n_frame = 5
opencv_installed = True


def initialize_opencv():
	global opencv_installed
	opencv_installed = True
	try:
		import cv2
	except ImportError, e:
		pass
		print('OpenCV is not detected. Using Identity as an initial')
		opencv_installed = False
	if opencv_installed:
		print('OpenCV is detected. Using ORB + 5pt algorithm')


def process_one_rgbd_pair(s, t, color_files, depth_files):
	#SetVerbosityLevel(VerbosityLevel.Debug)

	# read images
	color_s = ReadImage(color_files[s])
	depth_s = ReadImage(depth_files[s])
	color_t = ReadImage(color_files[t])
	depth_t = ReadImage(depth_files[t])
	# print(color_files[s])
	# print(color_files[t])
	source_rgbd_image = CreateRGBDImageFromColorAndDepth(color_s, depth_s)
	target_rgbd_image = CreateRGBDImageFromColorAndDepth(color_t, depth_t)

	pinhole_camera_intrinsic = ReadPinholeCameraIntrinsic(
			"../../TestData/camera.json")

	# initialize_camera_pose
	if abs(s-t) is not 1 and opencv_installed:
		odo_init = pose_estimation(source_rgbd_image, target_rgbd_image,
				pinhole_camera_intrinsic)
	else:
		odo_init = np.identity(4)

	# perform RGB-D odometry
	[success, trans, info] = ComputeRGBDOdometry(
			source_rgbd_image, target_rgbd_image,
			pinhole_camera_intrinsic, odo_init,
			RGBDOdometryJacobianFromHybridTerm())
	if success:
		# print(trans)
		return [trans, info]
	else:
		return [np.identity(4), np.zeros(6)]


def get_flie_lists(path_dataset):
	# get list of color and depth images
	path_color = path_dataset + 'image/'
	path_depth = path_dataset + 'depth/'
	color_files = [path_color + f for f in listdir(path_color)
			if isfile(join(path_color, f))]
	depth_files = [path_depth + f for f in listdir(path_depth)
			if isfile(join(path_depth, f))]
	return color_files, depth_files


def make_one_fragment(fragment_id):
	SetVerbosityLevel(VerbosityLevel.Error)
	pose_graph = PoseGraph()
	# odometry
	sid = fragment_id * frames_per_fragment
	eid = sid + frames_per_fragment - 1
	trans_odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(trans_odometry))

	# keyframe loop closure
	for s in range(sid, eid):
		for t in range(sid, eid):
			if s is not t:
				if t is s + 1:
					[trans, info] = process_one_rgbd_pair(
							s, t, color_files, depth_files)
					trans_odometry = np.dot(trans, trans_odometry)
					pose_graph.nodes.append(PoseGraphNode(trans_odometry))
					# todo: this should be t, s according to the definition
					pose_graph.edges.append(
							PoseGraphEdge(s, t, trans, info, False))
					print(pose_graph)

				if s % keyframes_per_n_frame is 0 \
						and t % keyframes_per_n_frame is 0:
					print([s,t])
					[trans, info] = process_one_rgbd_pair(
							s, t, color_files, depth_files)
					# todo: this should be t, s according to the definition
					pose_graph.edges.append(
							PoseGraphEdge(s, t, trans, info, True))
					print(pose_graph)
	return pose_graph


def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	# to display messages from GlobalOptimization
	SetVerbosityLevel(VerbosityLevel.Debug)
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	line_process_option = GlobalOptimizationLineProcessOption()
	pose_graph = ReadPoseGraph(pose_graph_name)
	GlobalOptimization(pose_graph, method, criteria, line_process_option)
	WritePoseGraph(pose_graph_optmized_name, pose_graph)
	SetVerbosityLevel(VerbosityLevel.Error)


def integration(pose_graph_name):
	intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault
	pose_graph = ReadPoseGraph(pose_graph_name)

	min_depth = 0.3
	cubic_length = 4.0
	volume = UniformTSDFVolume(length = cubic_length, resolution = 512,
			sdf_trunc = 0.04, with_color = True)
	trans = np.identity(4)
	trans[0:2,3] = cubic_length / 2
	trans[2,3] = -min_depth
	#print(trans)

	for i in range(len(pose_graph.nodes)):
		print("Integrate {:d}-th image into the volume.".format(i))
		color = ReadImage(color_files[i])
		depth = ReadImage(depth_files[i])
		rgbd = CreateRGBDImageFromColorAndDepth(color, depth, depth_trunc = 4.0,
				convert_rgb_to_intensity = False)
		pose = pose_graph.nodes[i].pose
		transformed_pose = np.dot(trans, pose)
		#print(transformed_pose)
		volume.Integrate(rgbd, intrinsic, transformed_pose)

	print("Extract a triangle mesh from the volume and visualize it.")
	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	return mesh
	#DrawGeometries([mesh])


def test_single_frame_integrate(i):
	min_depth = 0.3
	cubic_length = 4.0
	volume = UniformTSDFVolume(length = cubic_length, resolution = 512,
			sdf_trunc = 0.04, with_color = True)
	trans_offset = np.identity(4)
	trans_offset[0:2,3] = cubic_length / 2
	trans_offset[2,3] = -min_depth
	#print(trans_offset)

	print("Integrate {:d}-th image into the volume.".format(i))
	color = ReadImage(color_files[i])
	depth = ReadImage(depth_files[i])
	rgbd = CreateRGBDImageFromColorAndDepth(color, depth, depth_trunc = 4.0,
			convert_rgb_to_intensity = False)
	intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault
	volume.Integrate(rgbd, intrinsic, trans_offset)

	print("Extract a triangle mesh from the volume and visualize it.")
	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	mesh.Transform(np.linalg.inv(trans_offset))

	return mesh


def test_single_pair(s, t):

	SetVerbosityLevel(VerbosityLevel.Debug)

	if not exists(path_fragment):
		makedirs(path_fragment)

	# todo: need to test with and without OpenCV
	pose_graph = PoseGraph()
	[trans, info] = process_one_rgbd_pair(s, t, color_files, depth_files)
	pose_graph.nodes.append(PoseGraphNode(trans))
	pose_graph.nodes.append(PoseGraphNode(np.identity(4)))
	print('from process_one_rgbd_pair')

	# integration

	mesh_s = test_single_frame_integrate(s)
	mesh_t = test_single_frame_integrate(t)
	#mesh_s.Transform(np.linalg.inv(trans)) # for RGBD odometry
	mesh_s.Transform(trans) # for 5pt
	DrawGeometries([mesh_s, mesh_t])


# # test wide baseline matching
# if __name__ == "__main__":
#
# 	initialize_opencv()
# 	if opencv_installed:
# 		from opencv_pose_estimation import pose_estimation
#
# 	[color_files, depth_files] = get_flie_lists(path_dataset) #todo: is this global variable?
#
# 	# weird pairs 20-130
# 	s = 20
# 	t =	130
# 	test_single_pair(s, t)


if __name__ == "__main__":

	# check opencv python package
	initialize_opencv()
	if opencv_installed:
		from opencv_pose_estimation import pose_estimation

	if not exists(path_fragment):
		makedirs(path_fragment)

	[color_files, depth_files] = get_flie_lists(path_dataset) #todo: is this global variable?
	n_files = len(color_files)
	n_fragments = int(math.ceil(n_files / frames_per_fragment))

	#for fragment_id in range(n_fragments):
	for fragment_id in range(1):
		pose_graph_name = path_fragment + "fragments_%03d.json" % fragment_id
		#integration(pose_graph_name)
		pose_graph = make_one_fragment(fragment_id)
		WritePoseGraph(pose_graph_name, pose_graph)
		pose_graph_optmized_name = path_fragment + \
				"fragments_opt_%03d.json" % fragment_id
		optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
		mesh = integration(pose_graph_optmized_name)
		mesh_name = path_fragment + "fragment_%03d.ply" % fragment_id
		WriteTriangleMesh(mesh_name, mesh, False, True)
