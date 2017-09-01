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
n_frames_per_fragment = 100
n_keyframes_per_n_frame = 5

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
	# read images
	color_s = ReadImage(color_files[s])
	depth_s = ReadImage(depth_files[s])
	color_t = ReadImage(color_files[t])
	depth_t = ReadImage(depth_files[t])
	source_rgbd_image = CreateRGBDImageFromColorAndDepth(color_s, depth_s)
	target_rgbd_image = CreateRGBDImageFromColorAndDepth(color_t, depth_t)

	# initialize_camera_pose
	if abs(s-t) is not 1 and opencv_installed:
		odo_init = pose_estimation(source_rgbd_image, target_rgbd_image,
				pinhole_camera_intrinsic, False)
	else:
		odo_init = np.identity(4)

	# perform RGB-D odometry
	[success, trans, info] = ComputeRGBDOdometry(
			source_rgbd_image, target_rgbd_image,
			pinhole_camera_intrinsic, odo_init,
			RGBDOdometryJacobianFromHybridTerm(), OdometryOption())
	return [trans, info]


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
	sid = fragment_id * n_frames_per_fragment
	eid = sid + n_frames_per_fragment

	pose_graph = PoseGraph()
	trans_odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(trans_odometry))

	for s in range(sid, eid):
		for t in range(s+1, eid):
			# odometry
			if t is s + 1:
				#print([s,t])
				[trans, info] = process_one_rgbd_pair(
						s, t, color_files, depth_files)
				trans_odometry = np.dot(trans, trans_odometry)
				trans_odometry_inv = np.linalg.inv(trans_odometry)
				pose_graph.nodes.append(PoseGraphNode(trans_odometry_inv))
				pose_graph.edges.append(
						PoseGraphEdge(s, t, trans, info, False))
				#print(trans)
				print(pose_graph)

			# keyframe loop closure
			if s % n_keyframes_per_n_frame is 0 \
					and t % n_keyframes_per_n_frame is 0:
				#print([s,t])
				[trans, info] = process_one_rgbd_pair(
						s, t, color_files, depth_files)
				pose_graph.edges.append(
						PoseGraphEdge(s, t, trans, info, True))
				#print(trans)
				#print(info)
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


def integrate_rgb_frames(pose_graph_name):

	pose_graph = ReadPoseGraph(pose_graph_name)
	min_depth = 0.3
	cubic_length = 4.0
	volume = UniformTSDFVolume(length = cubic_length, resolution = 512,
			sdf_trunc = 0.04, with_color = True)
	trans = np.identity(4)
	trans[0:2,3] = cubic_length / 2
	trans[2,3] = -min_depth

	for i in range(len(pose_graph.nodes)):
		print("Integrate rgbd images %d/%d." % (i+1, len(pose_graph.nodes)))
		color = ReadImage(color_files[i])
		depth = ReadImage(depth_files[i])
		rgbd = CreateRGBDImageFromColorAndDepth(color, depth, depth_trunc = 4.0,
				convert_rgb_to_intensity = False)
		pose = pose_graph.nodes[i].pose
		transformed_pose = np.dot(trans, pose)
		volume.Integrate(rgbd, pinhole_camera_intrinsic, transformed_pose)

	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	return mesh

if __name__ == "__main__":

	print(len(sys.argv))
	if len(sys.argv) is 0:
		print('usage: ')
	else:
		if sys.argv is in "test_mode":
			print("testmode")

		# check opencv python package
		initialize_opencv()
		if opencv_installed:
			from opencv_pose_estimation import pose_estimation

		if not exists(path_fragment):
			makedirs(path_fragment)

		[color_files, depth_files] = get_flie_lists(path_dataset) 
		n_files = len(color_files)
		n_fragments = int(math.ceil(n_files / n_frames_per_fragment))

		pinhole_camera_intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault

		#for fragment_id in range(n_fragments):
		for fragment_id in range(1):
			pose_graph_name = path_fragment + "fragments_%03d.json" % fragment_id
			pose_graph = make_one_fragment(fragment_id)
			WritePoseGraph(pose_graph_name, pose_graph)
			pose_graph_optmized_name = path_fragment + \
					"fragments_opt_%03d.json" % fragment_id
			optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
			mesh = integrate_rgb_frames(pose_graph_optmized_name)
			mesh_name = path_fragment + "fragment_%03d.ply" % fragment_id
			WriteTriangleMesh(mesh_name, mesh, False, True)
