import numpy as np
from os import listdir
from os.path import isfile, join
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
	print([s,t])

	# read images
	color_s = ReadImage(color_files[s])
	depth_s = ReadImage(depth_files[s])
	color_t = ReadImage(color_files[t])
	depth_t = ReadImage(depth_files[t])
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
		return [trans, info]
	else:
		return [np.identity(4), np.zeros(6)]


def get_flie_lists(path_dataset):
	# get list of color and depth images
	path_color = path_dataset + 'image/'
	path_depth = path_dataset + 'depth/'
	color_files = [path_color + f for f in listdir(path_color) if isfile(join(path_color, f))]
	depth_files = [path_depth + f for f in listdir(path_depth) if isfile(join(path_depth, f))]
	return color_files, depth_files


def make_one_fragment(fragment_id):
	pose_graph = PoseGraph()
	print(pose_graph)
	# odometry
	for s in range(frames_per_fragment-1): # fragment name?
		t = s + 1
		[trans, info] = process_one_rgbd_pair(s, t, color_files, depth_files)
		pose_graph.nodes.append(PoseGraphNode(trans))
		print(pose_graph)

	# keyframe loop closure
	for s in range(frames_per_fragment):
		for t in range(frames_per_fragment):
			if s is not t \
				and s % keyframes_per_n_frame == 0 \
				and t % keyframes_per_n_frame == 0:
					[trans, info] = process_one_rgbd_pair(
							s, t, color_files, depth_files)
					pose_graph.edges.append(
							PoseGraphEdge(t, s, trans, info, False))
					print(pose_graph)
	return pose_graph


def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	line_process_option = GlobalOptimizationLineProcessOption()
	pose_graph = ReadPoseGraph(pose_graph_name)
	print(pose_graph_fragment)
	GlobalOptimization(pose_graph_fragment, method, criteria, line_process_option)
	WritePoseGraph(pose_graph_optmized_name, pose_graph_fragment)


if __name__ == "__main__":
	# check opencv python package
	initialize_opencv()
	if opencv_installed:
		from opencv_pose_estimation import pose_estimation

	[color_files, depth_files] = get_flie_lists(path_dataset)
	n_files = len(color_files)
	n_fragments = int(math.ceil(n_files / frames_per_fragment))

	#for fragment_id in range(n_fragments):
	for fragment_id in range(1):
		pose_graph_name = "fragments_%03d.json" % fragment_id
		pose_graph = make_one_fragment(fragment_id)
		WritePoseGraph(pose_graph_name, pose_graph)
		pose_graph_optmized_name = "fragments_opt_%03d.json" % fragment_id
		optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
