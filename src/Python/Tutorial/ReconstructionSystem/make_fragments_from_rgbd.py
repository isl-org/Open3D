import numpy as np
from os import makedirs
from os.path import exists
import math
import sys
sys.path.append("../..")
from py3d import *
from utility import *
import copy


def process_one_rgbd_pair(s, t, color_files, depth_files,
		intrinsic, with_opencv):
	# read images
	color_s = ReadImage(color_files[s])
	depth_s = ReadImage(depth_files[s])
	color_t = ReadImage(color_files[t])
	depth_t = ReadImage(depth_files[t])
	source_rgbd_image = CreateRGBDImageFromColorAndDepth(color_s, depth_s)
	target_rgbd_image = CreateRGBDImageFromColorAndDepth(color_t, depth_t)

	# initialize_camera_pose
	if abs(s-t) is not 1 and with_opencv:
		odo_init = pose_estimation(
				source_rgbd_image, target_rgbd_image, intrinsic, False)
	else:
		odo_init = np.identity(4)

	# perform RGB-D odometry
	option = OdometryOption(max_depth_diff = max_correspondence_distance)
	[success, trans, info] = ComputeRGBDOdometry(
			source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
			RGBDOdometryJacobianFromHybridTerm(), option)
	return [success, trans, info]


def get_file_lists(path_dataset):
	# get list of color and depth images
	path_color = path_dataset + 'image/'
	path_depth = path_dataset + 'depth/'
	color_files = get_file_list(path_color)
	depth_files = get_file_list(path_depth)
	return color_files, depth_files


def make_one_fragment(fragment_id, intrinsic, with_opencv):
	#SetVerbosityLevel(VerbosityLevel.Error)
	SetVerbosityLevel(VerbosityLevel.Warning)
	sid = fragment_id * n_frames_per_fragment
	eid = min(sid + n_frames_per_fragment, n_files)

	pose_graph = PoseGraph()
	trans_odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(trans_odometry))

	for s in range(sid, eid):
		for t in range(s + 1, eid):
			# odometry
			if t == s + 1:
				print("Fragment [%03d/%03d] :: RGBD matching between frame : %d and %d"
				 		% (fragment_id, n_fragments, s, t))
				[success, trans, info] = process_one_rgbd_pair(
						s, t, color_files, depth_files, intrinsic, with_opencv)
				trans_odometry = np.dot(trans, trans_odometry)
				trans_odometry_inv = np.linalg.inv(trans_odometry)
				pose_graph.nodes.append(PoseGraphNode(trans_odometry_inv))
				pose_graph.edges.append(
						PoseGraphEdge(s-sid, t-sid, trans, info, False))

			# keyframe loop closure
			if s % n_keyframes_per_n_frame == 0 \
					and t % n_keyframes_per_n_frame == 0:
				print("Fragment [%03d/%03d] :: RGBD matching between frame : %d and %d"
				 		% (fragment_id, n_fragments, s, t))
				[success, trans, info] = process_one_rgbd_pair(
						s, t, color_files, depth_files, intrinsic, with_opencv)
				if success:
					pose_graph.edges.append(
							PoseGraphEdge(s-sid, t-sid, trans, info, True))
	return pose_graph


def optimize_posegraph(pose_graph_name, pose_graph_optmized_name):
	# to display messages from GlobalOptimization
	SetVerbosityLevel(VerbosityLevel.Debug)
	pose_graph = ReadPoseGraph(pose_graph_name)

	method = GlobalOptimizationLevenbergMarquardt()
	criteria = GlobalOptimizationConvergenceCriteria()
	line_process_option = GlobalOptimizationLineProcessOption(
			max_correspondence_distance = max_correspondence_distance)
	print(line_process_option)

	GlobalOptimization(pose_graph, method, criteria, line_process_option)
	WritePoseGraph(pose_graph_optmized_name, pose_graph)
	SetVerbosityLevel(VerbosityLevel.Error)


def integrate_rgb_frames(fragment_id, pose_graph_name, intrinsic):
	pose_graph = ReadPoseGraph(pose_graph_name)
	min_depth = 0.3
	cubic_length = 4.0
	volume = UniformTSDFVolume(length = cubic_length, resolution = 512,
			sdf_trunc = 0.04, with_color = True)
	trans = np.identity(4)
	trans[0:2,3] = cubic_length / 2
	trans[2,3] = -min_depth

	for i in range(len(pose_graph.nodes)):
		i_abs = fragment_id * n_frames_per_fragment + i
		print("Fragment [%03d/%03d] :: Integrate rgbd frame %d (%d of %d)."
				% (fragment_id, n_fragments, i_abs, i+1, len(pose_graph.nodes)))
		color = ReadImage(color_files[i_abs])
		depth = ReadImage(depth_files[i_abs])
		rgbd = CreateRGBDImageFromColorAndDepth(color, depth, depth_trunc = 4.0,
				convert_rgb_to_intensity = False)
		pose = pose_graph.nodes[i].pose
		transformed_pose = np.dot(trans, pose)
		volume.Integrate(rgbd, intrinsic, transformed_pose)

	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	return mesh

if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset")
	path_intrinsic = parse_argument(sys.argv, "--path_intrinsic")

	if path_dataset:

		# some global parameters
		n_frames_per_fragment = 100
		n_keyframes_per_n_frame = 5
		max_correspondence_distance = 0.07

		# check opencv python package
		with_opencv = initialize_opencv()
		if with_opencv:
			from opencv_pose_estimation import pose_estimation

		path_fragment = path_dataset + 'fragments/'
		if not exists(path_fragment):
			makedirs(path_fragment)

		[color_files, depth_files] = get_file_lists(path_dataset)
		n_files = len(color_files)
		n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))

		if path_intrinsic:
			intrinsic = ReadPinholeCameraIntrinsic(path_intrinsic)
		else:
			intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault

		for fragment_id in range(n_fragments):
		#for fragment_id in [12]:
			pose_graph_name = path_fragment + "fragments_%03d.json" % fragment_id
			pose_graph = make_one_fragment(fragment_id, intrinsic, with_opencv)
			WritePoseGraph(pose_graph_name, pose_graph)
			pose_graph_optmized_name = path_fragment + \
					"fragments_opt_%03d.json" % fragment_id
			optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
			mesh = integrate_rgb_frames(
					fragment_id, pose_graph_optmized_name, intrinsic)
			mesh_name = path_fragment + "fragment_%03d.ply" % fragment_id
			WriteTriangleMesh(mesh_name, mesh, False, True)
