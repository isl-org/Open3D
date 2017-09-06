import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import math
import sys
sys.path.append("../..")
from py3d import *
from utility import *
from make_fragments_from_rgbd import get_file_lists


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
	[success, trans, info] = ComputeRGBDOdometry(
			source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
			RGBDOdometryJacobianFromHybridTerm(), OdometryOption())
	return [trans, info]


def test_single_frame_integrate(i, intrinsic):
	min_depth = 0.3
	cubic_length = 4.0
	volume = UniformTSDFVolume(length = cubic_length, resolution = 512,
			sdf_trunc = 0.04, with_color = True)
	trans_offset = np.identity(4)
	trans_offset[0:2,3] = cubic_length / 2
	trans_offset[2,3] = -min_depth

	print("Integrate a rgbd image.")
	color = ReadImage(color_files[i])
	depth = ReadImage(depth_files[i])
	print(color_files[i])
	print(depth_files[i])
	rgbd = CreateRGBDImageFromColorAndDepth(color, depth, depth_trunc = 4.0,
			convert_rgb_to_intensity = False)
	volume.Integrate(rgbd, intrinsic, trans_offset)

	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	mesh.Transform(np.linalg.inv(trans_offset))

	return mesh


def test_single_pair(s, t, intrinsic, with_opencv):
	SetVerbosityLevel(VerbosityLevel.Debug)

	pose_graph = PoseGraph()
	[trans, info] = process_one_rgbd_pair(s, t,
			color_files, depth_files, intrinsic, with_opencv)
	pose_graph.nodes.append(PoseGraphNode(trans))
	pose_graph.nodes.append(PoseGraphNode(np.identity(4)))

	# integration
	mesh_s = test_single_frame_integrate(s, intrinsic)
	mesh_t = test_single_frame_integrate(t, intrinsic)
	mesh_s.Transform(trans) # for 5pt
	DrawGeometries([mesh_s, mesh_t])


# test wide baseline matching
if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset")
	path_intrinsic = parse_argument(sys.argv, "--path_intrinsic")
	source_id = parse_argument_int(sys.argv, "--source_id")
	target_id = parse_argument_int(sys.argv, "--target_id")

	if path_dataset:
		with_opencv = initialize_opencv()
		if with_opencv:
			from opencv_pose_estimation import pose_estimation

		[color_files, depth_files] = get_file_lists(path_dataset)
		if path_intrinsic:
			intrinsic = ReadPinholeCameraIntrinsic(path_intrinsic)
		else:
			intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault

		test_single_pair(source_id, target_id, intrinsic, with_opencv)
