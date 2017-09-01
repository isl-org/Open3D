import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, exists
import math
import sys
sys.path.append("../..")
from py3d import *


def test_single_frame_integrate(i):
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
	volume.Integrate(rgbd, pinhole_camera_intrinsic, trans_offset)

	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	mesh.Transform(np.linalg.inv(trans_offset))

	return mesh


def test_single_pair(s, t):

	SetVerbosityLevel(VerbosityLevel.Debug)

	pose_graph = PoseGraph()
	[trans, info] = process_one_rgbd_pair(s, t, color_files, depth_files)
	pose_graph.nodes.append(PoseGraphNode(trans))
	pose_graph.nodes.append(PoseGraphNode(np.identity(4)))

	# integration
	mesh_s = test_single_frame_integrate(s)
	mesh_t = test_single_frame_integrate(t)
	mesh_s.Transform(trans) # for 5pt
	DrawGeometries([mesh_s, mesh_t])


# test wide baseline matching
if __name__ == "__main__":

	initialize_opencv()
	if opencv_installed:
		from opencv_pose_estimation import pose_estimation

	[color_files, depth_files] = get_flie_lists(path_dataset) 
	pinhole_camera_intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault

	s = 20
	t =	130
	test_single_pair(s, t)
