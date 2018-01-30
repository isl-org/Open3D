# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import argparse
import math
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from opencv import *
from optimize_posegraph import *


def register_one_rgbd_pair(s, t, color_files, depth_files,
		intrinsic, with_opencv):
	# read images
	color_s = read_image(color_files[s])
	depth_s = read_image(depth_files[s])
	color_t = read_image(color_files[t])
	depth_t = read_image(depth_files[t])
	source_rgbd_image = create_rgbd_image_from_color_and_depth(color_s, depth_s,
			depth_trunc = 3.0, convert_rgb_to_intensity = True)
	target_rgbd_image = create_rgbd_image_from_color_and_depth(color_t, depth_t,
			depth_trunc = 3.0, convert_rgb_to_intensity = True)

	if abs(s-t) is not 1:
		if with_opencv:
			success_5pt, odo_init = pose_estimation(
					source_rgbd_image, target_rgbd_image, intrinsic, False)
			if success_5pt:
				[success, trans, info] = compute_rgbd_odometry(
						source_rgbd_image, target_rgbd_image, intrinsic,
						odo_init, RGBDOdometryJacobianFromHybridTerm(),
						OdometryOption())
				return [success, trans, info]
		return [False, np.identity(4), np.identity(6)]
	else:
		odo_init = np.identity(4)
		[success, trans, info] = compute_rgbd_odometry(
				source_rgbd_image, target_rgbd_image, intrinsic, odo_init,
				RGBDOdometryJacobianFromHybridTerm(), OdometryOption())
		return [success, trans, info]


def make_posegraph_for_fragment(path_dataset, sid, eid, color_files, depth_files,
		fragment_id, n_fragments, intrinsic, with_opencv):
	set_verbosity_level(VerbosityLevel.Error)
	pose_graph = PoseGraph()
	trans_odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(trans_odometry))
	for s in range(sid, eid):
		for t in range(s + 1, eid):
			# odometry
			if t == s + 1:
				print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
						% (fragment_id, n_fragments-1, s, t))
				[success, trans, info] = register_one_rgbd_pair(
						s, t, color_files, depth_files, intrinsic, with_opencv)
				trans_odometry = np.dot(trans, trans_odometry)
				trans_odometry_inv = np.linalg.inv(trans_odometry)
				pose_graph.nodes.append(PoseGraphNode(trans_odometry_inv))
				pose_graph.edges.append(
						PoseGraphEdge(s-sid, t-sid, trans, info,
								uncertain = False))

			# keyframe loop closure
			if s % n_keyframes_per_n_frame == 0 \
					and t % n_keyframes_per_n_frame == 0:
				print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
						% (fragment_id, n_fragments-1, s, t))
				[success, trans, info] = register_one_rgbd_pair(
						s, t, color_files, depth_files, intrinsic, with_opencv)
				if success:
					pose_graph.edges.append(
							PoseGraphEdge(s-sid, t-sid, trans, info,
									uncertain = True))
	write_pose_graph(path_dataset + template_fragment_posegraph % fragment_id,
			pose_graph)


def integrate_rgb_frames_for_fragment(color_files, depth_files,
		fragment_id, n_fragments, pose_graph_name, intrinsic):
	pose_graph = read_pose_graph(pose_graph_name)
	volume = ScalableTSDFVolume(voxel_length = 3.0 / 512.0,
			sdf_trunc = 0.04, with_color = True)

	for i in range(len(pose_graph.nodes)):
		i_abs = fragment_id * n_frames_per_fragment + i
		print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)."
				% (fragment_id, n_fragments-1,
				i_abs, i+1, len(pose_graph.nodes)))
		color = read_image(color_files[i_abs])
		depth = read_image(depth_files[i_abs])
		rgbd = create_rgbd_image_from_color_and_depth(color, depth,
				depth_trunc = 3.0, convert_rgb_to_intensity = False)
		pose = pose_graph.nodes[i].pose
		volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	return mesh


def make_mesh_for_fragment(path_dataset, color_files, depth_files,
		fragment_id, n_fragments, intrinsic):
	mesh = integrate_rgb_frames_for_fragment(
			color_files, depth_files, fragment_id, n_fragments,
			path_dataset + template_fragment_posegraph_optimized % fragment_id,
			intrinsic)
	mesh_name = path_dataset + template_fragment_mesh % fragment_id
	write_triangle_mesh(mesh_name, mesh, False, True)


def process_fragments(path_dataset, path_intrinsic):
	if path_intrinsic:
		intrinsic = read_pinhole_camera_intrinsic(path_intrinsic)
	else:
		intrinsic = PinholeCameraIntrinsic.prime_sense_default

	make_folder(path_dataset + folder_fragment)
	[color_files, depth_files] = get_rgbd_file_lists(path_dataset)
	n_files = len(color_files)
	n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))

	for fragment_id in range(n_fragments):
		sid = fragment_id * n_frames_per_fragment
		eid = min(sid + n_frames_per_fragment, n_files)
		make_posegraph_for_fragment(path_dataset, sid, eid, color_files, depth_files,
				fragment_id, n_fragments, intrinsic, with_opencv)
		optimize_posegraph_for_fragment(path_dataset, fragment_id)
		make_mesh_for_fragment(path_dataset, color_files, depth_files,
				fragment_id, n_fragments, intrinsic)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
			description="making fragments from RGBD sequence.")
	parser.add_argument("path_dataset", help="path to the dataset")
	parser.add_argument("-path_intrinsic",
			help="path to the RGBD camera intrinsic")
	args = parser.parse_args()

	# check opencv python package
	with_opencv = initialize_opencv()
	if with_opencv:
		from opencv_pose_estimation import pose_estimation
	process_fragments(args.path_dataset, args.path_intrinsic)
