import numpy as np
from os import makedirs
from os.path import exists
import math
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from opencv import *
from optimize_posegraph import *


def process_one_rgbd_pair(s, t, color_files, depth_files,
		intrinsic, with_opencv):
	# read images
	color_s = read_image(color_files[s])
	depth_s = read_image(depth_files[s])
	color_t = read_image(color_files[t])
	depth_t = read_image(depth_files[t])
	source_rgbd_image = create_rgbd_image_from_color_and_depth(color_s, depth_s,
			depth_trunc = 4.0, convert_rgb_to_intensity = True)
	target_rgbd_image = create_rgbd_image_from_color_and_depth(color_t, depth_t,
			depth_trunc = 4.0, convert_rgb_to_intensity = True)

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


def make_one_fragment(fragment_id, intrinsic, with_opencv):
	set_verbosity_level(VerbosityLevel.Error)
	sid = fragment_id * n_frames_per_fragment
	eid = min(sid + n_frames_per_fragment, n_files)

	pose_graph = PoseGraph()
	trans_odometry = np.identity(4)
	pose_graph.nodes.append(PoseGraphNode(trans_odometry))

	for s in range(sid, eid):
		for t in range(s + 1, eid):
			# odometry
			if t == s + 1:
				print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
				 		% (fragment_id, n_fragments-1, s, t))
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
				print("Fragment %03d / %03d :: RGBD matching between frame : %d and %d"
				 		% (fragment_id, n_fragments-1, s, t))
				[success, trans, info] = process_one_rgbd_pair(
						s, t, color_files, depth_files, intrinsic, with_opencv)
				if success:
					pose_graph.edges.append(
							PoseGraphEdge(s-sid, t-sid, trans, info, True))
	return pose_graph


def integrate_rgb_frames(fragment_id, pose_graph_name, intrinsic):
	pose_graph = read_pose_graph(pose_graph_name)
	volume = ScalableTSDFVolume(voxel_length = 4.0 / 512.0, sdf_trunc = 0.04,\
			with_color = True)

	for i in range(len(pose_graph.nodes)):
		i_abs = fragment_id * n_frames_per_fragment + i
		print("Fragment %03d / %03d :: Integrate rgbd frame %d (%d of %d)."
				% (fragment_id, n_fragments-1, i_abs, i+1, len(pose_graph.nodes)))
		color = read_image(color_files[i_abs])
		depth = read_image(depth_files[i_abs])
		rgbd = create_rgbd_image_from_color_and_depth(color, depth, depth_trunc = 4.0,
				convert_rgb_to_intensity = False)
		pose = pose_graph.nodes[i].pose
		volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	return mesh

if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset")
	path_intrinsic = parse_argument(sys.argv, "--path_intrinsic")
	if not path_dataset:
		print("usage : %s " % sys.argv[0])
		print("  --path_dataset [path]   : Path to rgbd_dataset. Mandatory.")
		print("  --path_intrinsic [path] : Path to json camera intrinsic file. Optional.")
		sys.exit()

	# some global parameters
	n_frames_per_fragment = 100
	n_keyframes_per_n_frame = 5

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
		intrinsic = read_pinhole_camera_intrinsic(path_intrinsic)
	else:
		intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault

	for fragment_id in range(n_fragments):
		pose_graph_name = path_fragment + "fragments_%03d.json" % fragment_id
		pose_graph = make_one_fragment(fragment_id, intrinsic, with_opencv)
		write_pose_graph(pose_graph_name, pose_graph)
		pose_graph_optmized_name = path_fragment + \
				"fragments_opt_%03d.json" % fragment_id
		optimize_posegraph(pose_graph_name, pose_graph_optmized_name)
		mesh = integrate_rgb_frames(
				fragment_id, pose_graph_optmized_name, intrinsic)
		mesh_name = path_fragment + "fragment_%03d.ply" % fragment_id
		write_triangle_mesh(mesh_name, mesh, False, True)
