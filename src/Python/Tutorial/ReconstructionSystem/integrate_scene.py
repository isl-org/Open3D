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

def scalable_integrate_rgb_frames(path_dataset, intrinsic):
	[color_files, depth_files] = get_rgbd_file_lists(path_dataset)
	n_files = len(color_files)
	n_frames_per_fragment = 100
	n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))
	volume = ScalableTSDFVolume(voxel_length = 3.0 / 512.0,
			sdf_trunc = 0.04, with_color = True)

	pose_graph_fragment = read_pose_graph(
			path_dataset + template_global_posegraph_optimized)

	for fragment_id in range(len(pose_graph_fragment.nodes)):
		pose_graph_rgbd = read_pose_graph(path_dataset +
				template_fragment_posegraph_optimized % fragment_id)

		for frame_id in range(len(pose_graph_rgbd.nodes)):
			frame_id_abs = fragment_id * n_frames_per_fragment + frame_id
			print("Fragment %03d / %03d :: integrate rgbd frame %d (%d of %d)."
					% (fragment_id, n_fragments-1, frame_id_abs, frame_id+1,
					len(pose_graph_rgbd.nodes)))
			color = read_image(color_files[frame_id_abs])
			depth = read_image(depth_files[frame_id_abs])
			rgbd = create_rgbd_image_from_color_and_depth(color, depth,
					depth_trunc = 3.0, convert_rgb_to_intensity = False)
			pose = np.dot(pose_graph_fragment.nodes[fragment_id].pose,
					pose_graph_rgbd.nodes[frame_id].pose)
			volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	draw_geometries([mesh])

	mesh_name = path_dataset + template_global_mesh
	write_triangle_mesh(mesh_name, mesh, False, True)


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description=
			"integrate the whole RGBD sequence using estimated camera pose")
	parser.add_argument("path_dataset", help="path to the dataset")
	parser.add_argument("-path_intrinsic",
			help="path to the RGBD camera intrinsic")
	args = parser.parse_args()

	if args.path_intrinsic:
		intrinsic = read_pinhole_camera_intrinsic(args.path_intrinsic)
	else:
		intrinsic = PinholeCameraIntrinsic.prime_sense_default
	scalable_integrate_rgb_frames(args.path_dataset, intrinsic)
