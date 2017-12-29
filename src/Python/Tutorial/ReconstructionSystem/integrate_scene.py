# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import argparse
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *

def scalable_integrate_rgb_frames(path_dataset, intrinsic):
	[color_files, depth_files] = get_file_lists(path_dataset)
	n_files = len(color_files)
	n_frames_per_fragment = 100
	n_fragments = n_files // n_frames_per_fragment + 1
	volume = ScalableTSDFVolume(voxel_length = 3.0 / 512.0, sdf_trunc = 0.04,\
            with_color = True)

	path_fragment = path_dataset + 'fragments/'
	global_pose_graph_name = path_fragment + "global_registration_optimized.json"
	global_pose_graph = read_pose_graph(global_pose_graph_name)

	for fragment_id in range(len(global_pose_graph.nodes)):
		pose_graph_name = path_fragment + "fragments_opt_%03d.json" % fragment_id
		pose_graph = read_pose_graph(pose_graph_name)

		for frame_id in range(len(pose_graph.nodes)):
			frame_id_abs = fragment_id * n_frames_per_fragment + frame_id
			print("Fragment %03d / %03d :: Integrate rgbd frame %d (%d of %d)."
					% (fragment_id, n_fragments-1, frame_id_abs, frame_id+1,
					len(pose_graph.nodes)))
			color = read_image(color_files[frame_id_abs])
			depth = read_image(depth_files[frame_id_abs])
			rgbd = create_rgbd_image_from_color_and_depth(color, depth,
					depth_trunc = 3.0, convert_rgb_to_intensity = False)
			pose = np.dot(global_pose_graph.nodes[fragment_id].pose,
					pose_graph.nodes[frame_id].pose)
			volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

	mesh = volume.extract_triangle_mesh()
	mesh.compute_vertex_normals()
	draw_geometries([mesh])
	return mesh


if __name__ == "__main__":
	parser = argparse.ArgumentParser(description='integrate whole scene from RGBD sequence.')
	parser.add_argument('path_dataset', help='path to the dataset')
	parser.add_argument('-path_intrinsic', help='path to the RGBD camera intrinsic')
	args = parser.parse_args()

	if args.path_intrinsic:
		intrinsic = ReadPinholeCameraIntrinsic(args.path_intrinsic)
	else:
		intrinsic = PinholeCameraIntrinsic.prime_sense_default

	mesh = scalable_integrate_rgb_frames(args.path_dataset, intrinsic)
	mesh_name = args.path_dataset + "integrated.ply"
	print("Saving mesh as %s" % mesh_name)
	write_triangle_mesh(mesh_name, mesh, False, True)
