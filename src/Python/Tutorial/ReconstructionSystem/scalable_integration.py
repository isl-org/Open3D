import numpy as np
import math
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from visualization import *

def scalable_integrate_rgb_frames(path_dataset, intrinsic):
	[color_files, depth_files] = get_file_lists(path_dataset)
	n_files = len(color_files)
	n_frames_per_fragment = 100
	n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))
	volume = ScalableTSDFVolume(voxel_length = 4.0 / 512.0, sdf_trunc = 0.04,\
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
			rgbd = CreateRGBDImageFromColorAndDepth(color, depth,
					depth_trunc = 4.0, convert_rgb_to_intensity = False)
			pose = np.dot(global_pose_graph.nodes[fragment_id].pose,
					pose_graph.nodes[frame_id].pose)
			volume.integrate(rgbd, intrinsic, np.linalg.inv(pose))

	mesh = volume.ExtractTriangleMesh()
	mesh.ComputeVertexNormals()
	DrawGeometries([mesh])
	return mesh


if __name__ == "__main__":
	path_dataset = parse_argument(sys.argv, "--path_dataset")
	path_intrinsic = parse_argument(sys.argv, "--path_intrinsic")
	if not path_dataset:
		print("usage : %s " % sys.argv[0])
		print("  --path_dataset [path]   : Path to rgbd_dataset. Mandatory.")
		print("  --path_intrinsic [path] : Path to json camera intrinsic file. Optional.")
		sys.exit()

	if path_intrinsic:
		intrinsic = ReadPinholeCameraIntrinsic(path_intrinsic)
	else:
		intrinsic = PinholeCameraIntrinsic.PrimeSenseDefault

	mesh = scalable_integrate_rgb_frames(path_dataset, intrinsic)
	mesh_name = path_dataset + "integrated.ply"
	print("Saving mesh as %s" % mesh_name)
	WriteTriangleMesh(mesh_name, mesh, False, True)
