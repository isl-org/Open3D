import numpy as np
import sys
sys.path.append("../..")
sys.path.append("../Utility")
from py3d import *
from common import *
from visualization import *
from global_registration import *


def register_point_cloud_pairwise(path_dataset, ply_file_names,
		source_id, target_id, transformation_init = np.identity(4),
		feature_matching = True,
		registration_type = "color", draw_result = True):

	(source_down, source_fpfh) = preprocess_point_cloud(
			ply_file_names[source_id])
	(target_down, target_fpfh) = preprocess_point_cloud(
			ply_file_names[target_id])

	if feature_matching:
		print("Do feature matching")
		(success_ransac, result_ransac) = register_point_cloud_FPFH(
				source_down, target_down,
				source_fpfh, target_fpfh)
		if not success_ransac:
			print("No resonable solution for initial pose.")
		else:
			transformation_init = result_ransac.transformation
			print(transformation_init)
	if draw_result:
		draw_registration_result(source_down, target_down,
				transformation_init)

	if (registration_type == "color"):
		print("RegistrationPointCloud - color ICP")
		(transformation_icp, information_icp) = \
				register_colored_point_cloud_ICP(
				source_down, target_down, transformation_init)
	else:
		print("RegistrationPointCloud - ICP")
		(transformation_icp, information_icp) = \
				register_point_cloud_ICP(
				source_down, target_down, transformation_init)
	if draw_result:
		DrawRegistrationResultOriginalColor(source_down, target_down,
				transformation_icp)


if __name__ == "__main__":
	set_verbosity_level(VerbosityLevel.Debug)
	path_dataset = parse_argument(sys.argv, "--path_dataset") # todo use argparse
	path_init = parse_argument(sys.argv, "--init_pose")
	source_id = parse_argument_int(sys.argv, "--source_id")
	target_id = parse_argument_int(sys.argv, "--target_id")
	if not path_dataset or not source_id or not target_id:
		print("usage : %s " % sys.argv[0])
		print("  --path_dataset [path]   : Path to the dataset. Mandatory.")
		print("  --source_id [id]        : ID of source point cloud. Mandatory.")
		print("  --target_id [id]        : ID of target point cloud. Mandatory.")
		print("  --path_init [id]        : Path of initial pose [4x4]. Optional.")
		sys.exit()

	ply_file_names = get_file_list(path_dataset + "/fragments/", ".ply")
	if not path_init:
		register_point_cloud_pairwise(path_dataset, ply_file_names,
				source_id, target_id)
	else:
		transformation_init = np.loadtxt(path_init)
		register_point_cloud_pairwise(path_dataset, ply_file_names,
				source_id, target_id, transformation_init)
