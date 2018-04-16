# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import os
import sys
sys.path.append("../Utility")
sys.path.append("../ReconstructionSystem")
import numpy as np # todo: this is just for testing
from global_registration import *
from downloader import *
from redwood_dataset_trajectory_io import *

do_visualization = False


def get_ply_path(dataset_name, id):
	return "%s/%s/cloud_bin_%d.ply" % (dataset_path, dataset_name, id)


def get_log_path(dataset_name):
	return "%s/ransac_%s.log" % (dataset_path, dataset_name)


if __name__ == "__main__":
	# data preparation
	get_redwood_dataset()

	# do RANSAC based alignment
	for dataset_name in dataset_names:
		ply_file_names = get_file_list_from_custom_format(
				"%s/%s/" % (dataset_path, dataset_name), "cloud_bin_%d.ply")
		n_ply_files = len(ply_file_names)

		alignment = []
		for s in range(n_ply_files):
			for t in range(s + 1, n_ply_files):
				(source_down, source_fpfh) = \
						preprocess_point_cloud(get_ply_path(dataset_name, s))
				(target_down, target_fpfh) = \
						preprocess_point_cloud(get_ply_path(dataset_name, t))
				print("preprocessing done. points %d %d" % \
						(len(source_down.points), len(target_down.points)))

				(success_ransac, result_ransac) = register_point_cloud_FPFH(
						source_down, target_down, source_fpfh, target_fpfh)
				# Note: we save inverse of result_ransac.transformation
				# to comply with http://redwood-data.org/indoor/fileformat.html
				if not success_ransac:
					print("No resonable solution.")
				else:
					alignment.append(CameraPose([s, t, n_ply_files],
							np.linalg.inv(result_ransac.transformation)))
					print(np.linalg.inv(result_ransac.transformation))

				if do_visualization:
					draw_registration_result(source_down, target_down,
							result_ransac.transformation)
		write_trajectory(alignment, get_log_path(dataset_name))

	# do evaluation
