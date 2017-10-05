import os
import sys
sys.path.append("../Utility")
sys.path.append("../ReconstructionSystem")
import numpy as np # todo: this is just for testing
from global_registration import *
from utility_download import *
from utility_redwood_dataset_trajectory_io import *

# dataset from redwood-data.org
dataset_names = ["livingroom1", "livingroom2", "office1", "office2"]
dataset_path = "testdata/"
do_visualization = False


def get_dataset():
	# download and unzip dataset
	for name in dataset_names:
		print("==================================")
		file_downloader("http://redwood-data.org/indoor/data/%s-fragments-ply.zip" % \
				name)
		unzip_data("%s-fragments-ply.zip" % name,
				"%s/%s" % (dataset_path, name))
		os.remove("%s-fragments-ply.zip" % name)
		print("")
	return 0


def get_ply_path(dataset_name, id):
	return "%s/%s/cloud_bin_%d.ply" % (dataset_path, dataset_name, id)


def get_log_path(dataset_name):
	return "%s/%s.log" % (dataset_path, dataset_name)


if __name__ == "__main__":
	SetVerbosityLevel(VerbosityLevel.Debug)

	# data preparation
	if not os.path.exists(dataset_path):
		get_dataset()

	# do RANSAC based alignment
	# for dataset_name in dataset_names:
	for dataset_name in dataset_names:
		ply_file_names = get_file_list(
				"%s/%s/" % (dataset_path, dataset_name), ".ply")
		n_ply_files = len(ply_file_names)

		alignment = []
		for s in range(n_ply_files):
			for t in range(s + 1, n_ply_files):
				(source_down, source_fpfh) = \
						preprocess_point_cloud(get_ply_path(dataset_name, s))
				(target_down, target_fpfh) = \
						preprocess_point_cloud(get_ply_path(dataset_name, t))

				result_ransac = register_point_cloud_FPFH(
						source_down, target_down, source_fpfh, target_fpfh)
				# Push only successful transformations. Ignoring identity
				# Note: we save inverse of result_ransac.transformation
				# to comply with http://redwood-data.org/indoor/fileformat.html
				if (np.trace(result_ransac.transformation) != 4.0):
					print("[Successfully found transformation]")
					alignment.append(CameraPose([s, t, n_ply_files],
							np.linalg.inv(result_ransac.transformation)))
				else:
					print("[No solution]")
				print("")

				if do_visualization:
					DrawRegistrationResult(source_down, target_down,
							result_ransac.transformation)
		write_trajectory(alignment, get_log_path(dataset_name))

	# do evaluation
