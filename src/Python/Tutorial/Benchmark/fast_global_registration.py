import os
import sys
import struct
sys.path.append("../Utility")
sys.path.append("../ReconstructionSystem")
from global_registration import *
from redwood_dataset_trajectory_io import *
from visualization import *

# binary path to FGR
FGR_PATH = "/Users/jaesikpa/Research/FastGlobalRegistration/build/FastGlobalRegistration/FastGlobalRegistration"

# dataset from redwood-data.org
dataset_names = ["livingroom1", "livingroom2", "office1", "office2"]
dataset_path = "testdata/"

def write_binary_file_for_FGR(filename, pcd, fpfh):
	n_points = len(pcd.points)
	n_feature_dim = fpfh.data.shape[0]
	print("Writing %d points %d-dimentional features." % (n_points, n_feature_dim))
	fout = open(filename, 'wb')
	fout.write(struct.pack('i', n_points))
	fout.write(struct.pack('i', n_feature_dim))
	for i in range(n_points):
		for d in range(3):
			fout.write(struct.pack('f', float(pcd.points[i][d])))
		for d in range(n_feature_dim):
			fout.write(struct.pack('f', float(fpfh.data[d,i])))
	fout.close()

def get_full_bin_path(ply_file_names, i):
	return "%s.bin" % ply_file_names[i]

def get_full_txt_path(ply_file_names, i, j):
	return "%s/alignment_%s_%s.txt" % (ply_file_names, i, j)

def read_txt_file(txt_file_name):
	read_trajectory(filename)
	return transformation

def get_log_path(dataset_name):
	return "%s/fgr_%s.log" % (dataset_path, dataset_name)

# read files
if __name__ == "__main__":

	for dataset_name in dataset_names:
		ply_file_names = get_file_list(
				"%s/%s/" % (dataset_path, dataset_name), ".ply")
		n_ply_files = len(ply_file_names)

		# preprocessing
		for i in range(n_ply_files):
			(pcd_down, pcd_fpfh) = preprocess_point_cloud(ply_file_names[i])
			filename_i = get_full_bin_path(ply_file_names, i)
			write_binary_file_for_FGR(filename_i, pcd_down, pcd_fpfh)

		# global alignment
		traj = []
		for i in range(n_ply_files):
			filename_i = get_full_bin_path(ply_file_names, i)
			for j in range(i + 1, n_ply_files):
				filename_j = get_full_bin_path(ply_file_names, j)
				os.system("%s %s %s temp.log" %
						(FGR_PATH, filename_i, filename_j))
				traj_ij = read_trajectory("temp.log")
				traj_ij[0].metadata = [i,j,n_ply_files]
				traj.append(traj_ij[0])
				print(traj_ij[0])

				DrawRegistrationResult(source, target, transformation)

		write_trajectory(traj, get_log_path(dataset_name))
