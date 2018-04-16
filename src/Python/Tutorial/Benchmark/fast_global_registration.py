# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import os
import sys
import struct
import math
sys.path.append("../Utility")
sys.path.append("../ReconstructionSystem")
from global_registration import *
from downloader import *
from redwood_dataset_trajectory_io import *
from visualization import *

# binary path to FGR
FGR_PATH = "C:/git/FastGlobalRegistration/build/FastGlobalRegistration/Release/FastGlobalRegistration.exe"
FGR_INLIER_RATIO = 0.3
FGR_MAXIUM_DISTANCE = 0.075

def write_binary_file_for_FGR(filename, pcd, fpfh):
	n_points = len(pcd.points)
	n_feature_dim = fpfh.data.shape[0]
	print("Writing %d points and %d-dimentional features." % (n_points, n_feature_dim))
	fout = open(filename, 'wb')
	fout.write(struct.pack('i', n_points))
	fout.write(struct.pack('i', n_feature_dim))
	for i in range(n_points):
		for d in range(3):
			fout.write(struct.pack('f', float(pcd.points[i][d])))
		for d in range(n_feature_dim):
			fout.write(struct.pack('f', float(fpfh.data[d,i])))
	fout.close()

def read_binary_file_for_FGR(filename):
	fin = open(filename, 'rb')
	n_points = struct.unpack('i', fin.read(4))[0]
	n_feature_dim = struct.unpack('i', fin.read(4))[0]
	print("Reading %d points and %d-dimentional features." % (n_points, n_feature_dim))
	vec_point = []
	vec_feature = []
	for i in range(n_points):
		point = [0,0,0]
		for d in range(3):
			point[d] = struct.unpack('f', fin.read(4))[0]
		fin.read(4*n_feature_dim)
		vec_point.append(point)
	fin.close()
	pcd = PointCloud()
	pcd.points = Vector3dVector(vec_point)
	return pcd

def validating_transform_swap(pcd_i, pcd_j, transform):
	if len(pcd_i.points) > len(pcd_j.points):
		return validating_transform(pcd_j, pcd_i, np.linalg.inv(transform))
	else:
		return validating_transform(pcd_i, pcd_j, transform)

def validating_transform(pcd_i, pcd_j, transform):
	pcd_i_trans = copy.deepcopy(pcd_i)
	pcd_i_trans.transform(transform)
	tree_pcd_j = KDTreeFlann(pcd_j)
	inlier_number = 0
	for i in range(len(pcd_i.points)):
		[_, idx, dis] = tree_pcd_j.search_knn_vector_3d(pcd_i_trans.points[i], 1)
		if math.sqrt(dis[0]) < FGR_MAXIUM_DISTANCE:
			inlier_number = inlier_number + 1
	inlier_ratio = inlier_number / len(pcd_i.points);
	print("inlier_ratio : %f" % inlier_ratio)
	if inlier_ratio > FGR_INLIER_RATIO:
		return True
	else:
		return False

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
	# data preparation
	get_redwood_dataset()

	for dataset_name in dataset_names:
		ply_file_names = get_file_list_from_custom_format(
				"%s/%s/" % (dataset_path, dataset_name), "cloud_bin_%d.ply")
		n_ply_files = len(ply_file_names)

		# preprocessing
		for i in range(n_ply_files):
			filename_i = get_full_bin_path(ply_file_names, i)
			if not os.path.exists(filename_i):
				(pcd_down, pcd_fpfh) = preprocess_point_cloud(ply_file_names[i])
				write_binary_file_for_FGR(filename_i, pcd_down, pcd_fpfh)

		# global alignment
		traj = []
		for i in range(n_ply_files):
			filename_i = get_full_bin_path(ply_file_names, i)
			for j in range(i + 1, n_ply_files):
				filename_j = get_full_bin_path(ply_file_names, j)
				print(filename_i)
				print(filename_j)
				command = "%s %s %s temp.log" % \
						(FGR_PATH, filename_i, filename_j)
				print(command)
				os.system(command)
				traj_ij = read_trajectory("temp.log")
				if (traj_ij[0].pose.trace() == 4.0):
					print("No resonable solution.")
				else:
					source = read_binary_file_for_FGR(filename_i)
					target = read_binary_file_for_FGR(filename_j)
					if validating_transform_swap(source, target,
							np.linalg.inv(traj_ij[0].pose)):
						traj_ij[0].metadata = [i, j, n_ply_files]
						traj.append(traj_ij[0])
						print(traj_ij[0])

		write_trajectory(traj, get_log_path(dataset_name))
