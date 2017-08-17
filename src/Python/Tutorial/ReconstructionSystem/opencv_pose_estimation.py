import numpy as np
import sys
import cv2
sys.path.append("../..")
from py3d import *
from matplotlib import pyplot as plt
import copy

# following code is tested with OpenCV 3.2.0
# how to install opencv
# conda create --prefix py27_test python=2.7
# conda install -c conda-forge opencv
# conda install -c conda-forge openblas
def pose_estimation(source_rgbd_image, target_rgbd_image,
		pinhole_camera_intrinsic):
	# transform double array to unit8 array
	color_cv_s = np.uint8(np.asarray(source_rgbd_image.color)*255.0)
	color_cv_t = np.uint8(np.asarray(target_rgbd_image.color)*255.0)

	orb = cv2.ORB_create()
	[kp_s, des_s] = orb.detectAndCompute(color_cv_s, None)
	[kp_t, des_t] = orb.detectAndCompute(color_cv_t, None)

	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	matches = bf.match(des_s,des_t)

	pts_s = []
	pts_t = []
	for match in matches:
		pts_t.append(kp_t[match.trainIdx].pt)
		pts_s.append(kp_s[match.queryIdx].pt)
	pts_s = np.int32(pts_s)
	pts_t = np.int32(pts_t)

	focal_input = (pinhole_camera_intrinsic.intrinsic_matrix[0,0] +
			pinhole_camera_intrinsic.intrinsic_matrix[1,1]) / 2.0
	pp_x = pinhole_camera_intrinsic.intrinsic_matrix[0,2]
	pp_y = pinhole_camera_intrinsic.intrinsic_matrix[1,2]

	# Essential matrix is just used for masking inliers using Epipolar geometry
	[E, mask] = cv2.findEssentialMat(pts_s, pts_t, focal=focal_input,
			pp=(pp_x, pp_y), method=cv2.RANSAC, prob=0.999, threshold=3.0)

	# make 3D correspondences
	depth_s = np.asarray(source_rgbd_image.depth)
	depth_t = np.asarray(target_rgbd_image.depth)
	pts_xyz_s = np.zeros([3, pts_s.shape[0]])
	pts_xyz_t = np.zeros([3, pts_s.shape[0]])
	cnt = 0
	for i in range(pts_s.shape[0]):
		if mask[i]:
			xyz_s = get_xyz_from_pts(pts_s[i,:], depth_s, pp_x, pp_y, focal_input)
			pts_xyz_s[:,cnt] = xyz_s
			xyz_t = get_xyz_from_pts(pts_t[i,:], depth_t, pp_x, pp_y, focal_input)
			pts_xyz_t[:,cnt] = xyz_t
			cnt = cnt + 1
	pts_xyz_s = pts_xyz_s[:,:cnt]
	pts_xyz_t = pts_xyz_t[:,:cnt]
	R,t = estimate_3D_transform_RANSAC(pts_xyz_s, pts_xyz_t)
	trans = np.identity(4)
	trans[:3,:3] = R
	trans[:3,3] = [t[0],t[1],t[2]]
	return trans

def estimate_3D_transform_RANSAC(pts_xyz_s, pts_xyz_t):
	max_iter = 1000
	max_distance = 0.05
	max_distance2 = max_distance * max_distance
	n_sample = 5
	n_points = pts_xyz_s.shape[1]
	R_good = np.identity(3)
	t_good = np.zeros([3,1])
	max_inlier = n_sample
	for i in range(max_iter):
		rand_idx = np.random.randint(n_points, size=n_sample)
		sample_xyz_s = pts_xyz_s[:,rand_idx]
		sample_xyz_t = pts_xyz_t[:,rand_idx]
		R_approx, t_approx = estimate_3D_transform(sample_xyz_s, sample_xyz_t)
		# evaluation
		diff_mat = pts_xyz_t - (np.matmul(R_approx, pts_xyz_s) +
				np.tile(t_approx, [1, n_points]))
		diff = [np.linalg.norm(diff_mat[:,i]) for i in range(n_points)]
		n_inlier = len([1 for diff_iter in diff if diff_iter < max_distance2])
		if (n_inlier > max_inlier):
			R_good = copy.copy(R_approx)
			t_good = copy.copy(t_approx)
			max_inlier = n_inlier
	return R_good, t_good

# singular value decomposition approach
# based on the description in the sec 3.1.2 in
# http://graphics.stanford.edu/~smr/ICP/comparison/eggert_comparison_mva97.pdf
def estimate_3D_transform(input_xyz_s, input_xyz_t):
	# compute H
	xyz_s = copy.copy(input_xyz_s)
	xyz_t = copy.copy(input_xyz_t)
	n_points = xyz_s.shape[1]
	mean_s = np.mean(xyz_s, axis=1)
	mean_t = np.mean(xyz_t, axis=1)
	mean_s.shape = (3,1)
	mean_t.shape = (3,1)
	xyz_diff_s = xyz_s - np.tile(mean_s, [1, n_points])
	xyz_diff_t = xyz_t - np.tile(mean_t, [1, n_points])
	H = np.matmul(xyz_diff_s,xyz_diff_t.transpose())
	# solve system
	U, s, V = np.linalg.svd(H)
	R_approx = np.matmul(V.transpose(), U.transpose())
	if np.linalg.det(R_approx) < 0.0:
		det = np.linalg.det(np.matmul(U,V))
		D = np.identity(3)
		D[2,2] = det
		R_approx = np.matmul(U,np.matmul(D,V))
	t_approx = mean_t - np.matmul(R_approx, mean_s)
	return R_approx, t_approx

def get_xyz_from_pts(pts_row, depth, px, py, focal):
	u = pts_row[0]
	v = pts_row[1]
	d = depth[v, u]
	return get_xyz_from_uv(u, v, d, px, py, focal)

def get_xyz_from_uv(u, v, d, px, py, focal):
	x = (u - px) / focal * d
	y = (v - py) / focal * d
	return np.array([x, y, d]).transpose()
