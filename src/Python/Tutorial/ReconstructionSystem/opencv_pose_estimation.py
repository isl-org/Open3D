import numpy as np
import sys
import cv2
sys.path.append("../..")
from py3d import *
from matplotlib import pyplot as plt
import copy

# following code is tested with OpenCV 3.2.0 and Python2.7
# how to install opencv
# conda create --prefix py27opencv python=2.7
# source activate py27opencv
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
	# print('pinhole_camera_intrinsic.intrinsic_matrix')
	# print(pinhole_camera_intrinsic.intrinsic_matrix)
	# print('focal_input, pp_x, pp_y')
	# print([focal_input, pp_x, pp_y])

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

	# # can I draw correspondences here?
	# draw_correspondences(np.asarray(source_rgbd_image.color),
	# 		np.asarray(target_rgbd_image.color), pts_s, pts_t, mask)
	#R,t,inlier_id_vec = estimate_3D_transform_RANSAC(pts_xyz_s, pts_xyz_t)
	R,t = estimate_3D_transform_RANSAC(pts_xyz_s, pts_xyz_t)
	trans = np.identity(4)
	trans[:3,:3] = R
	trans[:3,3] = [t[0],t[1],t[2]]

	# pts_s_new = np.zeros(shape=(len(inlier_id_vec),2))
	# pts_t_new = np.zeros(shape=(len(inlier_id_vec),2))
	# mask = np.ones(len(inlier_id_vec))
	# cnt = 0
	# for inlier_id in inlier_id_vec:
	# 	u_s,v_s = get_uv_from_xyz(pts_xyz_s[0,inlier_id], pts_xyz_s[1,inlier_id],
	# 			pts_xyz_s[2,inlier_id], pp_x, pp_y, focal_input)
	# 	u_t,v_t = get_uv_from_xyz(pts_xyz_t[0,inlier_id], pts_xyz_t[1,inlier_id],
	# 			pts_xyz_t[2,inlier_id], pp_x, pp_y, focal_input)
	# 	pts_s_new[cnt,:] = [u_s,v_s]
	# 	pts_t_new[cnt,:] = [u_t,v_t]
	# 	cnt = cnt + 1
	# draw_correspondences(np.asarray(source_rgbd_image.color),
	# 		np.asarray(target_rgbd_image.color), pts_s_new, pts_t_new, mask)
	return trans


def draw_correspondences(img_s, img_t, pts_s, pts_t, mask):
	ha,wa = img_s.shape[:2]
	hb,wb = img_t.shape[:2]
	total_width = wa+wb
	new_img = np.zeros(shape=(ha, total_width))
	new_img[:ha,:wa]=img_s
	new_img[:hb,wa:wa+wb]=img_t
	for i in range(pts_s.shape[0]):
		if mask[i]:
		#if 1:
			sx = pts_s[i,0]
			sy = pts_s[i,1]
			tx = pts_t[i,0] + wa
			ty = pts_t[i,1]
			plt.plot([sx,tx], [sy,ty], color=np.random.random(3)/2+0.5, lw=1.0)
	plt.imshow(new_img)
	plt.show()


def estimate_3D_transform_RANSAC(pts_xyz_s, pts_xyz_t):
	max_iter = 1000
	max_distance = 0.05
	n_sample = 5
	n_points = pts_xyz_s.shape[1]
	R_good = np.identity(3)
	t_good = np.zeros([3,1])
	max_inlier = n_sample
	idvec = range(n_points)
	inlier_vec_good = []
	for i in range(max_iter):
		rand_idx = np.random.randint(n_points, size=n_sample)
		sample_xyz_s = pts_xyz_s[:,rand_idx]
		sample_xyz_t = pts_xyz_t[:,rand_idx]
		R_approx, t_approx = estimate_3D_transform(sample_xyz_s, sample_xyz_t)
		# evaluation
		diff_mat = pts_xyz_t - (np.matmul(R_approx, pts_xyz_s) +
				np.tile(t_approx, [1, n_points]))
		diff = [np.linalg.norm(diff_mat[:,i]) for i in range(n_points)]
		# inlier_vec = [id_iter for diff_iter, id_iter in zip(diff, idvec) \
		# 		if diff_iter < max_distance]
		#n_inlier = len(inlier_vec)
		n_inlier = len([1 for diff_iter in diff if diff_iter < max_distance])
		if (n_inlier > max_inlier):
			print(max_inlier)
			R_good = copy.copy(R_approx)
			t_good = copy.copy(t_approx)
			max_inlier = n_inlier
			# inlier_vec_good = inlier_vec
	return R_good, t_good#, inlier_vec_good


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


def get_uv_from_xyz(x, y, z, px, py, focal):
	u = focal * x / z + px
	v = focal * y / z + py
	return u, v
