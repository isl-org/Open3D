# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":
	pinhole_camera_intrinsic = read_pinhole_camera_intrinsic(
			"../../TestData/camera.json")
	print(pinhole_camera_intrinsic.intrinsic_matrix)

	source_color = read_image("../../TestData/RGBD/color/00000.jpg")
	source_depth = read_image("../../TestData/RGBD/depth/00000.png")
	target_color = read_image("../../TestData/RGBD/color/00001.jpg")
	target_depth = read_image("../../TestData/RGBD/depth/00001.png")
	source_rgbd_image = create_rgbd_image_from_color_and_depth(
			source_color, source_depth)
	target_rgbd_image = create_rgbd_image_from_color_and_depth(
			target_color, target_depth)
	target_pcd = create_point_cloud_from_rgbd_image(
			target_rgbd_image, pinhole_camera_intrinsic)

	option = OdometryOption()
	odo_init = np.identity(4)
	print(option)

	[success_color_term, trans_color_term, info] = compute_rgbd_odometry(
			source_rgbd_image, target_rgbd_image,
			pinhole_camera_intrinsic, odo_init,
			RGBDOdometryJacobianFromColorTerm(), option)
	[success_hybrid_term, trans_hybrid_term, info] = compute_rgbd_odometry(
			source_rgbd_image, target_rgbd_image,
			pinhole_camera_intrinsic, odo_init,
			RGBDOdometryJacobianFromHybridTerm(), option)

	if success_color_term:
		print("Using RGB-D Odometry")
		print(trans_color_term)
		source_pcd_color_term = create_point_cloud_from_rgbd_image(
				source_rgbd_image, pinhole_camera_intrinsic)
		source_pcd_color_term.transform(trans_color_term)
		draw_geometries([target_pcd, source_pcd_color_term])
	if success_hybrid_term:
		print("Using Hybrid RGB-D Odometry")
		print(trans_hybrid_term)
		source_pcd_hybrid_term = create_point_cloud_from_rgbd_image(
				source_rgbd_image, pinhole_camera_intrinsic)
		source_pcd_hybrid_term.transform(trans_hybrid_term)
		draw_geometries([target_pcd, source_pcd_hybrid_term])
