# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":
	source_color = read_image("../../TestData/RGBD/color/00000.jpg")
	source_depth = read_image("../../TestData/RGBD/depth/00000.png")
	target_color = read_image("../../TestData/RGBD/color/00001.jpg")
	target_depth = read_image("../../TestData/RGBD/depth/00001.png")
	source_rgbd_image = create_rgbd_image_from_color_and_depth(
			source_color, source_depth);
	target_rgbd_image = create_rgbd_image_from_color_and_depth(
			target_color, target_depth);

	pinhole_camera_intrinsic = read_pinhole_camera_intrinsic(
			"../../TestData/camera.json")
	option = OdometryOption()
	odo_init = np.identity(4)
	print(pinhole_camera_intrinsic.intrinsic_matrix)
	print(option)

	[success, trans, info] = compute_rgbd_odometry(
			source_rgbd_image, target_rgbd_image,
			pinhole_camera_intrinsic, odo_init,
			RGBDOdometryJacobianFromColorTerm(), option)
	if success:
		print("Using RGB-D Odometry")
		print(trans)

	[success, trans, info] = compute_rgbd_odometry(
			source_rgbd_image, target_rgbd_image,
			pinhole_camera_intrinsic, odo_init,
			RGBDOdometryJacobianFromHybridTerm(), option)
	if success:
		print("Using Hybrid RGB-D Odometry")
		print(trans)
