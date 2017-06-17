import sys
sys.path.append("../..")
from py3d import *
import numpy as np

if __name__ == "__main__":
	source_color = ReadImage("../../TestData/RGBD/color/00000.jpg")
	source_depth = ReadImage("../../TestData/RGBD/depth/00000.png")
	target_color = ReadImage("../../TestData/RGBD/color/00001.jpg")
	target_depth = ReadImage("../../TestData/RGBD/depth/00001.png")
	source_rgbd_image = CreateRGBDImageFromColorAndDepth(
			source_color, source_depth);
	target_rgbd_image = CreateRGBDImageFromColorAndDepth(
			target_color, target_depth);

	camera_intrinsic = ReadPinholeCameraIntrinsic("../../TestData/camera.json")
	option = OdometryOption()
	odo_init = np.identity(4)
	print(camera_intrinsic.intrinsic_matrix)
	print(option)

	[success, trans, info] = ComputeRGBDOdometry(
			source_rgbd_image, target_rgbd_image,
			camera_intrinsic, odo_init, option)
	if success:
		print("Using RGB-D Odometry")
		print(trans)

	[success, trans, info] = ComputeRGBDHybridOdometry(
			source_rgbd_image, target_rgbd_image,
			camera_intrinsic, odo_init, option)
	if success:
		print("Using Hybrid RGB-D Odometry")
		print(trans)
