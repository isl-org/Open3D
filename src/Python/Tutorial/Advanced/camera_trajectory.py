# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
import numpy as np
sys.path.append("../..")
from py3d import *

if __name__ == "__main__":

	print("Testing camera in py3d ...")
	print(PinholeCameraIntrinsic.prime_sense_default)
	print(PinholeCameraIntrinsic.prime_sense_default.intrinsic_matrix)
	print(PinholeCameraIntrinsic())
	x = PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
	print(x)
	print(x.intrinsic_matrix)
	write_pinhole_camera_intrinsic("test.json", x)
	y = read_pinhole_camera_intrinsic("test.json")
	print(y)
	print(np.asarray(y.intrinsic_matrix))

	print("Read a trajectory and combine all the RGB-D images.")
	pcds = [];
	trajectory = read_pinhole_camera_trajectory("../../TestData/RGBD/trajectory.log")
	write_pinhole_camera_trajectory("test.json", trajectory)
	print(trajectory)
	print(trajectory.extrinsic)
	print(np.asarray(trajectory.extrinsic))
	for i in range(5):
		im1 = read_image("../../TestData/RGBD/depth/{:05d}.png".format(i))
		im2 = read_image("../../TestData/RGBD/color/{:05d}.jpg".format(i))
		im = create_rgbd_image_from_color_and_depth(im2, im1, 1000.0, 5.0, False)
		pcd = create_point_cloud_from_rgbd_image(im, trajectory.intrinsic, trajectory.extrinsic[i])
		pcds.append(pcd)
	draw_geometries(pcds)
	print("")
