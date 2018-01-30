# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import argparse
import sys
sys.path.append("../..")
from py3d import *
sys.path.append("../Utility")
from common import *
from make_fragments import *


def register_one_rgbd_pair(s, t, color_files, depth_files,
		intrinsic, with_opencv):
	# read images
	color_s = read_image(color_files[s])
	depth_s = read_image(depth_files[s])
	color_t = read_image(color_files[t])
	depth_t = read_image(depth_files[t])
	source_rgbd_image = create_rgbd_image_from_color_and_depth(color_s, depth_s)
	target_rgbd_image = create_rgbd_image_from_color_and_depth(color_t, depth_t)

	# initialize_camera_pose
	if abs(s-t) is not 1 and with_opencv:
		success_5pt, odo_init = pose_estimation(
				source_rgbd_image, target_rgbd_image, intrinsic, False)
	else:
		odo_init = np.identity(4)

	# perform RGB-D odometry
	[success, trans, info] = compute_rgbd_odometry(
			source_rgbd_image, target_rgbd_image, intrinsic,
			odo_init, RGBDOdometryJacobianFromHybridTerm(), OdometryOption())

	source = create_point_cloud_from_rgbd_image(source_rgbd_image, intrinsic)
	target = create_point_cloud_from_rgbd_image(target_rgbd_image, intrinsic)
	return [source, target, trans, info]


def test_single_pair(s, t, intrinsic, with_opencv):
	set_verbosity_level(VerbosityLevel.Debug)
	[source, target, trans, info] = register_one_rgbd_pair(s, t,
			color_files, depth_files, intrinsic, with_opencv)

	# integration
	source.transform(trans) # for 5pt
	draw_geometries([source, target])


# test wide baseline matching
if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="mathching two RGBD images")
	parser.add_argument("path_dataset", help="path to the dataset")
	parser.add_argument("source_id", type=int, help="ID of source RGBD image")
	parser.add_argument("target_id", type=int, help="ID of target RGBD image")
	parser.add_argument("-path_intrinsic", help="path to the RGBD camera intrinsic")
	args = parser.parse_args()

	with_opencv = initialize_opencv()
	if with_opencv:
		from opencv_pose_estimation import pose_estimation

	[color_files, depth_files] = get_rgbd_file_lists(args.path_dataset)
	if args.path_intrinsic:
		intrinsic = read_pinhole_camera_intrinsic(args.path_intrinsic)
	else:
		intrinsic = PinholeCameraIntrinsic.prime_sense_default
	test_single_pair(args.source_id, args.target_id, intrinsic, with_opencv)
