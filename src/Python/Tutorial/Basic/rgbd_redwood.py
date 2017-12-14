# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")

#conda install pillow matplotlib
from py3d import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
	print("Read Redwood dataset")
	color_raw = read_image("../../TestData/RGBD/color/00000.jpg")
	depth_raw = read_image("../../TestData/RGBD/depth/00000.png")
	rgbd_image = create_rgbd_image_from_color_and_depth(
		color_raw, depth_raw);
	print(rgbd_image)
	plt.subplot(1, 2, 1)
	plt.title('Redwood grayscale image')
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title('Redwood depth image')
	plt.imshow(rgbd_image.depth)
	plt.show()
	pcd = create_point_cloud_from_rgbd_image(rgbd_image,
			PinholeCameraIntrinsic.prime_sense_default)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])
