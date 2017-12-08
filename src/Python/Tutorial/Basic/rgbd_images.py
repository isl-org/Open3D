# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys
sys.path.append("../..")

#conda install pillow matplotlib
from py3d import *
import re
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# This is special function used for reading NYU pgm format
# as it is written in big endian byte order.
def read_nyu_pgm(filename, byteorder='>'):
	with open(filename, 'rb') as f:
		buffer = f.read()
	try:
		header, width, height, maxval = re.search(
			b"(^P5\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n])*"
			b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
	except AttributeError:
		raise ValueError("Not a raw PGM file: '%s'" % filename)
	img = np.frombuffer(buffer,
		dtype=byteorder+'u2',
		count=int(width)*int(height),
		offset=len(header)).reshape((int(height), int(width)))
	img_out = img.astype('u2')
	return img_out

if __name__ == "__main__":
	print("Read Redwood dataset")
	color_raw_redwood = read_image("../../TestData/RGBD/color/00000.jpg")
	depth_raw_redwood = read_image("../../TestData/RGBD/depth/00000.png")
	rgbd_image_redwood = create_rgbd_image_from_color_and_depth(
		color_raw_redwood, depth_raw_redwood);
	print(rgbd_image_redwood)
	plt.subplot(1, 2, 1)
	plt.title('Redwood grayscale image')
	plt.imshow(rgbd_image_redwood.color)
	plt.subplot(1, 2, 2)
	plt.title('Redwood depth image')
	plt.imshow(rgbd_image_redwood.depth)
	plt.show()
	pcd = create_point_cloud_from_rgbd_image(rgbd_image_redwood, PinholeCameraIntrinsic.prime_sense_default)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])

	print("Read SUN dataset")
	color_raw_sun = read_image("../../TestData/RGBD/other_formats/SUN_color.jpg")
	depth_raw_sun = read_image("../../TestData/RGBD/other_formats/SUN_depth.png")
	rgbd_image_sun = create_rgbd_image_from_sun_format(color_raw_sun, depth_raw_sun);
	print(rgbd_image_sun)
	plt.subplot(1, 2, 1)
	plt.title('SUN grayscale image')
	plt.imshow(rgbd_image_sun.color)
	plt.subplot(1, 2, 2)
	plt.title('SUN depth image')
	plt.imshow(rgbd_image_sun.depth)
	plt.show()
	pcd = create_point_cloud_from_rgbd_image(rgbd_image_sun, PinholeCameraIntrinsic.prime_sense_default)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])

	print("Read NYU dataset")
	# Open3D does not support ppm/pgm file yet. Not using read_image here.
	# MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
	color_raw_nyu = mpimg.imread("../../TestData/RGBD/other_formats/NYU_color.ppm")
	depth_raw_nyu = read_nyu_pgm("../../TestData/RGBD/other_formats/NYU_depth.pgm")
	color = Image(color_raw_nyu)
	depth = Image(depth_raw_nyu)
	rgbd_image_nyu = create_rgbd_image_from_nyu_format(color, depth);
	print(rgbd_image_nyu)
	plt.subplot(1, 2, 1)
	plt.title('NYU grayscale image')
	plt.imshow(rgbd_image_nyu.color)
	plt.subplot(1, 2, 2)
	plt.title('NYU depth image')
	plt.imshow(rgbd_image_nyu.depth)
	plt.show()
	pcd = create_point_cloud_from_rgbd_image(rgbd_image_nyu, PinholeCameraIntrinsic.prime_sense_default)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])

	print("Read TUM dataset")
	color_raw_tum = read_image("../../TestData/RGBD/other_formats/TUM_color.png")
	depth_raw_tum = read_image("../../TestData/RGBD/other_formats/TUM_depth.png")
	rgbd_image_tum = create_rgbd_image_from_tum_format(color_raw_tum, depth_raw_tum);
	print(rgbd_image_tum)
	plt.subplot(1, 2, 1)
	plt.title('TUM grayscale image')
	plt.imshow(rgbd_image_tum.color)
	plt.subplot(1, 2, 2)
	plt.title('TUM depth image')
	plt.imshow(rgbd_image_tum.depth)
	plt.show()
	pcd = create_point_cloud_from_rgbd_image(rgbd_image_tum, PinholeCameraIntrinsic.prime_sense_default)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])
