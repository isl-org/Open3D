.. _rgbd_nyu:

NYU dataset
-------------------------------------
This tutorial reads and visualizes a RGBD image of SUN dataset [SILBERMAN2012]_.
Let's see following tutorial.

.. code-block:: python

	# src/Python/Tutorial/Basic/rgbd_nyu.py

	import sys
	sys.path.append("../..")

	#conda install pillow matplotlib
	from py3d import *
	import numpy as np
	import re
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
		print("Read NYU dataset")
		# Open3D does not support ppm/pgm file yet. Not using read_image here.
		# MathplotImage having some ISSUE with NYU pgm file. Not using imread for pgm.
		color_raw = mpimg.imread("../../TestData/RGBD/other_formats/NYU_color.ppm")
		depth_raw = read_nyu_pgm("../../TestData/RGBD/other_formats/NYU_depth.pgm")
		color = Image(color_raw)
		depth = Image(depth_raw)
		rgbd_image = create_rgbd_image_from_nyu_format(color, depth)
		print(rgbd_image)
		plt.subplot(1, 2, 1)
		plt.title('NYU grayscale image')
		plt.imshow(rgbd_image.color)
		plt.subplot(1, 2, 2)
		plt.title('NYU depth image')
		plt.imshow(rgbd_image.depth)
		plt.show()
		pcd = create_point_cloud_from_rgbd_image(rgbd_image,
				PinholeCameraIntrinsic.prime_sense_default)
		# Flip it, otherwise the pointcloud will be upside down
		pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		draw_geometries([pcd])

Let's take a look at this script one by one.

.. code-block:: python

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

This function is specialized for reading pgm format depth images. The depth bits of NYU dataset is written in big endian byte order. This function is to transform it into little endian byte order. It returns 16bit depth images.

.. code-block:: python

	color_raw = mpimg.imread("../../TestData/RGBD/other_formats/NYU_color.ppm")
	depth_raw = read_nyu_pgm("../../TestData/RGBD/other_formats/NYU_depth.pgm")
	color = Image(color_raw)
	depth = Image(depth_raw)
	rgbd_image = create_rgbd_image_from_nyu_format(color, depth)

This script is bit tweaked for reading ppm and pgm images. The raw images are transformed float type ``Image`` class. The color image is normalized to [0,1] and depth image is [0,infinity].
The depth unit is metric: 1 means 1 meter and 0 indicates invalid depth. Open3D rgbd_image class is made with ``create_rgbd_image_from_nyu_format``.

``print(rgbd_image)`` prints brief information of ``rgbd_image``.

.. code-block:: python

	RGBDImage of size
	Color image : 640x480, with 1 channels.
	Depth image : 640x480, with 1 channels.
	Use numpy.asarray to access buffer data.

The next lines below

.. code-block:: python

	plt.subplot(1, 2, 1)
	plt.title('NYU grayscale image')
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title('NYU depth image')
	plt.imshow(rgbd_image.depth)
	plt.show()

displays two images using ``subplot``:

.. image:: ../../../_static/basic/rgbd_images/nyu_rgbd.png
	:width: 400px

Any RGBD image can be transformed into point cloud. This is interesting feature of RGBD image.

.. code-block:: python

	pcd = create_point_cloud_from_rgbd_image(rgbd_image,
			PinholeCameraIntrinsic.prime_sense_default)
	# Flip it, otherwise the pointcloud will be upside down
	pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
	draw_geometries([pcd])

``create_point_cloud_from_rgbd_image`` makes point cloud from ``rgbd_image``.
Here, ``PinholeCameraIntrinsic.prime_sense_default`` is used as an input arguement.
It corresponds to default camera intrinsic matrix of Kinect camera with 640x480 resolution.

Note that ``pcd.transform`` is applied for the ``pcd`` just for visualization purpose.
This script will display:

.. image:: ../../../_static/basic/rgbd_images/nyu_pcd.png
	:width: 400px

.. [SILBERMAN2012] N. Silberman, D. Hoiem, P. Kohli and R. Fergus, Indoor Segmentation and Support Inference from RGBD Images, ECCV, 2012.
