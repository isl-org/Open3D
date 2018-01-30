.. _rgbd_nyu:

NYU dataset
-------------------------------------
This tutorial reads and visualizes an ``RGBDImage`` from `the NYU dataset <https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html>`_ [Silberman2012]_.

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

This tutorial is almost the same as the tutorial processing :ref:`rgbd_redwood`, with two differences. First, NYU images are not in standard ``jpg`` or ``png`` formats. Thus, we use ``mpimg.imread`` to read the color image as a numpy array and convert it to an Open3D ``Image``. An additional helper function ``read_nyu_pgm`` is called to read depth images from the special big endian ``pgm`` format used in the NYU dataset. Second, we use a different conversion function ``create_rgbd_image_from_nyu_format`` to parse depth images in the SUN dataset.

Similarly, the RGBDImage can be rendered as numpy arrays:

.. image:: ../../../_static/Basic/rgbd_images/nyu_rgbd.png
	:width: 400px

Or a point cloud:

.. image:: ../../../_static/Basic/rgbd_images/nyu_pcd.png
	:width: 400px
