.. _rgbd_tum:

TUM dataset
-------------------------------------
This tutorial reads and visualizes a RGBD image of SUN dataset [Strum2012]_.
Let's see following script.

.. code-block:: python

	# src/Python/Tutorial/Basic/rgbd_tum.py

	import sys
	sys.path.append("../..")

	#conda install pillow matplotlib
	from py3d import *
	import matplotlib.pyplot as plt


	if __name__ == "__main__":
		print("Read TUM dataset")
		color_raw = read_image("../../TestData/RGBD/other_formats/TUM_color.png")
		depth_raw = read_image("../../TestData/RGBD/other_formats/TUM_depth.png")
		rgbd_image = create_rgbd_image_from_tum_format(color_raw, depth_raw);
		print(rgbd_image)
		plt.subplot(1, 2, 1)
		plt.title('TUM grayscale image')
		plt.imshow(rgbd_image.color)
		plt.subplot(1, 2, 2)
		plt.title('TUM depth image')
		plt.imshow(rgbd_image.depth)
		plt.show()
		pcd = create_point_cloud_from_rgbd_image(rgbd_image,
				PinholeCameraIntrinsic.prime_sense_default)
		# Flip it, otherwise the pointcloud will be upside down
		pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
		draw_geometries([pcd])

Let's take a look at this script one by one.

.. code-block:: python

	print("Read TUM dataset")
	color_raw = read_image("../../TestData/RGBD/other_formats/TUM_color.png")
	depth_raw = read_image("../../TestData/RGBD/other_formats/TUM_depth.png")
	rgbd_image = create_rgbd_image_from_tum_format(color_raw, depth_raw);
	print(rgbd_image)

The above block reads color and depth images.
``create_rgbd_image_from_tum_format`` transforms color and depth image into ``rgbd_image`` that has float type images of color and depth.
The color image is normalized to [0,1] and depth image is [0,infinity].
The depth unit is metric: 1 means 1 meter and 0 indicates invalid depth.

``print(rgbd_image)`` prints brief information of ``rgbd_image``.

.. code-block:: python

	RGBDImage of size
	Color image : 640x480, with 1 channels.
	Depth image : 640x480, with 1 channels.
	Use numpy.asarray to access buffer data.

The next lines below

.. code-block:: python

	plt.subplot(1, 2, 1)
	plt.title('TUM grayscale image')
	plt.imshow(rgbd_image.color)
	plt.subplot(1, 2, 2)
	plt.title('TUM depth image')
	plt.imshow(rgbd_image.depth)
	plt.show()

displays two images using ``subplot``:

.. image:: ../../../_static/Basic/rgbd_images/tum_rgbd.png
	:width: 400px

The RGBD image can be transformed into point cloud. This is interesting feature of RGBD image.

``create_point_cloud_from_rgbd_image`` makes point cloud from ``rgbd_image``.
Here, ``PinholeCameraIntrinsic.prime_sense_default`` is used as an input arguement.
It corresponds to default camera intrinsic matrix of Kinect camera with 640x480 resolution.

Note that ``pcd.transform`` is applied for the ``pcd`` just for visualization purpose.
This script will display:

.. image:: ../../../_static/Basic/rgbd_images/tum_pcd.png
	:width: 400px
