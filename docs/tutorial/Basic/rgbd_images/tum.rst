.. _rgbd_tum:

TUM dataset
-------------------------------------
This tutorial reads and visualizes an ``RGBDImage`` from `the TUM dataset <https://vision.in.tum.de/data/datasets/rgbd-dataset>`_ [Strum2012]_.

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


This tutorial is almost the same as the tutorial processing :ref:`rgbd_redwood`. The only difference is that we use conversion function ``create_rgbd_image_from_tum_format`` to parse depth images in the TUM dataset.

Similarly, the ``RGBDImage`` can be rendered as numpy arrays:

.. image:: ../../../_static/Basic/rgbd_images/tum_rgbd.png
	:width: 400px

Or a point cloud:

.. image:: ../../../_static/Basic/rgbd_images/tum_pcd.png
	:width: 400px
