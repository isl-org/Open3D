RGBD Images
===================================================================

Open3D has a data structure for images. It supports various functions such as ``read_image``, ``write_image``, ``filter_image`` and ``draw_geometries``. An Open3D ``Image`` can be directly converted to/from a numpy array.

An Open3D ``RGBDImage`` is composed of two images, ``RGBDImage.depth`` and ``RGBDImage.color``. We require the two images to be registered into the same camera frame and have the same resolution. The following tutorials show how to read and use RGBD images from a number of well known RGBD datasets.

.. toctree::

	redwood
	sun
	nyu
	tum
