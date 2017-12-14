.. _tutorial:

Tutorial
#######################

This is a quick Python tutorial to be familiar with the library.


.. _tutorial_basic:

Basic
=================


Importing Open3D module
-------------------------------------
Let's get started Open3D by importing Python module.
Once you successfully compiled Open3D with Python binding option,
py3d.so should be visible under Open3D build folder. Let's import it.

.. code-block:: python

	import sys
	sys.path.append("../..")
	from py3d import *

This example uses ``sys.path.append()`` to refer the path where py3d.so is located.
If this script runs without any failure message, you are ready to move to the next tutorial.
If it fails, please go over :ref:`python_binding`.

.. note:: To get more information for Open3D functions or classes, it is recommended to use Python built-in ``help()``. For example, ``help(draw_geometries)`` will print detailed input/output arguments of ``draw_geometries`` function.

Reading and drawing 3D geometry
-------------------------------------
Reading and drawing 3D geometry is made easy. Let's see `Basic/visualize_pointcloud.py`.

.. code-block:: python

	pcd = read_point_cloud("../../TestData/fragment.ply")
	draw_geometries([pcd])

Note that ``draw_geometries`` can visualize multiple geometries by taking a list of objects.
We will see more examples of this function later.

For visualizing meshes, use ``read_triangle_mesh``. For example,

.. code-block:: python

	mesh = read_triangle_mesh("../../TestData/knot.ply")
	mesh.compute_vertex_normals()
	draw_geometries([mesh])

Here, method ``compute_vertex_normals`` computes point normal of meshed surface.
``draw_geometries`` will draw shaded geometry based on the computed normal direction.


Manipulating and saving 3D geometry
-------------------------------------

This is another example, where user gives selection of geometry, and

.. code-block:: python

	vol = read_selection_polygon_volume("../TestData/Crop/cropped.json")
	chair = vol.crop_point_cloud(pcd)
	draw_geometries([chair])


Read RGB Image
-------------------------------------

.. code-block:: python

	im_raw = mpimg.imread("../TestData/lena_color.jpg")
	im = Image(im_raw)
	im_g3 = filter_image(im, ImageFilterType.Gaussian3)
	im_g5 = filter_image(im, ImageFilterType.Gaussian5)
	im_g7 = filter_image(im, ImageFilterType.Gaussian7)
	im_gaussian = [im, im_g3, im_g5, im_g7]
	pyramid_levels = 4
	pyramid_with_gaussian_filter = True
	im_pyramid = create_image_pyramid(im, pyramid_levels,
			pyramid_with_gaussian_filter)
	im_dx = filter_image(im, ImageFilterType.Sobel3dx)
	im_dx_pyramid = filter_image_pyramid(im_pyramid, ImageFilterType.Sobel3dx)
	im_dy = filter_image(im, ImageFilterType.Sobel3dy)
	im_dy_pyramid = filter_image_pyramid(im_pyramid, ImageFilterType.Sobel3dy)


Read RGBD Image
-------------------------------------
Open3D provides a variety of 3D reconstruction algorithms using depth cameras.
A pair of color and depth image is a input for the reconstruction pipeline.



.. _tutorial_advanced:

Advanced
=================

Customized Visualization
-------------------------------------

RGBD Odometry
-------------------------------------

Pointcloud Registration
-------------------------------------
