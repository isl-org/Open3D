.. _mesh:

Reading and visualizing 3D geometry
-------------------------------------

Let's get started Open3D with the following example.

.. code-block:: python

	# src/Python/Tutorial/Basic/mesh.py

	import sys
	import numpy as np
	sys.path.append("../..")
	from py3d import *

	if __name__ == "__main__":

		print("Testing mesh in py3d ...")
		mesh = read_triangle_mesh("../../TestData/knot.ply")
		draw_geometries([mesh])
		mesh.compute_vertex_normals()
		draw_geometries([mesh])
		print(mesh)
		print(np.asarray(mesh.vertices))
		print(np.asarray(mesh.triangles))
		print("")

This example reads ``knot.ply`` file and visualize it.

Let's review the script line by line. The first few lines import necessary Python modules.

.. code-block:: python

	import sys
	import numpy as np
	sys.path.append("../..")
	from py3d import *

It uses ``sys.path.append()`` to refer the path where py3d.so is located.
Once you successfully compiled Open3D with Python binding option,
py3d.so should be visible under Open3D build folder.
If it is not, please go over :ref:`python_binding`.

Let's see the first few lines in the main function.

.. code-block:: python

	print("Testing mesh in py3d ...")
	mesh = read_triangle_mesh("../../TestData/knot.ply")
	draw_geometries([mesh])
	mesh.compute_vertex_normals()

Note that ``draw_geometries`` can visualize multiple geometries simultaneously by taking a list of objects.
We will see more examples of this function later.

.. note:: To get more information for Open3D functions or classes, it is recommended to use Python built-in ``help()``. For example, ``help(draw_geometries)`` will print detailed input/output arguments of ``draw_geometries`` function.

With this script, you shall see this interactive window:

.. image:: ../../_static/mesh_wo_shading.png
    :width: 400px

You can use your mouse/trackpad to see the geometry from different view point.
Wait, why this geometry looks just gray? It is because this mesh does not have surface normal.
Without surface normal, ``draw_geometries`` does not draw surface shading, and that's why we see the gray color.
Press :kbd:`q` to close this interactive window.

OK, let's draw geometry with surface normal. It is pretty easy. Let's continue:

.. code-block:: python

	mesh.compute_vertex_normals()
	draw_geometries([mesh])

Now have this!

.. image:: ../../_static/mesh_w_shading.png
	:width: 400px

You can freely access member variables of ``mesh`` such as its vertices and indices of vertices for mesh triangles.
The following line

.. code-block:: python

	print(mesh)
	print(np.asarray(mesh.vertices))
	print(np.asarray(mesh.triangles))

will print

.. code-block:: python

	TriangleMesh with 1440 points and 2880 triangles.
	[[  4.51268387  28.68865967 -76.55680847]
	 [  7.63622284  35.52046967 -69.78063965]
	 [  6.21986008  44.22465134 -64.82303619]
	 ...,
	 [-22.12651634  31.28466606 -87.37570953]
	 [-13.91188431  25.4865818  -86.25827026]
	 [ -5.27768707  23.36245346 -81.43279266]]
	[[   0   12   13]
	 [   0   13    1]
	 [   1   13   14]
	 ...,
	 [1438   11 1439]
	 [1439   11    0]
	 [1439    0 1428]]

Here, we got some help from ``numpy`` module. ``np.asarray`` transforms Open3D member variables ``mesh.vertices`` and ``mesh.triangles`` into numpy array.
