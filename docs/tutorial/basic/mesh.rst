.. _mesh:

Mesh
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


.. _visualize_3d_mesh:

Visualize 3D mesh
=====================================

Let's see the first few lines in the main function.

.. code-block:: python

	print("Testing mesh in py3d ...")
	mesh = read_triangle_mesh("../../TestData/knot.ply")
	draw_geometries([mesh])
	mesh.compute_vertex_normals()

Note that ``draw_geometries`` can visualize multiple geometries simultaneously by taking a list of objects.
There is more examples of this function later.

.. note:: To get more information for Open3D functions or classes, it is recommended to use Python built-in ``help()``. For example, ``help(draw_geometries)`` will print detailed input/output arguments of ``draw_geometries`` function.

With this script, this interactive window appears:

.. image:: ../../_static/basic/mesh_wo_shading.png
    :width: 400px

Use mouse/trackpad to see the geometry from different view point.
This geometry looks just gray because this mesh does not have surface normal.
Without surface normal, ``draw_geometries`` does not draw surface shading.
Press :kbd:`q` to close this interactive window.


.. _vertex_normal_estimation:

Vertex normal estimation
=====================================

Let's draw geometry with surface normal. Let's continue:

.. code-block:: python

	mesh.compute_vertex_normals()
	draw_geometries([mesh])

Now it looks like this!

.. image:: ../../_static/basic/mesh_w_shading.png
	:width: 400px

``mesh`` has several member variables such as its vertices and indices of vertices for mesh triangles.


.. _print_vertices_and_triangles:

Print vertices and triangles
=====================================

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

Here, the script got some help from ``numpy`` module. ``np.asarray`` transforms Open3D member variables ``mesh.vertices`` and ``mesh.triangles`` into numpy array.
