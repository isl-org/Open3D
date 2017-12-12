.. _visualization:

Visualization
-------------------------------------

Let's get started Open3D with the following example.

.. code-block:: python

	# src/Python/Tutorial/Basic/visualization.py

	import sys, copy
	import numpy as np
	sys.path.append("../..")
	from py3d import *

	if __name__ == "__main__":

		print("Testing visualization in py3d ...")
		mesh = read_triangle_mesh("../../TestData/knot.ply")

		print("Try to render a mesh with normals " +
				str(mesh.has_vertex_normals()) +
				" and colors " + str(mesh.has_vertex_colors()))
		draw_geometries([mesh])

		print("A mesh with no normals and no colors does not seem good.")
		mesh.compute_vertex_normals()
		mesh.paint_uniform_color([0.1, 0.1, 0.7])
		print(np.asarray(mesh.triangle_normals))
		print("We paint the mesh and render it.")
		draw_geometries([mesh])

		print("We make a partial mesh of only the first half triangles.")
		mesh1 = copy.deepcopy(mesh)
		print(mesh1.triangles)
		mesh1.triangles = Vector3iVector(
				np.asarray(mesh1.triangles)[:len(mesh1.triangles)/2, :])
		mesh1.triangle_normals = Vector3dVector(
				np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals)/2, :])
		print(mesh1.triangles)
		draw_geometries([mesh1])

		# let's draw some primitives
		mesh_sphere = create_mesh_sphere(radius = 1.0)
		mesh_sphere.compute_vertex_normals()
		mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
		mesh_cylinder = create_mesh_cylinder(radius = 0.3, height = 4.0)
		mesh_cylinder.compute_vertex_normals()
		mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
		mesh_frame = create_mesh_coordinate_frame(size = 0.6, origin = [-2, -2, -2])
		print("We draw a few primitives using collection.")
		draw_geometries([mesh_sphere, mesh_cylinder, mesh_frame])
		print("We draw a few primitives using + operator of mesh.")
		draw_geometries([mesh_sphere + mesh_cylinder + mesh_frame])

		print("")
