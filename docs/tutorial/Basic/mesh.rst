.. _mesh:

Mesh
-------------------------------------

Open3D has a data structure for triangle mesh.

.. code-block:: python

    # src/Python/Tutorial/Basic/mesh.py

        import sys
        import numpy as np
        sys.path.append("../..")
        from py3d import *

        if __name__ == "__main__":

        print("Testing mesh in py3d ...")
        mesh = read_triangle_mesh("../../TestData/knot.ply")
        print(mesh)
        print(np.asarray(mesh.vertices))
        print(np.asarray(mesh.triangles))
        print("")

        print("Try to render a mesh with normals (exist: " +
                str(mesh.has_vertex_normals()) +
                ") and colors (exist: " + str(mesh.has_vertex_colors()) + ")")
        draw_geometries([mesh])
        print("A mesh with no normals and no colors does not seem good.")

        print("Computing normal and rendering it.")
        mesh.compute_vertex_normals()
        print(np.asarray(mesh.triangle_normals))
        draw_geometries([mesh])

        print("We make a partial mesh of only the first half triangles.")
        mesh1 = copy.deepcopy(mesh)
        mesh1.triangles = Vector3iVector(
                np.asarray(mesh1.triangles)[:len(mesh1.triangles)//2, :])
        mesh1.triangle_normals = Vector3dVector(
                np.asarray(mesh1.triangle_normals)
                [:len(mesh1.triangle_normals)//2, :])
        print(mesh1.triangles)
        draw_geometries([mesh1])

        print("Painting the mesh")
        mesh1.paint_uniform_color([1, 0.706, 0])
        draw_geometries([mesh1])


.. _print_vertices_and_triangles:

Print vertices and triangles
=====================================

.. code-block:: python

    print("Testing mesh in py3d ...")
    mesh = read_triangle_mesh("../../TestData/knot.ply")
    print(mesh)
    print(np.asarray(mesh.vertices))
    print(np.asarray(mesh.triangles))
    print("")

Outputs:

.. code-block:: sh

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

The ``TriangleMesh`` class has a few data fields such as ``vertices`` and ``triangles``. Open3D provides direct memory access to these fields via numpy array.

.. _visualize_3d_mesh:

Visualize 3D mesh
=====================================

.. code-block:: python

    print("Try to render a mesh with normals (exist: " +
            str(mesh.has_vertex_normals()) +
            ") and colors (exist: " + str(mesh.has_vertex_colors()) + ")")
    draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

The GUI visualizes a mesh.

.. image:: ../../_static/Basic/mesh/without_shading.png
    :width: 400px

You can rotate and move the mesh but it is painted with uniform gray color and does not look "3d". The reason is that the current mesh does not have normals for vertices or faces. So uniform color shading is used instead of a more sophisticated Phong shading.

.. _surface_normal_estimation:

Surface normal estimation
=====================================

Let's draw the mesh with surface normal.

.. code-block:: python

    print("Computing normal, painting the mesh, and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    draw_geometries([mesh])

It uses ``compute_vertex_normals`` and ``paint_uniform_color`` which are member functions of ``mesh``.
Now it looks like:

.. image:: ../../_static/Basic/mesh/with_shading.png
    :width: 400px

Crop mesh
=====================================

We remove half of the surface by directly operate on the ``triangle`` and ``triangle_normals`` data fields of the mesh. This is done via numpy array.

.. code-block:: python

    print("We make a partial mesh of only the first half triangles.")
    mesh1 = copy.deepcopy(mesh)
    mesh1.triangles = Vector3iVector(
            np.asarray(mesh1.triangles)[:len(mesh1.triangles)//2, :])
    mesh1.triangle_normals = Vector3dVector(
            np.asarray(mesh1.triangle_normals)
            [:len(mesh1.triangle_normals)//2, :])
    print(mesh1.triangles)
    draw_geometries([mesh1])

Outputs:

.. image:: ../../_static/Basic/mesh/half.png
    :width: 400px


Paint mesh
=====================================

Painting mesh is the same as how it worked for point cloud.

.. code-block:: python

    print("Painting the mesh")
    mesh1.paint_uniform_color([1, 0.706, 0])
    draw_geometries([mesh1])

Outputs:

.. image:: ../../_static/Basic/mesh/half_color.png
    :width: 400px
