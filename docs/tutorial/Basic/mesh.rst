.. _mesh:

Mesh
-------------------------------------

This tutorial introduces basic usage regarding mesh in Open3D.

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
                np.asarray(mesh1.triangles)[:len(mesh1.triangles)/2, :])
        mesh1.triangle_normals = Vector3dVector(
                np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals)/2, :])
        print(mesh1.triangles)
        draw_geometries([mesh1])

        print("Painting the mesh")
        mesh1.paint_uniform_color([1, 0.706, 0])
        draw_geometries([mesh1])

This example reads ``knot.ply`` file, manipulates it, and visualize it.


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

will print

.. code-block:: shell

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


.. _visualize_3d_mesh:

Visualize 3D mesh
=====================================

The next line will visualize the mesh

.. code-block:: python

    print("Try to render a mesh with normals (exist: " +
            str(mesh.has_vertex_normals()) +
            ") and colors (exist: " + str(mesh.has_vertex_colors()) + ")")
    draw_geometries([mesh])
    print("A mesh with no normals and no colors does not seem good.")

With this script, this interactive window appears:

.. image:: ../../_static/Basic/mesh/without_shading.png
    :width: 400px

This geometry looks like gray silhouette because this mesh does not have surface normal.
Without surface normal, ``draw_geometries`` does not draw surface shading.
Press :kbd:`q` to close this interactive window.

This script also prints the following:

.. code-block:: shell

    Try to render a mesh with normals (exist: False) and colors (exist: False)
    A mesh with no normals and no colors does not seem good.

.. _surface_normal_estimation:

Surface normal estimation
=====================================

Let's draw geometry with surface normal. Let's continue with following script:

.. code-block:: python

    print("Computing normal, painting the mesh, and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    draw_geometries([mesh])

It uses ``compute_vertex_normals`` and ``paint_uniform_color`` which are member function of ``mesh``.
Now it looks like:

.. image:: ../../_static/Basic/mesh/with_shading.png
    :width: 400px

and prints the following

.. code-block:: shell

    Computing normal, painting the mesh, and rendering it.
    [[ 0.79164373 -0.53951444  0.28674793]
     [ 0.8319824  -0.53303008  0.15389681]
     [ 0.83488162 -0.09250101  0.54260136]
     ...,
     [ 0.16269924 -0.76215917 -0.6266118 ]
     [ 0.52755226 -0.83707495 -0.14489352]
     [ 0.56778973 -0.76467734 -0.30476777]]


Crop mesh
=====================================

``mesh`` has several member variables such as its vertices and indices of vertices for mesh triangles.
These member variables can be tweaked to modify the geometry.
The next script generates a new mesh with half of original surfaces.

.. code-block:: python

    print("We make a partial mesh of only the first half triangles.")
    mesh1 = copy.deepcopy(mesh)
    mesh1.triangles = Vector3iVector(
            np.asarray(mesh1.triangles)[:len(mesh1.triangles)/2, :])
    mesh1.triangle_normals = Vector3dVector(
            np.asarray(mesh1.triangle_normals)[:len(mesh1.triangle_normals)/2, :])
    print(mesh1.triangles)
    draw_geometries([mesh1])

``mesh1 = copy.deepcopy(mesh)`` is for hard copy of ``mesh`` instance.
Note that ``mesh1 = mesh`` just assigns pointer of ``mesh`` to ``mesh1``.

The next line assigns ``mesh1.triangles`` using half of triangles of the original mesh.
It uses following workflow.

1. Transform ``mesh1.triangles`` into numpy array using ``np.asarray()``.
2. Selects the first half of numpy array using ``[:len(mesh1.triangles)/2, :]``
3. Transform numpy array into vector of vectors used for Open3D. ``Vector3iVector()`` constructor used for this purpose here.
4. Assign instance of ``Vector3iVector()`` to ``mesh1``

The same idea is applied for ``mesh1.triangle_normals``, but it uses ``Vector3dVector`` as normal should be double type array.

After assignment, ``draw_geometries`` displays:

.. image:: ../../_static/Basic/mesh/half.png
    :width: 400px


Paint mesh
=====================================

Painting mesh is the same as how it worked for point cloud.
It uses ``paint_uniform_color``.

.. code-block:: python

    print("Painting the mesh")
    mesh1.paint_uniform_color([1, 0.706, 0])
    draw_geometries([mesh1])

``paint_uniform_color`` takes a list of red, green, and blue intensities in range of [0,1].

Now we have:

.. image:: ../../_static/Basic/mesh/half_color.png
    :width: 400px
