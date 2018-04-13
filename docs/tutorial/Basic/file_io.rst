.. _file_io:

File IO
-------------------------------------

This tutorial shows how basic geometries are read and written by Open3D.

.. code-block:: python

    # src/Python/Tutorial/Basic/io.py

    from open3d import *

    if __name__ == "__main__":

        print("Testing IO for point cloud ...")
        pcd = read_point_cloud("../../TestData/fragment.pcd")
        print(pcd)
        write_point_cloud("copy_of_fragment.pcd", pcd)

        print("Testing IO for meshes ...")
        mesh = read_triangle_mesh("../../TestData/knot.ply")
        print(mesh)
        write_triangle_mesh("copy_of_knot.ply", mesh)

        print("Testing IO for images ...")
        img = read_image("../../TestData/lena_color.jpg")
        print(img)
        write_image("copy_of_lena_color.jpg", img)

.. _io_point_cloud:

Point cloud
=====================================

This script reads and writes a point cloud.

.. code-block:: python

    print("Testing IO for point cloud ...")
    pcd = read_point_cloud("../../TestData/fragment.pcd")
    print(pcd)
    write_point_cloud("copy_of_fragment.pcd", pcd)

``print()`` function can be used for displaying a summary of ``pcd``. Output message is below:

.. code-block:: sh

    Testing IO for point cloud ...
    PointCloud with 113662 points.


.. _io_mesh:

Mesh
=====================================

This script reads and writes a mesh.

.. code-block:: python

    print("Testing IO for meshes ...")
    mesh = read_triangle_mesh("../../TestData/knot.ply")
    print(mesh)
    write_triangle_mesh("copy_of_knot.ply", mesh)

Compared to the data structure of point cloud, mesh has triangles that define surface.

.. code-block:: sh

    Testing IO for meshes ...
    TriangleMesh with 1440 points and 2880 triangles.


.. _io_image:

Image
=====================================

This script reads and writes an image.

.. code-block:: python

    print("Testing IO for images ...")
    img = read_image("../../TestData/lena_color.jpg")
    print(img)
    write_image("copy_of_lena_color.jpg", img)

Size of image is readily displayed using ``print(img)``.

.. code-block:: sh

    Testing IO for images ...
    Image of size 512x512, with 3 channels.
    Use numpy.asarray to access buffer data.
