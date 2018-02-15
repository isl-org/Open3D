.. _working_with_numpy:

Working with NumPy
-------------------------------------

Data structure of Open3D is natively compatible with `NumPy <http://www.numpy.org/>`_ buffer.
The following tutorial generates a variant of sync function using NumPy and visualizes the function using Open3D.

.. code-block:: python

    # src/Python/Tutorial/Basic/working_with_numpy.py

    import sys, copy
    import numpy as np
    sys.path.append("../..")
    from py3d import *

    if __name__ == "__main__":

        # generate some neat n times 3 matrix using a variant of sync function
        x = np.linspace(-3, 3, 401)
        mesh_x, mesh_y = np.meshgrid(x,x)
        z = np.sinc((np.power(mesh_x,2)+np.power(mesh_y,2)))
        xyz = np.zeros((np.size(mesh_x),3))
        xyz[:,0] = np.reshape(mesh_x,-1)
        xyz[:,1] = np.reshape(mesh_y,-1)
        xyz[:,2] = np.reshape(z,-1)
        print('xyz')
        print(xyz)

        # Pass xyz to Open3D.PointCloud and visualize
        pcd = PointCloud()
        pcd.points = Vector3dVector(xyz)
        write_point_cloud("../../TestData/sync.ply", pcd)

        # Load saved point cloud and transform it into NumPy array
        pcd_load = read_point_cloud("../../TestData/sync.ply")
        xyz_load = np.asarray(pcd_load.points)
        print('xyz_load')
        print(xyz_load)

        # visualization
        draw_geometries([pcd_load])

The first part of the script generates a :math:`n \times 3` matrix ``xyz``.
Each column has :math:`x, y, z` value of a function :math:`z = \frac{sin (x^2+y^2)}{(x^2+y^2)}`.

.. _from_numpy_to_open3d:

From NumPy to Open3D
=====================================

.. code-block:: python

    # Pass xyz to Open3D.PointCloud.points and visualize
    pcd = PointCloud()
    pcd.points = Vector3dVector(xyz)
    write_point_cloud("../../TestData/sync.ply", pcd)

Open3D provides conversion from NumPy matrix to a vector of 3D vectors. By using ``Vector3dVector``, NumPy matrix can be directly assigned for ``py3d.PointCloud.points``.

In this manner, any similar data structure such as ``py3d.PointCloud.colors`` or ``py3d.PointCloud.normals`` can be assigned or modified using NumPy. The script saves the point cloud as a ply file for the next step.


.. _from_open3d_to_numpy:

From Open3D to NumPy
=====================================

.. code-block:: python

    # Load saved point cloud and transform it into NumPy array
    pcd_load = read_point_cloud("../../TestData/sync.ply")
    xyz_load = np.asarray(pcd_load.points)
    print('xyz_load')
    print(xyz_load)

    # visualization
    draw_geometries([pcd_load])

As shown in this example, ``Vector3dVector`` is converted into a NumPy array using ``np.asarray``.

The tutorial script prints two identical matrices

.. code-block:: sh

    xyz
    [[-3.00000000e+00 -3.00000000e+00 -3.89817183e-17]
     [-2.98500000e+00 -3.00000000e+00 -4.94631078e-03]
     [-2.97000000e+00 -3.00000000e+00 -9.52804798e-03]
     ...
     [ 2.97000000e+00  3.00000000e+00 -9.52804798e-03]
     [ 2.98500000e+00  3.00000000e+00 -4.94631078e-03]
     [ 3.00000000e+00  3.00000000e+00 -3.89817183e-17]]
    Writing PLY: [========================================] 100%
    Reading PLY: [========================================] 100%
    xyz_load
    [[-3.00000000e+00 -3.00000000e+00 -3.89817183e-17]
     [-2.98500000e+00 -3.00000000e+00 -4.94631078e-03]
     [-2.97000000e+00 -3.00000000e+00 -9.52804798e-03]
     ...
     [ 2.97000000e+00  3.00000000e+00 -9.52804798e-03]
     [ 2.98500000e+00  3.00000000e+00 -4.94631078e-03]
     [ 3.00000000e+00  3.00000000e+00 -3.89817183e-17]]

and visualizes the function:

.. image:: ../../_static/Basic/working_with_numpy/sync.png
    :width: 400px
