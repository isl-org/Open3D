.. _python_binding_tutorial:

Python binding
-------------------------------------

This tutorial introduces how to import Open3D module in Python environment.
If Open3D is successfully compiled with Python binding option,
py3d.so should be visible under Open3D build folder.
If it is not, please go over :ref:`python_binding`.

This tutorial addresses a script below.

.. code-block:: python

    # src/Python/Tutorial/Basic/python_binding.py

    import sys
    import numpy as np
    sys.path.append("../..")

    def example_help_function():
        import py3d as py3d
        help(py3d)
        help(py3d.PointCloud)
        help(py3d.read_point_cloud)

    def example_import_function():
        from py3d import read_point_cloud
        pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        print(pcd)

    if __name__ == "__main__":
        example_help_function()
        example_import_function()

This scripts has of two functions: ``example_help_function`` and ``example_import_all``
that show fundamental usage example of Open3D Python module.
In the heading, it uses ``sys.path.append()`` to refer the path where py3d.so is located.

Let's consider each function from now on.


.. _import_py3d_module:

Import py3d module
=====================================

.. code-block:: python

    def example_import_function():
        from py3d import read_point_cloud
        pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
        print(pcd)

This script imports ``read_point_cloud`` from py3d module.
This is preferable choice for compact system resource usage.
To use all functions and classes in py3d,
``from py3d import *`` should be placed in heading of a Python script.

``read_point_cloud`` reads point cloud file and returns instance of ``PointCloud`` class, namely ``pcd``.
``print(pcd)`` prints brief information of ``pcd`` like below:

.. code-block:: shell

    PointCloud with 198835 points.



.. _using_builtin_help_function:

Using built-in help function
=====================================

It is highly recommended to use Python built-in ``help`` function to browse
Open3D function definitions, input argument, and classes so on. Here is an example.

.. code-block:: python

    def example_help_function():
        import py3d as py3d
        help(py3d)
        help(py3d.PointCloud)
        help(py3d.read_point_cloud)

This script imports ``py3d`` module and defines scope ``py3d``.


Browse py3d
``````````````````````````````````````

``help(py3d)`` prints documents of ``py3d`` module like below:

.. code-block:: shell

    Help on module py3d:

    NAME
        py3d - Python binding of Open3D

    FILE
        /Users/myaccount/Open3D/build/lib/py3d.so

    CLASSES
        __builtin__.object
            CorrespondenceChecker
                CorrespondenceCheckerBasedOnDistance
                CorrespondenceCheckerBasedOnEdgeLength
                CorrespondenceCheckerBasedOnNormal
            DoubleVector
            Feature
            Geometry
                Geometry2D
                    Image
                Geometry3D
                    PointCloud
                    TriangleMesh
    :


Description of a class in py3d
``````````````````````````````````````

The next script, ``help(py3d.PointCloud)`` provides description of ``PointCloud`` class.

.. code-block:: shell

    Help on class PointCloud in module py3d:

    class PointCloud(Geometry3D)
     |  Method resolution order:
     |      PointCloud
     |      Geometry3D
     |      Geometry
     |      __builtin__.object
     |
     |  Methods defined here:
     |
     |  __add__(...)
     |      __add__(self: py3d.PointCloud, arg0: py3d.PointCloud) -> py3d.PointCloud
     |
    :


Description of a function in py3d
``````````````````````````````````````

The last script, ``help(py3d.read_point_cloud)`` provides description of input argument and return type of ``read_point_cloud`` function.

.. code-block:: shell

    Help on built-in function read_point_cloud in module py3d:

    read_point_cloud(...)
        read_point_cloud(filename: unicode) -> py3d.PointCloud

        Function to read PointCloud from file

As it is shown above, ``read_point_cloud`` takes unicode string for filename and returns ``py3d.PointCloud`` instance.
