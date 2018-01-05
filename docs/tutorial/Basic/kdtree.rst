.. _kdtree:

KDTree
-------------------------------------

KDTree is core data structure used for fast retrieval of adjacent points or 3D features. This tutorial addresses basic usage of KDTree.

.. code-block:: python

    # src/Python/Tutorial/Basic/kdtree.py

    import sys
    import numpy as np
    sys.path.append("../..")
    from py3d import *

    if __name__ == "__main__":

        print("Testing kdtree in py3d ...")
        print("Load a point cloud and paint it gray.")
        pcd = read_point_cloud("../../TestData/Feature/cloud_bin_0.pcd")
        pcd.paint_uniform_color([0.5, 0.5, 0.5])
        pcd_tree = KDTreeFlann(pcd)

        print("Paint the 1500th point red.")
        pcd.colors[1500] = [1, 0, 0]

        print("Find its 200 nearest neighbors, paint blue.")
        [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
        np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

        print("Find its neighbors with distance less than 0.2, paint green.")
        [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
        np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

        print("Visualize the point cloud.")
        draw_geometries([pcd])
        print("")

.. _build_kdtree_from_pointcloud:

Build KDTree from point cloud
=====================================

.. code-block:: python

    print("Testing kdtree in py3d ...")
    print("Load a point cloud and paint it gray.")
    pcd = read_point_cloud("../../TestData/Feature/cloud_bin_0.pcd")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])
    pcd_tree = KDTreeFlann(pcd)

This script reads a point cloud. A method named ``paint_uniform_color`` of ``pcd`` paints its points as gray ``[0.5, 0.5, 0.5]``. With ``pcd``, ``KDTreeFlann`` constructs KDTree data structure ``pcd_tree``.


.. _find_neighboring_points:

Find neighboring points
=====================================

Let's use ``pcd_tree`` for searching neighboring points. Consider a case finding neighbors of the 1500th point in the point cloud. Let's mark 1500th point first:

.. code-block:: python

    print("Paint the 1500th point red.")
    pcd.colors[1500] = [1, 0, 0]

The script shows how to directly access member value ``colors``. Each point has color in [0,1] scale with red, green, and blue channel order. Therefore, ``[1, 0, 0]`` means pure red.


Using search_knn_vector_3d
``````````````````````````````````````

.. code-block:: python

    print("Find its 200 nearest neighbors, paint blue.")
    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[1500], 200)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 0, 1]

This script calls a function ``search_knn_vector_3d`` of ``pcd_tree``. It searches N-nearest neighbors regardless how far it is from the query point. The input arguments are:

- a query point: ``pcd.points[1500]``

- how many adjacent points are going to be found: ``200``

``search_knn_vector_3d`` returns:

- how many points are actually found: ``k``

- list of neighboring point index: ``idx``.

Note that ``k`` can be less than requested if a point cloud has less than 200 points.

Given the list of adjacent points, neighboring points are painted in blue. ``np.asarray(pcd.colors)`` transforms ``pcd.colors`` into numpy array to make batch access of point colors. ``[idx[1:], :]`` indicates the second to 200th neighboring points. They get blue color. Note that the script ignores the first neighboring points because it is query point itself.


Using search_radius_vector_3d
``````````````````````````````````````

.. code-block:: python

    print("Find its neighbors with distance less than 0.2, paint green.")
    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[1500], 0.2)
    np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]

The next script calls ``search_radius_vector_3d``. This function finds a neighboring points of a query point within a specified radius. In this example, a query point is ``pcd.points[1500]`` and searching radius is ``0.2``. The next line ``np.asarray(pcd.colors)[idx[1:], :] = [0, 1, 0]`` paints the found the adjacent points in green.

.. code-block:: python

    print("Visualize the point cloud.")
    draw_geometries([pcd])
    print("")

Finally, it visualizes colored point cloud:

.. image:: ../../_static/Basic/kdtree/kdtree.png
    :width: 400px

Note that 1500th point is colored in red, and its 199 neighbors are colored in blue, and neighbors within 0.2 distance to 1500th point are colored in green.
