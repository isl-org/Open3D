.. _visualization:

Visualization
-------------------------------------

.. code-block:: python

    # src/Python/Tutorial/Basic/visualization.py

    import sys, copy
    import numpy as np
    sys.path.append("../..")
    from py3d import *

    if __name__ == "__main__":

        print("Load a ply point cloud, print it, and render it")
        pcd = read_point_cloud("../../TestData/fragment.ply")
        draw_geometries([pcd])

        print('Lets draw some primitives')
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


.. _function_draw_geometries:

Function draw_geometries
=====================================

.. code-block:: python

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("../../TestData/fragment.ply")
    draw_geometries([pcd])

Open3D provides a convenient visualization function ``draw_geometries`` which takes a list of geometry objects (``PointCloud``, ``TriangleMesh``, or ``Image``), and renders them together. We have implemented many functions in the visualizer, such as rotation, translation, and scaling via mouse operations, changing rendering style, and screen capture. Press :kbd:`h` inside the window to print out a comprehensive list of functions.

.. code-block:: sh

    -- Mouse view control --
      Left button + drag        : Rotate.
      Ctrl + left button + drag : Translate.
      Wheel                     : Zoom in/out.

    -- Keyboard view control --
      [/]          : Increase/decrease field of view.
      R            : Reset view point.
      Ctrl/Cmd + C : Copy current view status into the clipboard.
      Ctrl/Cmd + V : Paste view status from clipboard.

    -- General control --
      Q, Esc       : Exit window.
      H            : Print help message.
      P, PrtScn    : Take a screen capture.
      D            : Take a depth capture.
      O            : Take a capture of current rendering settings.
    :

.. Note:: In some operating systems (e.g., OS X), the visualization window may not respond to keyboard input. This is usually because the console retains the input focus instead of passing it to the visualization window. Calling ``pythonw visualization.py`` instead of ``python visualization.py`` will resolve this issue.

.. Note:: In addition to ``draw_geometries``, Open3D has a set of sibling functions with more advanced functionality. ``draw_geometries_with_custom_animation`` allows the programmer to define a custom view trajectory and play an animation in the GUI. ``draw_geometries_with_animation_callback`` and ``draw_geometries_with_key_callback`` accept Python callback functions as input. The callback function is called in an automatic animation loop, or upon a key press event. See :ref:`customized_visualization` for details.

.. _store_view_point:

Store view point
=====================================

In the beginning, the point cloud is rendered upside down.

.. image:: ../../_static/Basic/visualization/badview.png
    :width: 400px

After adjusting view points using mouse left button + drag, we can reach a better view point.

.. image:: ../../_static/Basic/visualization/color.png
    :width: 400px

To retain this view point, press :kbd:`ctrl+c`. The view point will be translated into a json string stored in clipboard. When you move the camera to a different view, such as:

.. image:: ../../_static/Basic/visualization/newview.png
    :width: 400px

You can get back to the original view by pressing :kbd:`ctrl+v`.

.. image:: ../../_static/Basic/visualization/color.png
    :width: 400px

.. _rendering_style:

Rendering styles
=====================================

Open3D ``Visualizer`` supports several rendering styles. For example, pressing :kbd:`l` will switch between a Phong lighting and a simple color rendering. Pressing :kbd:`2` shows points colored based on x-coordinate.

.. image:: ../../_static/Basic/visualization/colormap_jet.png
    :width: 400px

The color map can also be adjusted by, for example, pressing :kbd:`shift+4`. This changes jet color map to hot color map.

.. image:: ../../_static/Basic/visualization/colormap_hot.png
    :width: 400px

.. _geometry_primitives:

Geometry primitives
=====================================

.. code-block:: python

    print('Lets draw some primitives')
    mesh_sphere = create_mesh_sphere(radius = 1.0)
    mesh_sphere.compute_vertex_normals()
    mesh_sphere.paint_uniform_color([0.1, 0.1, 0.7])
    mesh_cylinder = create_mesh_cylinder(radius = 0.3, height = 4.0)
    mesh_cylinder.compute_vertex_normals()
    mesh_cylinder.paint_uniform_color([0.1, 0.9, 0.1])
    mesh_frame = create_mesh_coordinate_frame(size = 0.6, origin = [-2, -2, -2])

This script generates a sphere and a cylinder using ``create_mesh_sphere`` and
``create_mesh_cylinder``.  The sphere is painted in blue. The cylinder is painted in green. Normals are computed for both meshes to support the Phong shading (see :ref:`visualize_3d_mesh` and :ref:`surface_normal_estimation`). We can even create a coordinate axis using ``create_mesh_coordinate_frame``, with its origin point set at (-2, -2, -2).

.. _draw_multiple_geometries:

Draw multiple geometries
=====================================

.. code-block:: python

    print("We draw a few primitives using collection.")
    draw_geometries([mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    draw_geometries([mesh_sphere + mesh_cylinder + mesh_frame])

``draw_geometries`` takes a list of geometries and renders them all together. Alternatively, ``TriangleMesh`` supports a ``+`` operator to combine multiple meshes into one. We recommend the first approach since it supports a combination of different geometries (e.g., a mesh can be rendered in tandem with a point cloud).

.. image:: ../../_static/Basic/visualization/premitive.png
    :width: 400px
