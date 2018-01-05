.. _visualization:

Visualization
-------------------------------------

This tutorial introduces useful tips for the visualization of 3D geometries.

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

The first part of the script reads a ply file and render it.


.. _function_draw_geometries:

Function draw_geometries
=====================================

Let's see the first part of the tutorial.

.. code-block:: python

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("../../TestData/fragment.ply")
    draw_geometries([pcd])

``draw_geometries`` have many features that can be very useful for various purposes.
Once the visualization window shows up, press :kbd:`h` to see a hidden help.
It prints the following message on the terminal:

.. code-block:: shell

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

This tutorial introduce few frequently used features among these.


.. _store_view_point:

Store view point
=====================================

The first part of the script displays fragment.ply:

.. image:: ../../_static/Basic/visualization/badview.png
    :width: 400px

After adjusting view points using left button + drag, or mouse scroll,
it is easy to get a better view point:

.. image:: ../../_static/Basic/visualization/color.png
    :width: 400px

If this view point is needed to be memorized, press :kbd:`ctrl+c`.
Next, keep navigating the geometry. It might show:

.. image:: ../../_static/Basic/visualization/newview.png
    :width: 400px

Now press press :kbd:`ctrl+v`. It goes back to the memorized view point below:

.. image:: ../../_static/Basic/visualization/color.png
    :width: 400px


.. _color_map:

Color map
=====================================

Another intersting features of ``draw_geometries`` is changing color map.
From the visualization window, press :kbd:`2`. It shows colored points based on x-coordinate.

.. image:: ../../_static/Basic/visualization/colormap_jet.png
    :width: 400px

``draw_geometries`` provides other color maps worth to try. For example, press :kbd:`shift + 4`.
This changes jet color map to hot color map.

.. image:: ../../_static/Basic/visualization/colormap_hot.png
    :width: 400px

Remember, help messages can be displayed anytime by pressing :kbd:`h`


.. _geometric_premitives:

Geometric premitives
=====================================

The next part of the tutorial script generates geometric premitives.

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
``create_mesh_cylinder``.  The sphere is painted in blue. The cylinder is painted in green.

Another useful premitive is coordinate axis. In this example, ``create_mesh_coordinate_frame``
puts 3D axis on x = -2, y = -2, z = -2. The scale of axis can be adjusted using ``size``.


.. _draw_multiple_geometries:

Draw multiple geometries
=====================================

The last part of this tutorial shows how to visualize multiple geometries.
Consider following script:

.. code-block:: python

    print("We draw a few primitives using collection.")
    draw_geometries([mesh_sphere, mesh_cylinder, mesh_frame])

    print("We draw a few primitives using + operator of mesh.")
    draw_geometries([mesh_sphere + mesh_cylinder + mesh_frame])

``draw_geometries`` takes a list of geometries.
For example, ``[mesh_sphere, mesh_cylinder, mesh_frame]`` displays the three primitives.
Another way is to grouping geometries by using ``+`` operator like ``[mesh_sphere + mesh_cylinder + mesh_frame]``.

Both of function call displays the same geometry like below:

.. image:: ../../_static/Basic/visualization/premitive.png
    :width: 400px
