.. _open3d_visualizer:

Open3D Visualizer
=================

Introduction
---------------

Open3D provides a convenient function for visualizing geometric objects: ``draw``. The ``draw`` function allows you to visualize multiple geometry objects *(PointClouds, LineSets, TriangleMesh)* and images together along with optional, high-quality, physically based (PBR) materials. As will be demonstrated in the subsequent sections, ``draw`` can be used for both - simple, quick visualization or complex use-cases.


Examples
--------



Basics
::::::

This example shows how to create and visualize a simple 3D box.

First, open a command-line terminal:

.. image:: ../../_static/visualization/open3d_visualizer/vis_terminal_window.png
    :width: 700px
    
.. image:: https://asciinema.org/a/UwKbLM68mXg5khtFGLre7JXSe.svg

`Click here to play video <https://asciinema.org/a/UwKbLM68mXg5khtFGLre7JXSe>`_

Then:

1. **Optionally**, if you have a ``conda`` virtual environment, activate it from the command line like so:

.. code-block:: sh

    $ conda activate <...your virtual environment name...>
    
2. Run the ``python`` command:

.. code-block:: sh

    $ python

3. At the python prompt, enter the following four lines to open the 3D Visualizer:

.. code-block:: python

		import open3d as o3d
		import open3d.visualization as vis

		cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
		vis.draw(cube)

At the end of the process, the Open3D Visualizer window should appear:

.. image:: ../../_static/visualization/open3d_visualizer/vis_model_viewer.png
    :width: 600px

``draw`` documentation
----------------------

.. code-block:: python

    def draw(geometry=None,
             title="Open3D",
             width=1024,
             height=768,
             actions=None,
             lookat=None,
             eye=None,
             up=None,
             ield_of_view=60.0,
             bg_color=(1.0, 1.0, 1.0, 1.0),
             bg_image=None,
             ibl=None,
             ibl_intensity=None,
             show_skybox=None,
             show_ui=None,
             point_size=None,
             animation_time_step=1.0,
             animation_duration=None,
             rpc_interface=False,
             on_init=None,
             on_animation_frame=None,
             on_animation_tick=None,
             non_blocking_and_return_uid=False):
             
``draw`` Input Parameters
:::::::::::::::::::::::::
             
             
	``geometry`` Rene - description

	``title`` Rene - description

	``width`` Rene - description

	``height``  Rene - description

	``actions`` Rene - description

	``lookat`` Rene - description

	``eye`` Rene - description

	``up`` Rene - description

	``ield_of_view`` Rene - description // !!! Should this be spelled "field_of_view" ???

	``bg_color`` Rene - description

	``bg_image`` Rene - description

	``ibl`` Rene - description

	``ibl_intensity`` Rene - description

	``show_skybox`` Rene - description

	``show_ui`` Rene - description

	``point_size`` Rene - description

	``animation_time_step`` Rene - description

	``animation_duration`` Rene - description

	``rpc_interface`` Rene - description

	``on_init`` Rene - description

	``on_animation_frame`` Rene - description

	``on_animation_tick`` Rene - description

	``non_blocking_and_return_uid`` Rene - description

