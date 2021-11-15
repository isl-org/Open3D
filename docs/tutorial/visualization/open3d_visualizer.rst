.. _open3d_visualizer:

Open3D Visualizer
=================

Introduction
---------------

Rene: Text Goes Here

Pre-requisites
::::::::::::::

 - Python versions 3.6, 3.7, 3.8, or 3.9
 - Optional: ``conda`` virtual environments


Examples
--------



Basics
::::::

This example shows how to create and visualize a simple 3D box. To run it:

1. Open a command-line terminal;
#. **Optionally**, if you have a ``conda`` virtual environment, activate it from the command line like so:

.. code-block:: sh

    $ conda activate <...your virtual environment name...>
    
3. Run the ``python`` command:

.. code-block:: sh

    $ python

4. At the python prompt, enter the following four lines to open the 3D Visualizer:

.. code-block:: python

		import open3d as o3d
		import open3d.visualization as vis

		cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
		vis.draw(cube)

Input Parameters
----------------
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
