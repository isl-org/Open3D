.. _open3d_visualizer:

Open3D Visualizer
=================

Introduction
---------------

Open3D provides a convenient function for visualizing geometric objects: ``draw``. The ``draw`` function allows you to visualize multiple geometry objects *(PointClouds, LineSets, TriangleMesh)* and images together along with optional, high-quality, physically based (PBR) materials. As will be demonstrated in the subsequent sections, ``draw`` can be used for both - simple, quick visualization or complex use-cases.


Overview
--------

.. note::
	 This **Overview** section applies to all subsequent examples below
	 
For all examples in this tutorial, we will be running a Python session. Please follow these preliminary steps :

1. First, open a command-line terminal. From there, Change Directory (``cd``) to ``Open3D``:
 
.. code-block:: sh

	$ cd <... Path to Open3D on your computer...>
	
.. image:: https://user-images.githubusercontent.com/41028320/146986379-57328766-d4b9-4858-81bb-761888660814.jpg
    :width: 700px	
    
2. **Optionally**, if you have a ``conda`` virtual environment, activate it from the command line like so:

.. code-block:: sh

    $ conda activate <...your virtual environment name...>
    
3. Run the ``python`` command:

.. code-block:: sh

    $ python

4. At the python prompt, enter the following line to create an Open3D object:

.. code-block:: python

		>>> import open3d as o3d
		
This ``o3d`` object will be used throughout the following examples.


Examples
--------

In the Overview section, we activated a ``conda`` environment, started a Python session, and declared an Open3D object to be used throughout this tutorial. Let’s now test various Open3D ``draw()`` function capabilities with various geometries.

Drawing a Triangle Mesh
:::::::::::::::::::::::

This example shows how to create and visualize a simple 3D box.


At the python prompt, enter the following four lines to open the 3D Visualizer:

.. code-block:: python

		>>> import open3d.visualization as vis
		>>> cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
		>>> vis.draw(cube)

At the end of the process, the Open3D Visualizer window should appear:

.. image:: ../../_static/visualization/open3d_visualizer/1_Cube.jpg
    :width: 600px
    

Drawing a Sphere
::::::::::::::::

At the Python prompt in your terminal, enter the following lines of code:

.. code-block:: python

		>>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0)
		>>> o3d.visualization.draw(sphere)
		
A rendered sphere appears:

.. image:: ../../_static/visualization/open3d_visualizer/2_Sphere.jpg
    :width: 600px

To see what type of rendering was used to draw our sphere above, at the Python prompt, enter: 

.. code-block:: python
	
		>>> sphere

Open3D returns:

.. code-block:: sh
	
		TriangleMesh with 762 points and 1520 triangles.



Drawing a Tensor-based Sphere
:::::::::::::::::::::::::::::

In the example above we rendered a TriangleMesh version of sphere. Now, we will do the same using a Tensor-based object. Continuing from the previous example, at the Python prompt in your terminal, enter:

.. code-block:: python

		>>> sphere_t = o3d.t.geometry.TriangleMesh.from_legacy(sphere)
		>>> o3d.visualization.draw(sphere_t)
		
		
A sphere similar to that shown in the previous example is rendered, but this time using a Tensor data type. Now, enter ``sphere_t`` at the Python prompt:

.. code-block:: python

		>>> sphere_t

Open3D returns:

.. code-block:: sh

	TriangleMesh on CPU:0 [762 vertices (Float32) and 1520 triangles (Int64)]. 
	Vertices Attributes: None. 
	Triangles Attributes: None.

**This is how Tensor objects are denoted - by vertices and triangles.**



Rendering Pont Cloud Data (\*.pcd) files
::::::::::::::::::::::::::::::::::::::::

Enter the following code at the Python prompt:

.. code-block:: python

	>>> pcd = o3d.io.read_point_cloud("examples/test_data/fragment.pcd")
	>>> o3d.visualization.draw(pcd)
	
Open3D returns:
	
.. image:: ../../_static/visualization/open3d_visualizer/4.PCD.jpg
    :width: 600px
    

Working with Line Sets
::::::::::::::::::::::::

Specifying Wireframe ``line_width``
"""""""""""""""""""""""""""""""""""
   
Drawing Multiple Objects
::::::::::::::::::::::::

The ``draw()`` function can be used to render multiple 3D objects simultaneously. You can pass as may objects to the ``draw()`` as you need. In this example, we will render two objects: the **Sphere** and the **PCD**. 


At the Python prompt, enter these lines of code:

.. code-block:: python

	>>> o3d.visualization.draw([sphere, pcd])
	
Both objects appear and can be moved and rotated:

.. image:: ../../_static/visualization/open3d_visualizer/5.Multiple_obj.jpg
    :width: 600px
	
Objects can also be separated from each other by specifying distance. In the code below, we are separating the **sphere** from the **PCD**:

.. code-block:: python

	>>> sphere.translate([0, 6, 0])
  TriangleMesh with 762 points and 1520 triangles.
  >>> o3d.visualization.draw([sphere, pcd])


As you can see, this time, our objects are separated by a greater distance, and just like in the previous example, they can be moved, panned, and rotated:

.. image:: ../../_static/visualization/open3d_visualizer/5a.Sep_Multiple_obj.png
    :width: 600px
	
Displaying UI / Control Panel
"""""""""""""""""""""""""""""

By default, the ``draw()`` function renders 3D models without showing the user interface (UI) / control panel where users can interactively modify various rendering parameters of the visualizer. Let's now render our models with the UI shown:

.. code-block:: python

	>>> o3d.visualization.draw([sphere, pcd], show_ui=True)

.. image:: ../../_static/visualization/open3d_visualizer/5b.Multiple_obj_UI.jpg
    :width: 600px

At the bottom of the UI / control panel, you can see the section titled "*Geometries*" (outlined in a yellow box). This section contains a list of rendered objects that can be individually turned on or off by clicking a checkbox to the left of their names.
 

Working with Geometries and Materials
:::::::::::::::::::::::::::::::::::::

With the ``draw()`` function you can create customized geometries and materials. Let's see how this is done:

.. code-block:: python

	>>> geoms = {'name': 'sphere', 'geometry': sphere, 'material': mat}
  >>> o3d.visualization.draw(geoms)
  
.. image:: ../../_static/visualization/open3d_visualizer/6.Geoms.png
    :width: 600px
    
Note that after the ``draw()`` call of ``o3d.visualization.draw(geoms)`` Open3D displays a warning related to the absence of ``normals``:

.. code-block:: python

  [Open3D WARNING] Using a shader with lighting but geometry has no normals.
  
As you can see from the above image, the sphere shading looks somewhat jagged and to fix that, we need to call a method ``compute_vertex_normals()`` on our sphere object:

.. code-block:: python

	>>> sphere.compute_vertex_normals()
  TriangleMesh with 762 points and 1520 triangles.
  >>> o3d.visualization.draw(geoms)
  
This time, because we used ``compute_vertex_normals()``, the rendered sphere looks way better:

.. image:: ../../_static/visualization/open3d_visualizer/6a.Geoms_w_compute_normals.png
    :width: 600px

Compute Vertex Normals Method
"""""""""""""""""""""""""""""

Assigning Names to Multiple Objects in the UI
"""""""""""""""""""""""""""""""""""""""""""""

More ``draw()`` Options
:::::::::::::::::::::::

``show_skybox`` and ``bg_color`` Options
""""""""""""""""""""""""""""""""""""""""

Specifying ``point_size``
"""""""""""""""""""""""""



Running ``draw.py``
:::::::::::::::::::


``draw()`` Parameters
------------------------

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

