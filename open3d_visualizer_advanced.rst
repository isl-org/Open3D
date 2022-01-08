.. _open3d_visualizer_advanced:

Advanced Open3D Visualizer
==========================

Introduction
---------------

Open3D provides a convenient function for visualizing geometric objects: ``draw``. The ``draw`` function allows you to visualize multiple geometry objects *(PointClouds, LineSets, TriangleMesh)* and images together along with optional, high-quality, physically based (PBR) materials. As will be demonstrated in the subsequent sections, ``draw`` can be used for both - simple, quick visualization or complex use-cases.


Getting Started
---------------

.. note::
	 This **Getting Started** section applies to all subsequent examples below
	 
For all examples in this tutorial, we will be running a Python session. Please follow these preliminary steps :

1. First, open a command-line terminal. From there, Change Directory (``cd``) to ``Open3D``:
 
.. code-block:: sh

	$ cd <... Path to Open3D on your computer...>
	
.. image:: https://user-images.githubusercontent.com/93158890/148607427-9391c499-9fc5-4a38-89a4-fb088019ca0b.jpg
    :width: 700px	
    
2. **Optionally**, if you have a ``conda`` virtual environment, activate it from the command line like so:

.. code-block:: sh

    $ conda activate <...your virtual environment name...>
    
3. Run the ``python`` command:

.. code-block:: sh

    $ python

4. At the Python prompt, enter the following lines to create Open3D objects:

.. code-block:: python

		>>> import open3d as o3d
		>>> import open3d.visualization as vis
		>>> import numpy as np
		
These objects will be used throughout the following examples.


Advanced Examples
-----------------

In *Basic Examples*, we covered how to render Tensor and TriangleMesh shapes, raster models, and how to control their display programmatically via code and interactively by using Open3D Visualizer UI. This section expounds on those topics to cover more advanced visualization techniques.

Rendering a Tensor-based TriangleMesh Monkey
::::::::::::::::::::::::::::::::::::::::::::

At your python prompt, enter:

.. code-block:: python

		>>> monkey = o3d.io.read_triangle_mesh('examples/test_data/monkey/monkey.obj')

Here we are invoking the ``open3d.io`` library which allows us to read 3D model files and/or selectively extract their details. In this case, we are using the ``read_triangle_mesh()`` method for extracting the ``monkey.obj`` file ``TriangleMesh`` data. Since we can't load this object directly, we will convert it into **Open3D Tensor geometry**:

.. code-block:: python

		>>> monkey = o3d.t.geometry.TriangleMesh.from_legacy(monkey)

Let's see what properties ``monkey`` has:

.. code-block:: python

    >>> monkey
    TriangleMesh on CPU:0 [9908 vertices (Float32) and 15744 triangles (Int64)].
    Vertex Attributes: normals (dtype = Float32, shape = {9908, 3}).
    Triangle Attributes: texture_uvs (dtype = Float32, shape = {15744, 3, 2}).
		
Time to render the ``monkey``:

.. code-block:: python

		>>> vis.draw(monkey)

And we get:

.. image:: https://user-images.githubusercontent.com/93158890/148610827-4a8dc85f-5664-4f7a-b0da-1808387c9f71.jpg
    :width: 600px

Now, let's work on materials:

.. code-block:: python

    >>> mat = vis.rendering.MaterialRecord()
    >>> mat.base_color = np.asarray([1.0, 1.0, 0.0, 1.0])
    >>> vis.draw({'name': 'monkey', 'geometry': monkey, 'material': mat})
    
We have initialized ``mat.base_color`` to be yellow and get:

.. image:: https://user-images.githubusercontent.com/93158890/148610882-14e6d348-1e8e-4bd9-b0ef-90fa884d9706.jpg
    :width: 600px

Obviously, this looks ugly because the material (``mat``) lacks shading. To correct our 3D rendering, we use ``mat.shader`` property:

.. code-block:: python

    >>> mat.shader = 'defaultLit'
    >>> vis.draw({'name': 'monkey', 'geometry': monkey, 'material': mat})

This time, we see a big difference because the ``mat.shader`` property is initialized:

.. image:: https://user-images.githubusercontent.com/93158890/148611064-2fa5fe4c-b8cb-4588-ad46-df23cdf160be.jpg
    :width: 600px

You can experiment with different material colors to your liking by changing numeric values in the ``mat.base_color = np.asarray([1.0, 1.0, 0.0, 1.0])`` statement.




Rendering Models
::::::::::::::::

Up to this point, we have been rendering *TriangleMesh* and *Tensor-based TriangleMesh* objects. But the ``draw()`` function can also render full-fledged 3D models containing a set of textures and material properties. To read a complete model, we need to use the ``open3d.io.read_triangle_model()`` method, which imports all the material properties in addition to the *TriangleMesh*:

.. code-block:: python

    >>> monkey_model = o3d.io.read_triangle_model('examples/test_data/monkey/monkey.obj')
    >>> vis.draw(monkey_model)

Clearly, a staggering difference in rendering:

.. image:: https://user-images.githubusercontent.com/93158890/148611141-d424fc74-be7e-4833-913c-714fc3c4fbd2.jpg
    :width: 600px



Rendering Monkey Wireframe ``LineSet``
::::::::::::::::::::::::::::::::::::::

In order to render a given 3D model's wireframe, we need to:

1. extract its regular ``TriangleMesh`` information. Let's re-initialize our monkey object and check to see its current type:

.. code-block:: python

    >>> monkey = o3d.io.read_triangle_mesh('examples/test_data/monkey/monkey.obj')
    >>> monkey
    TriangleMesh with 9908 points and 15744 triangles.



2. Now that our *monkey* object is of regular ``TriangleMesh``, it's time to create a ``LineSet`` object from it. We will also color it blue with the ``paint_uniform_color()`` method. Then, we'll render it with ``draw()``:

.. code-block:: python

    >>> monkey_ls = o3d.geometry.LineSet.create_from_triangle_mesh(monkey)
    >>> monkey_ls.paint_uniform_color([0.0, 0.0, 1.0])
    >>> vis.draw(monkey_ls)
    
.. image:: https://user-images.githubusercontent.com/93158890/148611269-78820f1d-b981-44a6-bb08-60c17d0bb45f.jpg
    :width: 600px

Let's check to see what type of object ``monkey_ls`` is:

.. code-block:: python

    >>> monkey_ls
    LineSet with 25556 lines.



3. Let's convert *TriangleMesh LineSets* into *Tensor-based TriangleMesh* ones:

.. code-block:: python

    >>> monkey_ls = o3d.t.geometry.LineSet.from_legacy(monkey_ls)
    >>> monkey_ls
    LineSet on CPU:0
    [9908 points (Float32)] Attributes: None.
    [25556 lines (Int64)] Attributes: colors (dtype = Float32, shape = {25556, 3}).

Great. ``monkey_ls`` is now a ``t.geometry.LineSet`` (*Tensor-based LineSet*).


We can also change the ``line_width`` parameter for our wireframe. For this excercise, we'll make it thinner (``line_width=1``):

.. code-block:: python

    >>> vis.draw(monkey_ls, line_width=1)

.. image:: https://user-images.githubusercontent.com/93158890/148611385-cadcc6c9-a648-4775-a1b0-c6e543eea254.jpg
    :width: 600px

Experiment with different ``line_width`` values to see which one looks best for your purposes.


Scaling Wireframes
""""""""""""""""""

If you need to superimpose a wireframe *LineSet* on top of a 3D object, the way to do it is through scaling the wireframe to be a tiny bit bigger than the underlying 3D object. For such cases, a ``LineSet_object.scale()`` method is used. Let's see how we would do it with both - the monkey object and its wireframe:

.. code-block:: python

    >>> monkey_ls.scale(1.02, np.asarray([0, 0, 0]))
    LineSet on CPU:0
    [9908 points (Float32)] Attributes: None.
    [25556 lines (Int64)] Attributes: colors (dtype = Float32, shape = {25556, 3}).

We have just scaled the wireframe ``LineSet`` to be 2% larger. Now, let's render both - the wireframe (``monkey_ls``) and the underlying ``monkey`` object:

.. code-block:: python

    >>> vis.draw([monkey, monkey_ls])

.. image:: https://user-images.githubusercontent.com/93158890/148611540-7f4ea545-18bc-4562-b5c8-fb9e7cf60452.jpg
    :width: 600px

The above image shows a zoomed-in fragment of our model where we can clearly see some space between the wireframe and the object. Experiment with scale values further to see different visual results.




More Complex Models
:::::::::::::::::::

In the previous section (**Rendering Models**) we have covered how to render complete 3D models with the ``open3d.io.read_triangle_model()`` method. This method can also handle more complex models containing a collection of materials and parts (sub-models) from which the complete object gets assembled.

For this example, we will need to download / ``clone`` *glTF-Sample-Models*  from the KhronosGroup. `glTF (GL Transmission Format) <https://docs.fileformat.com/3d/gltf/>`_ is a 3D file format that stores 3D model information in JSON format. 

First, **minimize your current Python terminal session and open a new one. In a new terminal session:**

.. image:: https://user-images.githubusercontent.com/93158890/148611673-8af22794-75b0-49a6-babe-d0b50578c570.jpg
    :width: 800px

1. Change Directory (``cd``) to where you would like the *glTF-Sample-Models* repository to be copied;
2. Use the ``git clone`` command to download the *glTF-Sample-Models* repository:

.. code-block:: sh

    $ git clone https://github.com/KhronosGroup/glTF-Sample-Models

3. Wait for the cloning process to complete. The command prompt will return when the process is done.
4. Close the command prompt window you've just used for the ``git clone`` command.

Now that we have all *glTF-Sample-Models* files in place, let's switch back to our Python terminal session and load the model of a WWII-era flight helmet:

.. code-block:: python

    >>> helmet = o3d.io.read_triangle_model('../glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf')
    >>> vis.draw(helmet)
    
.. note::
   In your case, the *glTF-Sample-Models* directory location may be different, depending on where you chose to clone it.

.. image:: https://user-images.githubusercontent.com/93158890/148611761-40f95b2b-d257-4f2b-a8c0-60a73b159b96.jpg
    :width: 600px

We've just rendered a complex model - this one actually consists of multiple sub-models with multiple types of materials and textures in it, that can each be rendered separately as we will see shortly.

This and other complex models can also be rendered using the ``o3d.io.read_triangle_mesh()`` method. However, as we will see below, this  yields inferior results:

.. code-block:: python

    >>> helmet = o3d.io.read_triangle_mesh('../glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf')
    >>> vis.draw(helmet)

.. image:: https://user-images.githubusercontent.com/93158890/148611814-09c6fe17-d209-439d-8ae9-c186387fd698.jpg
    :width: 600px

.. note::
   For complex model rendering, please use the ``o3d.io.read_triangle_model()``, rather than ``read_triangle_mesh()``. ``read_triangle_mesh()`` is only good for loading basic meshes, but not complex materials.


Examining Complex Models
::::::::::::::::::::::::

Let's re-load our *FlightHelmet.gltf* model with ``o3d.io.read_triangle_model()``:

.. code-block:: python

    >>> helmet = o3d.io.read_triangle_model('../glTF-Sample-Models/2.0/FlightHelmet/glTF/FlightHelmet.gltf')

Take a look at what the ``helmet`` object consists of. First, we find out its type:

.. code-block:: python

    >>> helmet
    <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel object at 0x7f019efa7770>

Now, we'll look at its meshes:

.. code-block:: python

    >>> helmet.meshes
    [<open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0134034170>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f013402ff70>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09a30>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09fb0>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09a70>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d097b0>]

We can also list materials used in the model like so:

.. code-block:: python

    >>> helmet.materials
    [<open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09ab0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09db0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d092f0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09730>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09770>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09c70>]

Your display of these lengthy properties will vary depending on your terminal and screen resolution. Therefore, it is more practical to find out how many different materials or meshes a model has:

.. code-block:: python

    >>> len(helmet.materials)
    6
    >>> len(helmet.meshes)
    6

We can reference each individual mesh by its array index:

.. code-block:: python

    >>> helmet.meshes[0]
    <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0134034170>
    
Which material is it using?

.. code-block:: python

    >>> helmet.meshes[0].material_idx
    0

And what is its mesh name?

.. code-block:: python

    >>> helmet.meshes[0].mesh_name
    'Hose_low'


We can write a loop which displays all mesh names used in a complex model like so:


.. code-block:: python

   >>> for m in helmet.meshes:
   ...     print(m.mesh_name)
   ... 
   Hose_low
   RubberWood_low
   GlassPlastic_low
   MetalParts_low
   LeatherParts_low
   Lenses_low
























Running ``draw.py``
:::::::::::::::::::


``draw()`` Input Parameters
----------------------------

.. code-block:: python

    def draw(geometry=None,
             title="Open3D",
             width=1024,
             height=768,
             actions=None,
             lookat=None,
             eye=None,
             up=None,
             field_of_view=60.0,
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
             
Parameter Description
::::::::::::::::::::::::::::
             
             
	``geometry`` Rene - description

	``title`` Rene - description

	``width`` Rene - description

	``height``  Rene - description

	``actions`` Rene - description

	``lookat`` Rene - description

	``eye`` Rene - description

	``up`` Rene - description

	``field_of_view`` Rene - description

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

