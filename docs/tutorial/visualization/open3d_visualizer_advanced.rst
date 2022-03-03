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

.. image:: https://user-images.githubusercontent.com/93158890/150007965-4959165f-688d-43c0-a839-c1b8efea7073.jpg
    :width: 600px

The above image shows a zoomed-in fragment of our model where we can clearly see some space between the wireframe and the object. Experiment with scale values further to see different visual results.




More Complex Models
:::::::::::::::::::

In the previous section (**Rendering Models**) we have covered how to render complete 3D models with the ``open3d.io.read_triangle_model()`` method. This method can also handle more complex models containing a collection of materials and parts (sub-models) from which the complete object gets assembled.

For this example, we will need to download / ``clone`` *glTF-Sample-Models*  from the KhronosGroup. `glTF (GL Transmission Format) <https://docs.fileformat.com/3d/gltf/>`_ is a 3D file format that stores 3D model information in JSON format. 

First, **minimize your current Python terminal session and open a new one. In a new terminal session:**

.. image:: https://user-images.githubusercontent.com/93158890/150047410-de591582-67c5-42bd-b644-764c36b8c4b8.jpg
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


We can write a loop which displays all mesh names and material indices used in a complex model like so:


.. code-block:: python

   >>> for m in helmet.meshes:
   ...     print(m.mesh_name)
   ...     print(m.material_idx)
   ... 
   Hose_low
   0
   RubberWood_low
   1
   GlassPlastic_low
   2
   MetalParts_low
   3
   LeatherParts_low
   4
   Lenses_low
   5

   

We can also render meshes individually like:

.. code-block:: python

    >>> vis.draw(helmet.meshes[0].mesh)
    
.. image:: https://user-images.githubusercontent.com/93158890/149238095-5385d761-3bae-4172-ab45-1d47b6084d5c.jpg
    :width: 600px


Rendering Sub-Models
::::::::::::::::::::


Just like in the previous loop example which displays all ``mesh_name`` properties, we can write a loop which renders each mesh separately:

.. code-block:: python

    >>> for m in helmet.meshes:
    ...     vis.draw(m.mesh)
    
A series of Open3D visualizer windows should appear. As you close each of them, a new one will appear with a different mesh:

1) A hose:

.. image:: https://user-images.githubusercontent.com/93158890/149238208-961a0a8d-ebb2-4621-aff1-8bfcdeced734.jpg
    :width: 600px
    
2) All wooden and rubber parts:

.. image:: https://user-images.githubusercontent.com/93158890/149238298-98a894cd-72a2-4c76-8e30-da89e26f2fa4.jpg
    :width: 600px

3) The goggles and earphones parts:

.. image:: https://user-images.githubusercontent.com/93158890/149238367-e32d7d12-5472-4f83-90ff-e365c77ef30a.jpg
    :width: 600px
    
4) All metallic parts:

.. image:: https://user-images.githubusercontent.com/93158890/149238437-b225282b-afae-40a2-a485-7f13e0f3122d.jpg
    :width: 600px

5) Leather parts:

.. image:: https://user-images.githubusercontent.com/93158890/149238516-3f6a95f4-6c48-43b6-82e2-8363d0c30197.jpg
    :width: 600px

6) Lenses - they are transparent and thus, are different material as well:

.. image:: https://user-images.githubusercontent.com/93158890/149238634-7919b93d-1307-4ce4-9eb0-646237eceb6e.jpg
    :width: 600px


Cool, isn't it? Now, we can modify the same loop to display all materials and associated properties:

.. code-block:: python

    >>> for m in helmet.meshes:
    ...     vis.draw({'name' : m.mesh_name, 'geometry' : m.mesh, 'material' : helmet.materials[m.material_idx]})

This will give us a full display of each part:

1) A hose:

.. image:: https://user-images.githubusercontent.com/93158890/149238906-065fad20-ed3f-4585-b90b-7d30b5c06912.jpg
    :width: 600px
    
2) All wooden and rubber parts (breathing mask):

.. image:: https://user-images.githubusercontent.com/93158890/149239024-e361bb4a-5fe5-44e7-b41d-8b6d777a1b9b.jpg
    :width: 600px

3) The goggles and earphones parts:

.. image:: https://user-images.githubusercontent.com/93158890/149239132-cea7ad0d-3f42-4a69-a45b-9161c6e43deb.jpg
    :width: 600px
    
4) All metallic parts:

.. image:: https://user-images.githubusercontent.com/93158890/149239248-b884fa06-c121-4c06-a8fd-ef06bc992638.jpg
    :width: 600px

5) Leather parts:

.. image:: https://user-images.githubusercontent.com/93158890/149239346-13e07cd5-1d47-49b6-b43c-7840b01348e9.jpg
    :width: 600px

6) Lenses:

.. image:: https://user-images.githubusercontent.com/93158890/149239403-e6fa3954-8cce-47be-b5b5-b388e7250fe4.jpg
    :width: 600px
















