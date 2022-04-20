.. _visualizer_advanced:

Advanced Open3D visualizer
==========================

Introduction
---------------

.. epigraph:: Open3D provides a convenient function for visualizing geometric objects: ``draw()``. The ``draw()`` function allows you to visualize multiple geometry objects *(PointClouds, LineSets, TriangleMesh)* and images together along with optional, high-quality, physically based (PBR) materials. This tutorial covers more advanced usages of  ``draw()`` calls. For basic ``draw()`` usage, please see the Basic :doc:`visualizer_basic` tutorial.


Getting started
---------------

.. tip::
    This **Getting started** section applies to all subsequent examples below
	 
1. Open your preferred Python environment;
2. In Python, set up common imports that we will use throughout this tutorial:

.. code-block:: python

    >>> import open3d as o3d
    >>> import open3d.visualization as vis
    >>> import numpy as np


Advanced examples
-----------------

In the Basic :doc:`visualizer_basic` tutorial, we covered how to render ``Tensor`` and ``TriangleMesh`` shapes, raster models, and how to control their display programmatically via code and interactively by using Open3D Visualizer UI. This section expounds on those topics to cover more advanced visualization techniques.


.. _rendering_models:

Rendering models
::::::::::::::::

Rendering ``TriangleMesh``'es of 3D models
""""""""""""""""""""""""""""""""""""""""""

In the Basic :doc:`visualizer_basic` tutorial, we showed how to use Open3D datasets. In this tutorial, we will likewise be using Open3D datasets to load 3D models.

.. seealso::

    For more information on datasets, please refer to the 
    :doc:`Open3D Datasets page <../data/index>`

We also demonstrated how to apply materials manually to built-in Open3D geometries. It is also possible to load ``TriangleMesh``'es from full 3D models using the ``o3d.io.read_triangle_mesh()`` method, as you will see below: 


.. code-block:: python

    # Initialize the monkey dataset with downloaded and extracted 3D model
    >>> monkey_model = o3d.data.MonkeyModel()
    # Extract Triangle Mesh data from the preloaded monkey dataset
    >>> monkey = o3d.io.read_triangle_mesh(monkey_model.path)
    >>> vis.draw(monkey)


That will automatically apply the default material which exists in a 3D model:

.. image:: https://user-images.githubusercontent.com/93158890/160008560-4834c962-efa7-4d69-b99d-9ff321a03c02.jpg
    :width: 700px


Next, we will learn how to render full 3D models in all their glory.



Rendering full 3D models
""""""""""""""""""""""""

In the Basic :doc:`visualizer_basic` tutorial, we rendered ``TriangleMesh`` and ``Tensor``-based ``TriangleMesh`` objects. But the ``draw()`` function can also render full-fledged 3D models containing a set of textures and material properties. To read a complete model, we need to use the ``open3d.io.read_triangle_model()`` method, which imports all the material properties in addition to the ``TriangleMesh``:

.. code-block:: python

    >>> monkey_model = o3d.io.read_triangle_model(monkey.path)
    >>> vis.draw(monkey_model)

Clearly, a staggering difference in rendering:

.. image:: https://user-images.githubusercontent.com/93158890/148611141-d424fc74-be7e-4833-913c-714fc3c4fbd2.jpg
    :width: 700px



Rendering more complex models
:::::::::::::::::::::::::::::

In the previous section (:ref:`rendering_models`) we have covered how to render complete 3D models with the ``open3d.io.read_triangle_model()`` method. This method can also handle more complex models containing a collection of materials and parts (sub-models) from which the complete object gets assembled.

For this example, we will be rendering a model of a WWII-era flight helmet from the KhronosGroup *glTF-Sample-Models* . `glTF (GL Transmission Format) <https://docs.fileformat.com/3d/gltf/>`_ is a 3D file format that stores 3D model information in JSON format.


.. tip::

    If you are interested in looking at other *glTF-Sample-Models*, you can go to the KhronosGroup GitHub repository and clone it from this URL:
    
    https://github.com/KhronosGroup/glTF-Sample-Models 



Now, let's run the code which loads and renders the full 3D model of a flight helmet:

.. code-block:: python

    >>> helmet_model = o3d.data.FlightHelmetModel()
    >>> helmet = o3d.io.read_triangle_model(helmet_model.path)
    >>> vis.draw(helmet)
    

.. image:: https://user-images.githubusercontent.com/93158890/148611761-40f95b2b-d257-4f2b-a8c0-60a73b159b96.jpg
    :width: 700px

We've just rendered a complex model - this one actually consists of multiple sub-models with multiple types of materials and textures in it, that can each be rendered separately as we will see shortly.

This and other complex models can also be rendered using the ``o3d.io.read_triangle_mesh()`` method. However, as we will see below, this  yields inferior results:

.. code-block:: python

    >>> helmet = o3d.io.read_triangle_mesh(helmet_model.path)
    >>> vis.draw(helmet)


.. image:: https://user-images.githubusercontent.com/93158890/148611814-09c6fe17-d209-439d-8ae9-c186387fd698.jpg
    :width: 700px

.. note::
   For complex model rendering, please use the ``o3d.io.read_triangle_model()``, rather than ``read_triangle_mesh()``. ``read_triangle_mesh()`` is only good for loading basic meshes, but not complex materials.


Examining complex models
::::::::::::::::::::::::

Let's re-load our ``FlightHelmetModel`` with ``o3d.io.read_triangle_model()``:

.. code-block:: python

    >>> helmet = o3d.io.read_triangle_model(helmet_model.path)

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
    :width: 700px


Rendering sub-models
::::::::::::::::::::


Just like in the previous loop example which displays all ``mesh_name`` and ``material_idx`` properties, we can write a loop which renders each mesh separately:

.. code-block:: python

    >>> for m in helmet.meshes:
    ...     vis.draw(m.mesh)
    
A series of Open3D visualizer windows should appear. As you close each of them, a new one will appear with a different mesh:

1) A hose:

.. image:: https://user-images.githubusercontent.com/93158890/149238208-961a0a8d-ebb2-4621-aff1-8bfcdeced734.jpg
    :width: 700px
    
2) All wooden and rubber parts:

.. image:: https://user-images.githubusercontent.com/93158890/149238298-98a894cd-72a2-4c76-8e30-da89e26f2fa4.jpg
    :width: 700px

Other parts will follow:

3) The goggles and earphones parts
4) All metallic parts
5) Leather parts
6) Lenses

Cool, isn't it? Now, we can modify the same loop to display all materials and associated properties:

.. code-block:: python

    >>> for m in helmet.meshes:
    ...     vis.draw({'name' : m.mesh_name, 'geometry' : m.mesh, 'material' : helmet.materials[m.material_idx]})

This will give us a full display of each part:

1) A hose:

.. image:: https://user-images.githubusercontent.com/93158890/149238906-065fad20-ed3f-4585-b90b-7d30b5c06912.jpg
    :width: 700px
    
2) All wooden and rubber parts (breathing mask):

.. image:: https://user-images.githubusercontent.com/93158890/149239024-e361bb4a-5fe5-44e7-b41d-8b6d777a1b9b.jpg
    :width: 700px

And other parts, just like in the previous ``helmet.meshes`` loop:

3) The goggles and earphones parts    
4) All metallic parts
5) Leather parts:
6) Lenses



Rendering a ``Tensor``-based ``TriangleMesh`` monkey
::::::::::::::::::::::::::::::::::::::::::::::::::::

In the beginning of this tutorial (:ref:`rendering_models`), we rendered a ``TriangleMesh`` of a monkey model using the ``o3d.io.read_triangle_mesh()`` method. Now, we will modify our earlier exercise to convert regular ``TriangleMesh`` into ``Tensor``.

Once again, in your terminal, enter:

.. code-block:: python

    >>> monkey = o3d.io.read_triangle_mesh(monkey_model.path)

Here we are invoking the ``open3d.io`` library which allows us to read 3D model files and/or selectively extract their details. In this case, we are using the ``read_triangle_mesh()`` method for extracting the ``monkey.obj`` file ``TriangleMesh`` data. Now we convert it into **Open3D Tensor geometry**:

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
    :width: 700px

Now, let's work on materials:

.. code-block:: python

    >>> mat = vis.rendering.MaterialRecord()
    >>> mat.base_color = np.asarray([1.0, 1.0, 0.0, 1.0])
    >>> vis.draw({'name': 'monkey', 'geometry': monkey, 'material': mat})
    
We have initialized ``mat.base_color`` to be yellow and get:

.. image:: https://user-images.githubusercontent.com/93158890/148610882-14e6d348-1e8e-4bd9-b0ef-90fa884d9706.jpg
    :width: 700px

Obviously, this looks ugly because the material (``mat``) lacks shading. To correct our 3D rendering, we use ``mat.shader`` property:

.. code-block:: python

    >>> mat.shader = 'defaultLit'
    >>> vis.draw({'name': 'monkey', 'geometry': monkey, 'material': mat})

This time, we see a big difference because the ``mat.shader`` property is initialized:

.. image:: https://user-images.githubusercontent.com/93158890/148611064-2fa5fe4c-b8cb-4588-ad46-df23cdf160be.jpg
    :width: 700px

You can experiment with different material colors to your liking by changing numeric values in the ``mat.base_color = np.asarray([1.0, 1.0, 0.0, 1.0])`` statement.

