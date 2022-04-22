.. _draw:

Visualization with draw()
=========================

Open3D provides a convenient draw function (``open3d.visualization.draw()``) for
visualizing geometries. This tutorial will demonstrate the basic usage of the
draw function.

In this tutorial, we assume that you have already installed Open3D and have
run the following import statements:

.. code-block:: python

    import open3d as o3d
    import numpy as np

Basic drawing
-------------

Point cloud
:::::::::::

.. code-block:: python

    dataset = o3d.data.PCDPointCloud()  # Downloads the demo point cloud dataset
    pcd = o3d.io.read_point_cloud(dataset.path)
    o3d.visualization.draw(pcd)

.. image:: https://user-images.githubusercontent.com/93158890/159548100-404afe97-8960-4e68-956f-cc6957632a93.jpg
    :width: 700px

Triangle mesh
::::::::::::::

.. code-block:: python

    cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    sphere.compute_vertex_normals()  # See "vertex and triangle normals" section for details
    o3d.visualization.draw(cube)

@Alex change the image here. The mesh should be shown with vertex normals.

.. image:: https://user-images.githubusercontent.com/93158890/148607529-ee0ae0de-05af-423d-932c-2a5a6c8d7bda.jpg
    :width: 700px

Line set
::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 25)
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    line_set.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.visualization.draw(line_set)

.. image:: https://user-images.githubusercontent.com/93158890/157949589-8b87fa81-a5cf-4791-a4f7-2d5dc91e546e.jpg
    :width: 700px

Multiple objects
::::::::::::::::

This example draws a triangle mesh together with a line set.

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sphere.vertices)
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    line_set.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw([sphere, line_set])

.. image:: https://user-images.githubusercontent.com/93158890/157901535-fbe78fc0-9b85-476e-a0a1-01e0e5d80738.jpg
    :width: 700px

Vertex and triangle normals
---------------------------

Vertex normals and triangle normals are important for the shading of triangle
mesh.

Without normals
:::::::::::::::

First, we draw a sphere without normals.

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    o3d.visualization.draw(sphere)

@Alex, add an image here.

With triangle normals
:::::::::::::::::::::

Then, we compute the triangle normals of the sphere. The resulting visualization
shows a flat-shaded sphere for each face (triangles).

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_triangle_normals()
    o3d.visualization.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/157728100-0a495e56-c613-40c4-a292-6e45213d61f6.jpg
    :width: 700px

With vertex normals
:::::::::::::::::::

Finally, we compute the vertex normals of the sphere. The resulting
visualization shows a smooth-shaded sphere. Note that internally,
``TriangleMesh::compute_vertex_normals()`` will compute both the vertex and
triangle normals, while ``TriangleMesh::compute_triangle_normals()`` will only
compute the triangle normals.

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    o3d.visualization.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/157339234-1a92a944-ac38-4256-8297-0ad78fd24b9c.jpg
    :width: 700px

Materials
---------

Base color
::::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = np.asarray([1.0, 0.0, 1.0, 1.0])
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, 'material': mat})

Let's examine new elements in the code above:

- ``MaterialRecord()`` is a structure which holds various material properties.
- The ``mat.shader`` property accepts a string representing the material type.
  The two most common options are ``'defaultLit'`` and ``'defaultUnlit'``. Other
  available options will be covered in :doc:`visualizer_advanced` tutorial.
- The ``mat.base_color`` represents the base material RGBA color.

.. image:: https://user-images.githubusercontent.com/93158890/150883605-a5e65a3f-0a25-4ff4-b039-4aa6e53a1440.jpg
    :width: 700px

Metallic and roughness
::::::::::::::::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    sphere.rotate(rotate_90)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = np.asarray([0.8, 0.9, 1.0, 1.0])
    mat.base_roughness = 0.4
    mat.base_metallic = 1.0
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, 'material': mat}, ibl="nightlights")

Let's examine new elements in the code above:

- ``get_rotation_matrix_from_xyz()``: Creates a rotation matrix given angles to
  rotate around the ``x``, ``y``, and ``z`` axes.
- ``mat.base_roughness = 0.4``: PBR (physically based rendering) material
  property which controls the smoothness of the surface (see  `Filament Material
  Guide <https://google.github.io/filament/Materials.html>`_ for details).
- ``mat.base_metallic = 1.0``: PBR material property which defines whether the
  surface is metallic or not (see  `Filament Material Guide
  <https://google.github.io/filament/Materials.html>`_ for details).
- ``o3d.visualization.draw(..., ibl="nightlights")``: The ``ibl`` (image based
  lighting) property. The *'ibl'* parameter property allows the user to specify
  the built-in HDR lighting to use. ``"nightlights"`` is from a nighttime city
  scene.

.. image:: https://user-images.githubusercontent.com/93158890/157758092-9efb1ca0-b96a-4e1d-abd7-95243b279d2e.jpg
    :width: 700px

Reflectance
:::::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    sphere.rotate(rotate_90)
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    mat.base_color = np.asarray([0.8, 0.9, 1.0, 1.0])
    mat.base_roughness = 0.25
    mat.base_reflectance = 0.9
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, 'material':   mat}, ibl="nightlights")

Let's examine new elements in the code above:

- ``mat.base_reflectance = 0.9``: PBR material property which controls the
  reflectance (glossiness) of the surface (see  `Filament Material Guide
  <https://google.github.io/filament/Materials.html>`_ for details)

.. image:: https://user-images.githubusercontent.com/93158890/157770798-2c42e7dc-e063-4f26-90b4-16a45e263f36.jpg
    :width: 700px


Texture map
:::::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100, create_uv_map=True)
    sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    sphere.rotate(rotate_90)

    mat_data = o3d.data.TilesTexture()
    mat.shader = "defaultLit"
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, 'material': mat}, ibl="nightlights")

Let's examine new elements in the code above:

- ``create_sphere(2.0, 100, create_uv_map=True)``: Generates texture UV map coordinates.
- ``mat.albedo_img``: Sets the base color texture image.
- ``mat.normal_img``: Sets the normal texture image.
- ``mat.roughness_img``: Sets the roughness texture image.

.. image:: https://user-images.githubusercontent.com/93158890/157775220-443aad2d-9123-42d0-b584-31e9fb8f38c3.jpg
    :width: 700px


Common options
--------------

UI menu, title, and window dimension
::::::::::::::::::::::::::::::::::::

@Alex, update the screen capture, now the title has been changed to "Sphere and bounding box".

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sphere.vertices)
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    line_set.paint_uniform_color([0, 0, 1])

    o3d.visualization.draw([sphere, line_set],
                            show_ui=True,
                            title="Sphere and bounding box",
                            width=700,
                            height=700)

.. image:: https://user-images.githubusercontent.com/93158890/158281728-994ff828-53b0-485a-9feb-9b121d7354f7.jpg
    :width: 700px

Assigning object names
::::::::::::::::::::::

@Alex, rename the sphere to "Sphere".
@Alex, rename the line set to "Bounding box".
@Alex, create a new rendering.

.. code-block:: python

    geoms = [{"name": "sphere", "geometry": sphere},
             {"name": "Axis Aligned Bounding Box line_set", "geometry": line_set}]
    o3d.visualization.draw(geoms, show_ui=True)

.. image:: https://user-images.githubusercontent.com/93158890/159094500-83ddd46f-0e71-40e1-9b97-ae46480cd860.jpg
    :width: 700px

Show/hide the skybox
::::::::::::::::::::

.. code-block:: python

    o3d.visualization.draw(sphere, show_ui=True, show_skybox=False)

And the Visualizer window opens without the default skybox blue background:

.. image:: https://user-images.githubusercontent.com/93158890/159093215-31dcacf7-306f-4231-9155-0df474ce4828.jpg
    :width: 700px

Set background color
::::::::::::::::::::

@Alex, can we skip ``show_skybox=False``?

.. code-block:: python

    o3d.visualization.draw(sphere,
                           show_ui=True,
                           title="Green Background",
                           show_skybox=False,
                           bg_color=(0.56, 1.0, 0.69, 1.0))

.. image:: https://user-images.githubusercontent.com/93158890/160878317-a57755a0-8b8f-44db-b718-443aa435035a.jpg
    :width: 700px

@Alex, add ``raw_mode`` example.


Drawing TriangleMeshModel
-------------------------

TriangleMesh vs. TriangleMeshModel
::::::::::::::::::::::::::::::::::

In Open3D ``TriangleMeshModel`` is a class containing ``TriangleMesh``es and
materials.

The following example reads and render the monkey ``TriangleMesh`` with its
default material.

@Alex, explain why do we still get the default material?

.. code-block:: python

    monkey_data = o3d.data.MonkeyModel()
    monkey_mesh = o3d.io.read_triangle_mesh(monkey_data.path)
    o3d.visualization.draw(monkey_mesh)

.. image:: https://user-images.githubusercontent.com/93158890/160008560-4834c962-efa7-4d69-b99d-9ff321a03c02.jpg
    :width: 700px

With ``TriangleMeshModel`` and ``read_triangle_mesh_model`` we can read and
render the full set of materials.

.. code-block:: python

    monkey_model = o3d.io.read_triangle_mesh_model(monkey_data.path)
    o3d.visualization.draw(monkey_model)

.. image:: https://user-images.githubusercontent.com/93158890/148611141-d424fc74-be7e-4833-913c-714fc3c4fbd2.jpg
    :width: 700px

Now, let's run the code which loads and renders the full 3D model of a flight
helmet:

.. code-block:: python

    # For more models like this, checkout
    # - http://www.open3d.org/docs/latest/tutorial/data/index.html
    # - https://github.com/KhronosGroup/glTF-Sample-Models
    helmet_data = o3d.data.FlightHelmetModel()
    helmet_model = o3d.io.read_triangle_mesh_model(helmet_data.path)
    o3d.visualization.draw(helmet_model)

.. image:: https://user-images.githubusercontent.com/93158890/148611761-40f95b2b-d257-4f2b-a8c0-60a73b159b96.jpg
    :width: 700px

We've just rendered a complex model - this one actually consists of multiple
sub-models with multiple types of materials and textures in it, that can each be
rendered separately as we will see shortly.

This and other complex models can also be rendered using the
``o3d.io.read_triangle_mesh()`` method. However, as we will see below, this
yields inferior results:

.. code-block:: python

    helmet_mesh = o3d.io.read_triangle_mesh(helmet_data.path)
    o3d.visualization.draw(helmet_mesh)

.. image:: https://user-images.githubusercontent.com/93158890/148611814-09c6fe17-d209-439d-8ae9-c186387fd698.jpg
    :width: 700px

.. note::
   For complex model rendering, please use the ``o3d.io.read_triangle_mesh_model()``, rather than ``read_triangle_mesh()``. ``read_triangle_mesh()`` is only good for loading basic meshes, but not complex materials.

Examining complex models
::::::::::::::::::::::::

Let's re-load our ``FlightHelmetModel`` with ``o3d.io.read_triangle_mesh_model()``:

.. code-block:: python

    helmet_model = o3d.io.read_triangle_mesh_model(helmet_data.path)

Take a look at what the ``helmet`` object consists of. First, we find out its
type:

.. code-block:: python

    helmet_model
    <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel object at 0x7f019efa7770>

Now, we'll look at its meshes:

.. code-block:: python

    helmet_model.meshes
    [<open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0134034170>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f013402ff70>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09a30>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09fb0>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09a70>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d097b0>]

We can also list materials used in the model like so:

.. code-block:: python

    helmet_model.materials
    [<open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09ab0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09db0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d092f0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09730>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09770>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09c70>]

Your display of these lengthy properties will vary depending on your terminal
and screen resolution. Therefore, it is more practical to find out how many
different materials or meshes a model has:

.. code-block:: python

    len(helmet_model.materials)
    6
    len(helmet_model.meshes)
    6

We can reference each individual mesh by its array index:

.. code-block:: python

    helmet_model.meshes[0]
    <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0134034170>

Which material is it using?

.. code-block:: python

    helmet_model.meshes[0].material_idx
    0

And what is its mesh name?

.. code-block:: python

    helmet_model.meshes[0].mesh_name
    'Hose_low'

We can write a loop which displays all mesh names and material indices used in a
complex model like so:

.. code-block:: python

    for m in helmet_model.meshes:
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

    o3d.visualization.draw(helmet_model.meshes[0].mesh)

.. image:: https://user-images.githubusercontent.com/93158890/149238095-5385d761-3bae-4172-ab45-1d47b6084d5c.jpg
    :width: 700px

Rendering sub-models
::::::::::::::::::::

Just like in the previous loop example which displays all ``mesh_name`` and
``material_idx`` properties, we can write a loop which renders each mesh
separately:

.. code-block:: python

    for m in helmet_model.meshes:
    ...     o3d.visualization.draw(m.mesh)

A series of Open3D visualizer windows should appear. As you close each of them,
a new one will appear with a different mesh:

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

Cool, isn't it? Now, we can modify the same loop to display all materials and
associated properties:

.. code-block:: python

    for m in helmet_model.meshes:
    ...     o3d.visualization.draw({'name' : m.mesh_name, 'geometry' : m.mesh, 'material' : helmet_model.materials[m.material_idx]})

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

In the beginning of this tutorial (:ref:`rendering_models`), we rendered a
``TriangleMesh`` of a monkey model using the ``o3d.io.read_triangle_mesh()``
method. Now, we will modify our earlier exercise to convert regular
``TriangleMesh`` into ``Tensor``.

Once again, in your terminal, enter:

.. code-block:: python

    monkey_mesh = o3d.io.read_triangle_mesh(monkey_data.path)

Here we are invoking the ``open3d.io`` library which allows us to read 3D model
files and/or selectively extract their details. In this case, we are using the
``read_triangle_mesh()`` method for extracting the ``monkey.obj`` file
``TriangleMesh`` data. Now we convert it into **Open3D Tensor geometry**:

.. code-block:: python

    monkey_tensor = o3d.t.geometry.TriangleMesh.from_legacy(monkey_mesh)

Let's see what properties ``monkey_tensor`` has:

.. code-block:: python

    monkey_tensor
    TriangleMesh on CPU:0 [9908 vertices (Float32) and 15744 triangles (Int64)].
    Vertex Attributes: normals (dtype = Float32, shape = {9908, 3}).
    Triangle Attributes: texture_uvs (dtype = Float32, shape = {15744, 3, 2}).

Time to render the ``monkey_tensor``:

.. code-block:: python

    o3d.visualization.draw(monkey_tensor)

And we get:

.. image:: https://user-images.githubusercontent.com/93158890/148610827-4a8dc85f-5664-4f7a-b0da-1808387c9f71.jpg
    :width: 700px

Now, let's work on materials:

.. code-block:: python

    mat = o3d.visualization.rendering.MaterialRecord()
    mat.base_color = np.asarray([1.0, 1.0, 0.0, 1.0])
    o3d.visualization.draw({'name': 'monkey', 'geometry': monkey_tensor, 'material': mat})

We have initialized ``mat.base_color`` to be yellow and get:

.. image:: https://user-images.githubusercontent.com/93158890/148610882-14e6d348-1e8e-4bd9-b0ef-90fa884d9706.jpg
    :width: 700px

Obviously, this looks ugly because the material (``mat``) lacks shading. To
correct our 3D rendering, we use ``mat.shader`` property:

.. code-block:: python

    mat.shader = 'defaultLit'
    o3d.visualization.draw({'name': 'monkey', 'geometry': monkey_tensor, 'material': mat})

This time, we see a big difference because the ``mat.shader`` property is
initialized:

.. image:: https://user-images.githubusercontent.com/93158890/148611064-2fa5fe4c-b8cb-4588-ad46-df23cdf160be.jpg
    :width: 700px

You can experiment with different material colors to your liking by changing
numeric values in the ``mat.base_color = np.asarray([1.0, 1.0, 0.0, 1.0])``
statement.

