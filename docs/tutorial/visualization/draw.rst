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
    mat.base_color = [1.0, 0.0, 1.0, 1.0]
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, "material": mat})

Let's examine new elements in the code above:

- ``MaterialRecord()`` is a structure which holds various material properties.
- The ``mat.shader`` property accepts a string representing the material type.
  The two most common options are ``"defaultLit"`` and ``"defaultUnlit"``. Other
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
    mat.base_color = [0.8, 0.9, 1.0, 1.0]
    mat.base_roughness = 0.4
    mat.base_metallic = 1.0
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, "material": mat}, ibl="nightlights")

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
  lighting) property. The ``ibl`` parameter property allows the user to specify
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
    mat.base_color = [0.8, 0.9, 1.0, 1.0]
    mat.base_roughness = 0.25
    mat.base_reflectance = 0.9
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, "material": mat}, ibl="nightlights")

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
    o3d.visualization.draw({"name": "sphere", "geometry": sphere, "material": mat}, ibl="nightlights")

Let's examine new elements in the code above:

- ``create_sphere(2.0, 100, create_uv_map=True)``: Generates texture UV map coordinates.
- ``mat.albedo_img``: Sets the base color texture image.
- ``mat.normal_img``: Sets the normal texture image.
- ``mat.roughness_img``: Sets the roughness texture image.

.. image:: https://user-images.githubusercontent.com/93158890/157775220-443aad2d-9123-42d0-b584-31e9fb8f38c3.jpg
    :width: 700px

TriangleMeshModel
-----------------

Introducing TriangleMeshModel
:::::::::::::::::::::::::::::

In Open3D, ``TriangleMeshModel`` is a class containing one or more triangles
meshes and materials. When loading a ``.gltf`` model, we typically want to use
``TriangleMeshModel`` instead of ``TriangleMesh``.

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

Examining complex models
::::::::::::::::::::::::

In the following example, we will see that ``TriangleMeshModel`` can contain
multiple triangle meshes and materials.

.. code-block:: python

    # For more models like this, checkout
    # - http://www.open3d.org/docs/latest/tutorial/data/index.html
    # - https://github.com/KhronosGroup/glTF-Sample-Models
    helmet_data = o3d.data.FlightHelmetModel()
    helmet_model = o3d.io.read_triangle_mesh_model(helmet_data.path)
    o3d.visualization.draw(helmet_model)

.. image:: https://user-images.githubusercontent.com/93158890/148611761-40f95b2b-d257-4f2b-a8c0-60a73b159b96.jpg
    :width: 700px

Now, let's examine the ``helmet_model`` that we just read.

.. code-block:: python

    # helmet_model.meshes is a list of MeshInfo
    >>> print(helmet_model.meshes)
    [<open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0134034170>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f013402ff70>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09a30>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09fb0>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d09a70>,
     <open3d.cpu.pybind.visualization.rendering.TriangleMeshModel.MeshInfo object at 0x7f0132d097b0>]

    # helmet_model.materials is a list of MaterialRecord
    >>> helmet_model.materials
    [<open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09ab0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09db0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d092f0>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09730>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09770>,
     <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f0132d09c70>]

    # Each MeshInfo contains a mesh, a material index, and a name.
    >>> print(helmet_model.meshes[0].mesh)
    TriangleMesh with 10472 points and 19680 triangles.

    >>> print(helmet_model.meshes[0].material_idx)
    0

    >>> print(helmet_model.meshes[0].mesh_name)
    Hose_low

Let's render a sub-mesh from the ``helmet_model``.

.. code-block:: python

    sub_mesh = helmet_model.meshes[0]
    o3d.visualization.draw({"name": sub_mesh.mesh_name,
                            "geometry": sub_mesh.mesh,
                            "material": helmet_model.materials[sub_mesh.material_idx]})

.. image:: https://user-images.githubusercontent.com/93158890/149238906-065fad20-ed3f-4585-b90b-7d30b5c06912.jpg
    :width: 700px

If you use ``read_triangle_mesh()`` instead of ``read_triangle_mesh_model()``,
all sub-meshes will be merged into one ``TriangleMesh``, and the materials
might be ignored.

.. code-block:: python

    helmet_mesh = o3d.io.read_triangle_mesh(helmet_data.path)
    o3d.visualization.draw(helmet_mesh)

.. image:: https://user-images.githubusercontent.com/93158890/148611814-09c6fe17-d209-439d-8ae9-c186387fd698.jpg
    :width: 700px

Commonly used options
---------------------

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

@Alex, if we already set ``show_skybox=False``, do we still need to specify
``show_skybox=False``?

.. code-block:: python

    o3d.visualization.draw(sphere,
                           show_ui=True,
                           title="Green Background",
                           show_skybox=False,
                           bg_color=(0.56, 1.0, 0.69, 1.0))

.. image:: https://user-images.githubusercontent.com/93158890/160878317-a57755a0-8b8f-44db-b718-443aa435035a.jpg
    :width: 700px

@Alex, add ``raw_mode`` example.
