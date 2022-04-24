.. _draw:

Visualization with ``draw()``
=============================

Open3D provides a convenient ``draw()`` function (``open3d.visualization.draw()``) for visualizing geometries. This tutorial will demonstrate its basic usage.

In this tutorial, we assume that you have already installed Python and Open3D. If you need help setting up Open3D on your system, please follow instructions on the :doc:`Open3D Setup page <../../getting_started>` .

To get started, run the following import statements in Python:

.. code-block:: python

    import open3d as o3d
    import numpy as np    # needed for conversions of lists to arrays
    import math           # needed to reference Pi and rotate objects

Basic drawing
-------------

Point cloud
:::::::::::

.. code-block:: python

    # Create a PointCloud dataset and a PointCloud object
    dataset = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(dataset.path)

    # Rotate the PointCloud object
    rotate_180 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi, 0, 0))
    pcd.rotate(rotate_180)

    # Render with the draw() function
    o3d.visualization.draw(pcd)

.. image:: https://user-images.githubusercontent.com/93158890/159548100-404afe97-8960-4e68-956f-cc6957632a93.jpg
    :width: 700px

Triangle mesh
::::::::::::::

.. code-block:: python

    cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    cube.compute_vertex_normals()  # See "Vertex and triangle normals" section for details
    o3d.visualization.draw(cube)

For more information on ``compute_vertex_normals()`` and ``compute_triangle_normals()`` methods, please see the :ref:`vertex_and_triangle_normals` section.

.. image:: https://user-images.githubusercontent.com/93158890/164767767-4e980277-3125-4cdc-8b44-454459c2ea3c.jpg
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
    
    # Create a bounding box that fully encompasses the sphere
    bbox = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sphere.vertices)
    
    # Create a line set from the bounding box
    line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(bbox)
    
    # Paint the bounding box blue
    line_set.paint_uniform_color([0, 0, 1])
    
    # Draw mutliple objects at once
    o3d.visualization.draw([sphere, line_set])

.. image:: https://user-images.githubusercontent.com/93158890/157901535-fbe78fc0-9b85-476e-a0a1-01e0e5d80738.jpg
    :width: 700px

.. _vertex_and_triangle_normals:

Vertex and triangle normals
---------------------------

Vertex normals and triangle normals are important for the shading of triangle mesh.

Without normals
:::::::::::::::

First, we draw a sphere without normals:

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    o3d.visualization.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/164772409-05ae2de2-6b61-47f6-8443-8ab0b5bf87df.jpg
    :width: 700px

With triangle normals
:::::::::::::::::::::

Then, we compute the triangle normals of the sphere. The resulting visualization shows a flat-shaded sphere for each face (triangles).

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0)
    sphere.compute_triangle_normals()     # Computes a single normal per triangle
    o3d.visualization.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/157728100-0a495e56-c613-40c4-a292-6e45213d61f6.jpg
    :width: 700px

With vertex normals
:::::::::::::::::::

Finally, we compute the vertex normals of the sphere. The resulting visualization shows a smooth-shaded sphere. Note that internally, ``TriangleMesh::compute_vertex_normals()`` will compute both the vertex and triangle normals, while ``TriangleMesh::compute_triangle_normals()`` will only compute the triangle normals.

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0)
    sphere.compute_vertex_normals()     # Computes a smooth normal at each vertex
    o3d.visualization.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/157339234-1a92a944-ac38-4256-8297-0ad78fd24b9c.jpg
    :width: 700px

Materials
---------

``base_color`` property
:::::::::::::::::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_vertex_normals()
    mat = o3d.visualization.rendering.MaterialRecord()
    mat.shader = "defaultLit"
    
    # base material RGBA color. Should be numpy array
    mat.base_color = [1.0, 0.0, 1.0, 1.0]
    
    # Draw variant which allows the user to specify a material to use with the
    # geometry rather than the previous examples with which the draw call created a
    # default material automatically
    o3d.visualization.draw({"name": "sphere",
                            "geometry": sphere,
                            "material": mat})

Let's examine new elements in the code above:

- ``MaterialRecord()`` is a structure which holds various material properties.
- The ``mat.shader`` property accepts a string representing the material type. The two most common options are ``"defaultLit"`` and ``"defaultUnlit"``. 
- The ``mat.base_color`` represents the base material RGBA color.

.. image:: https://user-images.githubusercontent.com/93158890/150883605-a5e65a3f-0a25-4ff4-b039-4aa6e53a1440.jpg
    :width: 700px

``base_roughness`` and ``base_metallic`` properties
:::::::::::::::::::::::::::::::::::::::::::::::::::

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
    o3d.visualization.draw({"name": "sphere",
                            "geometry": sphere,
                            "material": mat},
                           ibl="nightlights")

Let's examine new elements in the code above:

- ``get_rotation_matrix_from_xyz()``: Creates a rotation matrix given angles to rotate around the ``x``, ``y``, and ``z`` axes.
- ``mat.base_roughness = 0.4``: PBR (physically based rendering) material property which controls the smoothness of the surface (see `Filament Material Guide <https://google.github.io/filament/Materials.html>`_ for details).
- ``mat.base_metallic = 1.0``: PBR material property which defines whether the surface is metallic or not (see `Filament Material Guide <https://google.github.io/filament/Materials.html>`_ for details).
- ``o3d.visualization.draw(..., ibl="nightlights")`` introduces the ``ibl`` (Image-Based Lighting) property. The ``ibl`` parameter property allows the user to specify the built-in HDR lighting to use. ``"nightlights"`` creates a realistic nighttime city scene:

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
    o3d.visualization.draw({"name": "sphere",
                            "geometry": sphere,
                            "material": mat}, 
                           ibl="nightlights")

Let's examine new elements in the code above:

- ``mat.base_reflectance = 0.9``: PBR material property which controls the reflectance (glossiness) of the surface (see  `Filament Material Guide <https://google.github.io/filament/Materials.html>`_ for details)

.. image:: https://user-images.githubusercontent.com/93158890/157770798-2c42e7dc-e063-4f26-90b4-16a45e263f36.jpg
    :width: 700px

Texture map
:::::::::::

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100, create_uv_map=True)
    sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-np.pi / 2, 0, 0))
    sphere.rotate(rotate_90)
    
    # Load a texture dataset
    mat_data = o3d.data.TilesTexture()
    mat.shader = "defaultLit"
    
    # o3d.io.read_image() loads an image and supports jpeg and PNG images
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path)
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path)
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path)
    o3d.visualization.draw({"name": "sphere",
                            "geometry": sphere,
                            "material": mat},
                           ibl="nightlights")

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

In Open3D, ``TriangleMeshModel`` is a class containing one or more triangle meshes and materials. When loading complete 3D models (including those in the ``.gltf`` format), we typically want to use ``TriangleMeshModel`` instead of ``TriangleMesh``.

The following example reads and renders the monkey ``TriangleMesh`` with its default material:

.. attention::

    **@errissa, Rene, please explain:**
    
    **WHY do we still get the default material?**

.. code-block:: python

    monkey_data = o3d.data.MonkeyModel()
    monkey_mesh = o3d.io.read_triangle_mesh(monkey_data.path)
    o3d.visualization.draw(monkey_mesh)

.. image:: https://user-images.githubusercontent.com/93158890/160008560-4834c962-efa7-4d69-b99d-9ff321a03c02.jpg
    :width: 700px

With ``TriangleMeshModel`` and ``read_triangle_mesh_model`` we can read and render the full set of materials:

.. code-block:: python

    monkey_model = o3d.io.read_triangle_mesh_model(monkey_data.path)
    o3d.visualization.draw(monkey_model)

.. image:: https://user-images.githubusercontent.com/93158890/148611141-d424fc74-be7e-4833-913c-714fc3c4fbd2.jpg
    :width: 700px

Examining complex models
::::::::::::::::::::::::

In the following example, we will see that the ``TriangleMeshModel`` can contain multiple triangle meshes and materials:

.. code-block:: python

    # For more models like this, checkout
    # - http://www.open3d.org/docs/latest/tutorial/data/index.html
    # - https://github.com/KhronosGroup/glTF-Sample-Models
    helmet_data = o3d.data.FlightHelmetModel()
    helmet_model = o3d.io.read_triangle_mesh_model(helmet_data.path)
    o3d.visualization.draw(helmet_model)

.. image:: https://user-images.githubusercontent.com/93158890/148611761-40f95b2b-d257-4f2b-a8c0-60a73b159b96.jpg
    :width: 700px

Now, let's examine the ``helmet_model`` we just rendered:

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

Let's render a sub-mesh from the ``helmet_model``:

.. code-block:: python

    sub_mesh = helmet_model.meshes[0]
    o3d.visualization.draw({"name": sub_mesh.mesh_name,
                            "geometry": sub_mesh.mesh,
                            "material": helmet_model.materials[sub_mesh.material_idx]})

.. image:: https://user-images.githubusercontent.com/93158890/149238906-065fad20-ed3f-4585-b90b-7d30b5c06912.jpg
    :width: 700px

If you use ``read_triangle_mesh()`` instead of ``read_triangle_mesh_model()``, all sub-meshes will be merged into one ``TriangleMesh``, and the materials might be ignored.

.. code-block:: python

    helmet_mesh = o3d.io.read_triangle_mesh(helmet_data.path)
    o3d.visualization.draw(helmet_mesh)

.. image:: https://user-images.githubusercontent.com/93158890/148611814-09c6fe17-d209-439d-8ae9-c186387fd698.jpg
    :width: 700px

Commonly used options
---------------------

UI menu, title, and window dimensions
:::::::::::::::::::::::::::::::::::::

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

.. image:: https://user-images.githubusercontent.com/93158890/164792527-121548f0-dacc-4197-a0cf-dec287d2d3bb.jpg
    :width: 700px

Assigning object names
::::::::::::::::::::::

.. code-block:: python

    geoms = [{"name": "Sphere", "geometry": sphere},
             {"name": "Bounding box", "geometry": line_set}]
    o3d.visualization.draw(geoms, show_ui=True)

.. image:: https://user-images.githubusercontent.com/93158890/164792623-795b80c3-332b-46be-9886-2ca9bf2a34f8.jpg
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

.. code-block:: python

    o3d.visualization.draw(sphere,
                           show_ui=True,
                           title="Green Background",
                           show_skybox=False,
                           bg_color=(0.56, 1.0, 0.69, 1.0))

.. image:: https://user-images.githubusercontent.com/93158890/160878317-a57755a0-8b8f-44db-b718-443aa435035a.jpg
    :width: 700px

Same example, but with the ``raw_mode`` activated:

.. code-block:: python

    o3d.visualization.draw(sphere,
                           show_ui=True,
                           title="Green Background - Raw mode",
                           show_skybox=False,
                           bg_color=(0.56, 1.0, 0.69, 1.0),
                           raw_mode=True)

.. image:: https://user-images.githubusercontent.com/93158890/164815973-2f291675-d43a-47a7-a8fe-38af225b9948.jpg
    :width: 700px
                           