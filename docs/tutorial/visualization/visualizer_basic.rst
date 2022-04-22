.. _visualizer_basic:

Open3D visualizer
=================

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

This example draws a point cloud.

.. code-block:: python

    dataset = o3d.data.PCDPointCloud()  # Downloads the demo point cloud dataset
    pcd = o3d.io.read_point_cloud(dataset.path)
    o3d.visualization.draw(pcd)

.. image:: https://user-images.githubusercontent.com/93158890/159548100-404afe97-8960-4e68-956f-cc6957632a93.jpg
    :width: 700px

Triangle mesh
::::::::::::::

This example draws a triangle mesh.

.. code-block:: python

    cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    sphere.compute_vertex_normals()  # See "vertex and triangle normals" section for details
    o3d.visualization.draw(cube)

@Alex change the image here. The mesh should be shown with vertex normals.

.. image:: https://user-images.githubusercontent.com/93158890/148607529-ee0ae0de-05af-423d-932c-2a5a6c8d7bda.jpg
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

First, we draw a sphere without normals.

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    o3d.visualization.draw(sphere)

@Alex, add an image here.

Then, we compute the triangle normals of the sphere. The resulting visualization
shows a flat-shaded sphere for each face (triangles).

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    sphere.compute_triangle_normals()
    o3d.visualization.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/157728100-0a495e56-c613-40c4-a292-6e45213d61f6.jpg
    :width: 700px

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

.. image:: https://user-images.githubusercontent.com/93158890/150883605-a5e65a3f-0a25-4ff4-b039-4aa6e53a1440.jpg
    :width: 700px

Let's examine new elements in the code above:

- ``MaterialRecord()`` is a structure which holds various material properties.
- The ``mat.shader`` property accepts a string representing the material type.
  The two most common options are ``'defaultLit'`` and ``'defaultUnlit'``. Other
  available options will be covered in :doc:`visualizer_advanced` tutorial.
- The ``mat.base_color`` represents the base material RGBA color.

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

.. image:: https://user-images.githubusercontent.com/93158890/157758092-9efb1ca0-b96a-4e1d-abd7-95243b279d2e.jpg
    :width: 700px

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

.. image:: https://user-images.githubusercontent.com/93158890/157770798-2c42e7dc-e063-4f26-90b4-16a45e263f36.jpg
    :width: 700px

Let's examine new elements in the code above:

- ``mat.base_reflectance = 0.9``: PBR material property which controls the
  reflectance (glossiness) of the surface (see  `Filament Material Guide
  <https://google.github.io/filament/Materials.html>`_ for details)

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

.. image:: https://user-images.githubusercontent.com/93158890/157775220-443aad2d-9123-42d0-b584-31e9fb8f38c3.jpg
    :width: 700px

Let's examine new elements in the code above:

- ``create_sphere(2.0, 100, create_uv_map=True)``: Generates texture UV map coordinates.
- ``mat.albedo_img``: Sets the base color texture image.
- ``mat.normal_img``: Sets the normal texture image.
- ``mat.roughness_img``: Sets the roughness texture image.

.. _trianglemesh_lineset:

Drawing a wireframe sphere
--------------------------

Line Sets are typically used to display a wireframe of a 3D model. Let's do that
by creating a custom ``LineSet`` object:

.. code-block:: python

    sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 25)
    sphere.compute_vertex_normals()
    rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.  pi / 2, 0, 0))
    sphere.rotate(rotate_90)
    line_set = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)
    line_set.paint_uniform_color([0.0, 0.0, 1.0])
    o3d.visualization.draw(line_set)

.. image:: https://user-images.githubusercontent.com/93158890/157949589-8b87fa81-a5cf-4791-a4f7-2d5dc91e546e.jpg
    :width: 700px

So, what's new in this code?

``line_set = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)`` - here we
create a line set from the edges of individual triangles of a triangle mesh.

``line_set.paint_uniform_color([0.0, 0.0, 1.0])`` - here we paint the wireframe
``LineSet`` blue. [*Red=0, Green=0, Blue=1*]

``LineSet`` objects
:::::::::::::::::::

As recently shown in the ``TriangleMesh LineSet`` Sphere example
(:ref:`trianglemesh_lineset`), Line Sets are used to render a wireframe of a 3D
model. In our case, we are creating a basic cubic frame around our sphere based
on the ``AxisAlignedBoundingBox`` object (``aabb``) we created earlier:

``line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)``

``line_set.paint_uniform_color([0, 0, 1])`` - paints the bounding box
``LineSet`` blue.

Multiple object parameters in ``draw()`` calls
::::::::::::::::::::::::::::::::::::::::::::::

Finally, we have a ``draw()`` call with multiple 3D object parameters:

``o3d.visualization.draw([sphere,line_set])``

You can pass as many objects to the ``draw()`` as you need.

Specifying wireframe ``line_width``
:::::::::::::::::::::::::::::::::::

Aside from rendering ``LineSet`` wireframes or grids, we can change their
thickness by passing in a ``line_width`` parameter with a numeric value to the
``draw()`` function like so:

.. code-block:: python

    o3d.visualization.draw([sphere,line_set], line_width=50)

Here we rendered a grotesquely thicker Bounding Box by increasing its thickness
(``line_width`` property) to ``50``:

.. image:: https://user-images.githubusercontent.com/93158890/158695002-f5976bfa-1e81-46dc-bf3b-b926d0c5e0af.jpg
    :width: 700px

The default value for the ``line_width`` parameter is ``2``. The minimum
supplied value is ``1``. The rendering at ``line_width=1`` will be more subtle:

.. code-block:: python

    o3d.visualization.draw([sphere,line_set], line_width=1)

.. image:: https://user-images.githubusercontent.com/93158890/158695717-042343a4-bbc3-45b8-ab6b-1118ad027cd7.jpg
    :width: 700px

Experiment with the ``line_width`` parameter values to find an optimal one for
your purposes.

Commonly used ``draw()`` options
--------------------------------

Displaying UI, window titles, and specifying window dimensions
--------------------------------------------------------------

Aside from rendering 3D objects, you can use the ``draw()`` function calls to
control a number of Open3D Visualizer display options that are not shown by
default, such as:

* displaying UI / control panel for interactively modifying 3D model rendering
  parameters of the Visualizer
* adding a Visualizer window title;
* specifying window dimensions (i.e. ``width`` and ``height``).

The code below illustrates how to rename a Visualizer title bar and set window
``width`` and ``height`` by customizing the ``draw()`` call, using our prior
:ref:`bounding_box_sphere` example:

.. code-block:: python

    o3d.visualization.draw([sphere,line_set], show_ui=True, title="Sphere and AABB LineSet", width=700, height=700)

.. image:: https://user-images.githubusercontent.com/93158890/158281728-994ff828-53b0-485a-9feb-9b121d7354f7.jpg
    :width: 700px

At the bottom of the UI / control panel, you can see the section titled
"*Geometries*" (outlined in a dark grey box). This section contains a list of
rendered objects that can be individually turned on or off by clicking a
checkbox to the left of their names.

Assigning names to objects in the UI
------------------------------------

Object collections
::::::::::::::::::

In prior examples, we used the the ``draw()`` function to render 3D objects
explicitly. The ``draw()`` function is not limited to 3D Objects only. You can
create a collection of objects with their properties, mix them with
visualizer-specific options, and render the result. In the previous example, we
learned how to control a number of Open3D Visualizer display options that are
not shown by default. In this case, our goal is to rename the default-assigned
name of *Object 1* in the "Geometries" frame of the Visualizer UI to *sphere* .

We now declare the ``geoms`` collection which will contain a geometry object
``sphere`` (from previous examples), and we will name it *sphere* (``"name":
"sphere"``). This will serve as a signal to the Visualizer UI to replace its
default "Geometries" from *Object 1* to *sphere*:

.. code-block:: python

    geoms = {"name": "sphere", "geometry": sphere}

We can now display the UI and confirm that our custom object is named
appropriately:

.. code-block:: python

    o3d.visualization.draw(geoms, show_ui=True)

And here is the named object:

.. image:: https://user-images.githubusercontent.com/93158890/159092908-a2462f6d-34fc-4703-9845-9b311a7f1630.jpg
    :width: 700px

So far, our ``geoms`` collection defined only a single object: *sphere*. But we
can turn it into a list and define multiple objects there:

1. Re-declare ``geoms`` object to contain a collection list of the ``sphere``
   and ``aabb`` bounding box from the :ref:`bounding_box_sphere` section.

2. Call ``draw(geoms, show_ui=True)``:

.. code-block:: python

    geoms = [{"name": "sphere", "geometry": sphere}, {"name": "Axis Aligned Bounding Box line_set", "geometry": line_set}]
    o3d.visualization.draw(geoms, show_ui=True)

.. image:: https://user-images.githubusercontent.com/93158890/159094500-83ddd46f-0e71-40e1-9b97-ae46480cd860.jpg
    :width: 700px

More ``draw()`` options
-----------------------

``show_skybox`` and ``bg_color``
::::::::::::::::::::::::::::::::

Aside from naming Open3D Visualizer status bar, geometries, and displaying the
UI, you also have options to programmatically turn the light blue *skybox* on or
off (``show_skybox=False/True``) as well as change the background color
(``bg_color=(x.x, x.x, x.x, x.x)``).

First, we'll demonstrate how to turn off the *skybox* using our *sphere*
example. At your Python prompt, enter:

.. code-block:: python

    o3d.visualization.draw(sphere, show_ui=True, show_skybox=False)

And the Visualizer window opens without the default *skybox* blue background:

.. image:: https://user-images.githubusercontent.com/93158890/159093215-31dcacf7-306f-4231-9155-0df474ce4828.jpg
    :width: 700px

Next, we will explore the *background color* (``bg_color``) parameter. At the
Python prompt, enter:

.. code-block:: python

    o3d.visualization.draw(sphere, show_ui=True, title="Green Background", show_skybox=False, bg_color=(0.56, 1.0, 0.69, 1.0))

Here, we have displayed the UI, renamed the title bar to *"Green Background"*,
turned off the default *skybox* background, and explicitly specified RGB-Alfa
values for the ``bg_color``:

.. image:: https://user-images.githubusercontent.com/93158890/160878317-a57755a0-8b8f-44db-b718-443aa435035a.jpg
    :width: 700px

