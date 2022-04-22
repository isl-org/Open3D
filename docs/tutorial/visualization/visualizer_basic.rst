.. _visualizer_basic:

Open3D Visualizer
=================

Introduction
---------------

Open3D provides a convenient function for visualizing geometric objects:
``draw()``. The ``draw()`` function allows to visualize multiple geometry
objects *(PointClouds, LineSets, TriangleMeshes)* and images together along with
optional, high-quality, physically based (PBR) materials. As will be
demonstrated in the subsequent sections, ``draw()`` can be used for both -
simple, quick visualization or complex use-cases.

Getting started
---------------

.. tip::
    This **Getting started** section applies to all subsequent examples below

For all examples in this tutorial, we will be using Python. Please follow these
preliminary steps :

1. Navigate to the ``Open3D`` location on your computer:

.. code-block:: sh

    $ cd <... Path to Open3D on your computer...>

2. **Optionally**, if you have a ``conda`` virtual environment, activate it from
   the command line like so:

.. code-block:: sh

    $ conda activate <...your virtual environment name...>

3. Run ``python``:

.. code-block:: sh

    $ python

4. In Python, set up common imports that we will use throughout this tutorial:

.. code-block:: python

    >>> import open3d as o3d
    >>> import open3d.visualization as vis
    >>> import numpy as np
    >>> import math


.. note::
    * The ``numpy`` object we are declaring is needed for conversions of lists to arrays.
    * The ``math`` library is needed to reference Pi and rotate objects

Below is a screenshot of how you would set up your environment from the command
terminal:


.. image:: https://user-images.githubusercontent.com/93158890/159073961-821e4768-3678-4385-bc37-20c5b212c030.jpg
    :width: 700px




Basic examples
--------------

In the Overview section, we activated a ``conda`` environment, started a Python
session, and declared Open3D objects to be used throughout this tutorial. Let’s
now test various Open3D ``draw()`` function capabilities with various
geometries.


Drawing point clouds
::::::::::::::::::::

Using Open3D datasets
"""""""""""""""""""""

In this example, we are going to learn how to load and render Point Clouds. To
retrieve our example, we will be using **Open3D Datasets**.


.. tip::

    Open3D provides a built-in *dataset* module for retrieval of commonly used
    3D model examples.

    * Datasets are automatically downloaded from the Internet and cached
      locally.
    * The **default local dataset  download directory** is ``~/open3d_data``.
    * Datasets will be downloaded to ``~/open3d_data/download`` and extracted to
      ``~/open3d_data/extract``


.. seealso::

    For more information on datasets, please refer to the :doc:`Open3D Datasets
    page <../data/index>`


Enter the following code at the Python prompt:

.. code-block:: python

    # Download and initialize the dataset >>> dataset = o3d.data.PLYPointCloud()

    # Create a Point Cloud object (pcd) from the dataset >>> pcd =
    o3d.io.read_point_cloud(dataset.path)

    # Customize the pcd object >>> rotate_180 =
    o3d.geometry.get_rotation_matrix_from_xyz((-math.pi, 0, 0)) >>>
    pcd.rotate(rotate_180) >>> vis.draw(pcd)

Open3D returns:

.. image:: https://user-images.githubusercontent.com/93158890/159548100-404afe97-8960-4e68-956f-cc6957632a93.jpg
    :width: 700px

Specifying ``point_size``
"""""""""""""""""""""""""

In this section, we will learn how to control 3D model rendering by passing in
``point_size`` as a parameter to the ``draw()`` function. To do this, let's
enter the following code at the Python prompt:

.. code-block:: python

    >>> vis.draw(pcd, point_size=9, show_ui=True)

Here we have programmatically specified a custom ``point_size`` for rendering.
It is recommended to set ``show_ui=True`` to make sure Open3D Visualizer
interprets ``draw()`` function input parameters correctly. You can experiment
with different point sizes by moving a slider in the UI:

.. image:: https://user-images.githubusercontent.com/93158890/159555822-5eb3562b-4432-4a73-ab48-342b0cd2a898.jpg
    :width: 700px


Drawing a box
:::::::::::::

Aside from rendering Point Clouds, the Open3D ``draw()`` function is fully
capable of rendering primitives, such as circles, spheres, rectangles, cubes,
etc..

This example shows how to create and visualize a simple 3D box.


At the python prompt, enter the following to open the 3D Visualizer:

.. code-block:: python

    >>> cube = o3d.geometry.TriangleMesh.create_box(1, 2, 4)
    >>> vis.draw(cube)

At the end of the process, the Open3D Visualizer window should appear:

.. image:: https://user-images.githubusercontent.com/93158890/148607529-ee0ae0de-05af-423d-932c-2a5a6c8d7bda.jpg
    :width: 700px

Let's examine what we did here:

1) We instantiated the ``cube`` object to be of ``open3d.geometry.TriangleMesh``
   type using the function ``create_box(1, 2, 4)`` to which we passed values for
   width (``1``), height (``2``), and depth (``4``);

2) We called the ``open3d.visualization.draw()`` method which rendered our
   ``cube``.



.. _compute_triangle_normals_s:

``compute_triangle_normals()`` method
"""""""""""""""""""""""""""""""""""""

In the above example we learned how to create a primitive (``cube``) and render
it with the ``draw()`` call. To improve it, we need to introduce some sort of
surface reflection information to give our object a better, more consistent 3D
look. For this, we will use the ``compute_triangle_normals()`` method as shown
below:

.. code-block:: python

    >>> cube.compute_triangle_normals()
    >>> vis.draw(cube)

Clearly, that makes a big difference:

.. image:: https://user-images.githubusercontent.com/93158890/157720147-cde9a54b-cba5-480e-ba0e-7784b5bd5677.jpg
    :width: 700px

The algorithm behind ``compute_triangle_normals()`` **computes a single normal
for every triangle** in a ``TriangleMesh``.



.. _smoothly_lit_sphere:

Drawing a smoothly lit sphere
:::::::::::::::::::::::::::::


``compute_vertex_normals()`` method
"""""""""""""""""""""""""""""""""""

In this example, we will learn how to draw a sphere using a different rendering
technique, represented by the ``compute_vertex_normals()`` method.
``compute_vertex_normals()`` uses an algorithm which **computes a smooth normal
at every vertex** of the triangle unit in a ``TriangleMesh``.

At the Python prompt in your terminal, enter the following lines of code:

.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    >>> sphere.compute_vertex_normals()
    >>> vis.draw(sphere)

A rendered sphere appears:

.. image:: https://user-images.githubusercontent.com/93158890/157339234-1a92a944-ac38-4256-8297-0ad78fd24b9c.jpg
    :width: 700px


As you can see, calling ``compute_vertex_normals()`` on the ``sphere`` object
gave us a realistic rendering of a ball-like object.

To see what type of rendering was used to draw our ``sphere`` above, at the
Python prompt, enter:

.. code-block:: python

    >>> sphere

Open3D returns:

.. code-block:: sh

    TriangleMesh with 19802 points and 39600 triangles.


Drawing a flat-shaded sphere
:::::::::::::::::::::::::::::

In this example, we are going to use a ``compute_triangle_normals()`` rendering
algorithm, - the same method we used for a 3D ``cube`` rendering before (see
:ref:`compute_triangle_normals_s`). Again, **this algorithm computes a single
normal for every triangle** in a ``TriangleMesh``:


.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0)
    >>> sphere.compute_triangle_normals()
    >>> vis.draw(sphere)


.. image:: https://user-images.githubusercontent.com/93158890/157728100-0a495e56-c613-40c4-a292-6e45213d61f6.jpg
    :width: 700px


The rendered sphere in this case has facets akin to what XIX-th century airships
or blimps used to look like.


Drawing a colored lit sphere
::::::::::::::::::::::::::::

``paint_uniform_color()``
"""""""""""""""""""""""""

When we rendered a lit sphere in one of the previous sections
(:ref:`smoothly_lit_sphere`), we did not specify which color we would like the
sphere to be. In this example, we will assign a subtle pink color to the sphere
with the ``paint_uniform_color()`` method:

.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    >>> sphere.compute_vertex_normals()
    >>> sphere.paint_uniform_color([0.65, 0.45, 0.62])
    >>> vis.draw(sphere)

.. image:: https://user-images.githubusercontent.com/93158890/160883817-5a22f449-62e2-45e0-8033-bfec72e09210.jpg
    :width: 700px

The ``paint_uniform_color()`` method accepts a numeric list of RGB values. Its
algorithm assigns a single color to all vertices of the triangle mesh. RGB
values should be in the ``0 - 1`` range. In our example, we passed respective
values for Red (``0.65``), Green (``0.45``), and Blue (``0.62``).


Drawing a sphere with materials
:::::::::::::::::::::::::::::::

In previous examples we only specified the geometry to visualize, and the
``draw()`` function internally created a default material for it. However, with
the ``draw()`` function you can render geometries with customized materials.

Let's create a sphere based on a custom material:


.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    >>> sphere.compute_vertex_normals()
    >>> mat = vis.rendering.MaterialRecord()
    >>> mat.shader = "defaultLit"
    >>> mat.base_color = np.asarray([1.0, 0.0, 1.0, 1.0])

We declare ``mat`` as a material rendering object and initialize it with a
default lighting scheme.

``rendering`` is a submodule of ``open3d.visualization``.

``MaterialRecord()`` is a structure which holds various material properties.

The ``shader`` property accepts a string representing the type of material. The
two most common options are ``'defaultLit'`` and ``'defaultUnlit'``. Its other
options will be covered in :doc:`visualizer_advanced` tutorial.

The ``mat.base_color`` represents the base material RGBA color. It expects a
``numpy`` array as a parameter. The ``numpy`` module we imported at the very
beginning of this tutorial helps us pass the RGBA values as an array to the
``mat.base_color`` property.

To find out what type of object *mat* is, we type in ``mat`` at the Python
prompt:

.. code-block:: python

    >>> mat
    <open3d.cpu.pybind.visualization.rendering.MaterialRecord object at 0x7f2be5e34430>


Now, we'll show a ``draw()`` call variant which allows the user to specify a
material to use with the geometry. This is different from previous examples
where the ``draw()`` call created a default material automatically:

.. code-block:: python

    >>> vis.draw({'name': 'sphere', 'geometry': sphere, 'material': mat})

.. image:: https://user-images.githubusercontent.com/93158890/150883605-a5e65a3f-0a25-4ff4-b039-4aa6e53a1440.jpg
    :width: 700px



Drawing a metallic sphere
:::::::::::::::::::::::::

In earlier examples, we used ``create_sphere()`` to render the sphere with basic
RGB/RGBA colors. Next, we will look at other material properties.

.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    >>> sphere.compute_vertex_normals()
    >>> rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi / 2, 0, 0))
    >>> sphere.rotate(rotate_90)
    >>> mat = vis.rendering.MaterialRecord()
    >>> mat.shader = "defaultLit"
    >>> mat.base_color = np.asarray([0.8, 0.9, 1.0, 1.0])
    >>> mat.base_roughness = 0.4
    >>> mat.base_metallic = 1.0
    >>> vis.draw({'name': 'sphere', 'geometry': sphere, 'material': mat}, ibl="nightlights")


.. image:: https://user-images.githubusercontent.com/93158890/157758092-9efb1ca0-b96a-4e1d-abd7-95243b279d2e.jpg
    :width: 700px

Let's examine new elements in the code above:

``rotate_90`` - utility object from a special function -
``get_rotation_matrix_from_xyz()`` - for creating a rotation matrix given angles
to rotate around the ``x``, ``y``, and ``z`` axes.

``sphere.rotate(rotate_90)`` - rotates the triangle mesh based on a rotation
matrix object we pass in.

``mat.base_roughness = 0.4`` - PBR (Physically-Based Rendering) material
property which controls the smoothness of the surface (see  `Filament Material
Guide <https://google.github.io/filament/Materials.html>`_ for details)

``mat.base_metallic = 1.0`` - PBR material property which defines whether the
surface is metallic or not (see  `Filament Material Guide
<https://google.github.io/filament/Materials.html>`_ for details)

``vis.draw({'name': 'sphere', 'geometry': sphere, 'material': mat},
ibl="nightlights")`` -  a different variant of the ``draw()`` call which uses
the ``ibl`` (Image Based Lighting) property. The *'ibl'* parameter property
allows the user to specify the HDR lighting to use. We assigned
``"nightlights"`` to ``ibl``, and thus get a realistic nighttime city scene.



Drawing a glossy sphere
:::::::::::::::::::::::

In a previous metallic sphere rendering we covered a number of methods,
parameters, and properties for beautifying its display. Let's now create a
non-metallic balloon-like sphere and see what transpires:


.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    >>> sphere.compute_vertex_normals()
    >>> rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.  pi / 2, 0, 0))
    >>> sphere.rotate(rotate_90)
    >>> mat = vis.rendering.MaterialRecord()
    >>> mat.shader = "defaultLit"
    >>> mat.base_color = np.asarray([0.8, 0.9, 1.0, 1.0])
    >>> mat.base_roughness = 0.25
    >>> mat.base_reflectance = 0.9
    >>> vis.draw({'name': 'sphere', 'geometry': sphere, 'material':   mat}, ibl="nightlights")

.. image:: https://user-images.githubusercontent.com/93158890/157770798-2c42e7dc-e063-4f26-90b4-16a45e263f36.jpg
    :width: 700px


This code is similar to that used in the rendering of a previous metallic
sphere. But, there are a couple of elements that make this version of the sphere
look different:

``mat.base_roughness = 0.25`` - PBR material roughness here is set to ``0.25``
in contrast to the previous metallic sphere version, where ``base_roughness``
was set to ``0.4``.

``mat.base_reflectance = 0.9`` - PBR material property which controls the
reflectance (glossiness) of the surface (see  `Filament Material Guide
<https://google.github.io/filament/Materials.html>`_ for details)

The ``draw()`` call here is identical to the metallic version of the sphere.



Drawing a sphere with textures
::::::::::::::::::::::::::::::

Running the code
""""""""""""""""

In this example, we will add textures to rendered objects:

.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100, create_uv_map=True)
    >>> sphere.compute_vertex_normals()
    >>> rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.pi / 2, 0, 0))
    >>> sphere.rotate(rotate_90)

    # Get the texture data from the dataset >>> mat_data =
    o3d.data.TilesTexture()

    # Create the material >>> mat = o3d.visualization.rendering.MaterialRecord()
    >>> mat.shader = "defaultLit"

    # Load graphic texture files from the dataset into material properties >>>
    mat.albedo_img = o3d.io.read_image(mat_data.albedo_texture_path) >>>
    mat.normal_img = o3d.io.read_image(mat_data.normal_texture_path) >>>
    mat.roughness_img = o3d.io.read_image(mat_data.roughness_texture_path) >>>
    vis.draw({'name': 'sphere', 'geometry': sphere, 'material': mat},
    ibl="nightlights")


.. image:: https://user-images.githubusercontent.com/93158890/157775220-443aad2d-9123-42d0-b584-31e9fb8f38c3.jpg
    :width: 700px


Let's examine new method calls and properties in this rendering:

``create_sphere(2.0, 100, create_uv_map=True)`` - generates texture coordinates
for the sphere that can be used later with textures

``mat.albedo_img`` - modifies the base color of the geometry

``mat.normal_img`` - modifies the normal of the geometry

``mat.roughness_img`` - modifies the roughness

All three properties are initialized by the ``o3d.io.read_image()`` method which
loads an image in either JPEG or PNG format.



.. _trianglemesh_lineset:

Drawing a wireframe sphere
::::::::::::::::::::::::::

Line Sets are typically used to display a wireframe of a 3D model. Let's do that
by creating a custom ``LineSet`` object:

.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 25)
    >>> sphere.compute_vertex_normals()
    >>> rotate_90 = o3d.geometry.get_rotation_matrix_from_xyz((-math.  pi / 2, 0, 0))
    >>> sphere.rotate(rotate_90)
    >>> line_set = o3d.geometry.LineSet.create_from_triangle_mesh  (sphere)
    >>> line_set.paint_uniform_color([0.0, 0.0, 1.0])
    >>> vis.draw(line_set)


.. image:: https://user-images.githubusercontent.com/93158890/157949589-8b87fa81-a5cf-4791-a4f7-2d5dc91e546e.jpg
    :width: 700px

So, what's new in this code?

``line_set = o3d.geometry.LineSet.create_from_triangle_mesh(sphere)`` - here we
create a line set from the edges of individual triangles of a triangle mesh.

``line_set.paint_uniform_color([0.0, 0.0, 1.0])`` - here we paint the wireframe
``LineSet`` blue. [*Red=0, Green=0, Blue=1*]



.. _bounding_box_sphere:

Drawing a sphere in a bounding box ``LineSet``
::::::::::::::::::::::::::::::::::::::::::::::

Rendering multiple objects
""""""""""""""""""""""""""

In prior examples, we rendered only one 3D object at a time. But the ``draw()``
function can be used to render multiple 3D objects simultaneously. In this
example, we will render two objects: the **Sphere** and its **Axis-Aligned
Bounding Box** represented by a cubic frame around the sphere:


.. code-block:: python

    >>> sphere = o3d.geometry.TriangleMesh.create_sphere(2.0, 100)
    >>> sphere.compute_vertex_normals()
    >>> aabb = o3d.geometry.AxisAlignedBoundingBox.create_from_points(sphere.vertices)
    >>> line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)
    >>> line_set.paint_uniform_color([0, 0, 1])
    >>> vis.draw([sphere,line_set])


Both objects appear and can be moved and rotated:

.. image:: https://user-images.githubusercontent.com/93158890/157901535-fbe78fc0-9b85-476e-a0a1-01e0e5d80738.jpg
    :width: 700px

Let's go over the new code here:

``aabb`` stands for *axis-aligned bounding box*.

``aabb =
o3d.geometry.AxisAlignedBoundingBox.create_from_points(sphere.vertices)`` -
creates a bounding box fully encompassing the sphere.


``LineSet`` objects
"""""""""""""""""""

As recently shown in the ``TriangleMesh LineSet`` Sphere example
(:ref:`trianglemesh_lineset`), Line Sets are used to render a wireframe of a 3D
model. In our case, we are creating a basic cubic frame around our sphere based
on the ``AxisAlignedBoundingBox`` object (``aabb``) we created earlier:

``line_set = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(aabb)``

``line_set.paint_uniform_color([0, 0, 1])`` - paints the bounding box
``LineSet`` blue.

Multiple object parameters in ``draw()`` calls
""""""""""""""""""""""""""""""""""""""""""""""

Finally, we have a ``draw()`` call with multiple 3D object parameters:

``vis.draw([sphere,line_set])``

You can pass as many objects to the ``draw()`` as you need.



Specifying wireframe ``line_width``
"""""""""""""""""""""""""""""""""""

Aside from rendering ``LineSet`` wireframes or grids, we can change their
thickness by passing in a ``line_width`` parameter with a numeric value to the
``draw()`` function like so:

.. code-block:: python

    >>> vis.draw([sphere,line_set], line_width=50)

Here we rendered a grotesquely thicker Bounding Box by increasing its thickness
(``line_width`` property) to ``50``:

.. image:: https://user-images.githubusercontent.com/93158890/158695002-f5976bfa-1e81-46dc-bf3b-b926d0c5e0af.jpg
    :width: 700px

The default value for the ``line_width`` parameter is ``2``. The minimum
supplied value is ``1``. The rendering at ``line_width=1`` will be more subtle:

.. code-block:: python

    >>> vis.draw([sphere,line_set], line_width=1)


.. image:: https://user-images.githubusercontent.com/93158890/158695717-042343a4-bbc3-45b8-ab6b-1118ad027cd7.jpg
    :width: 700px

Experiment with the ``line_width`` parameter values to find an optimal one for
your purposes.



Commonly used ``draw()`` options
--------------------------------

Displaying UI, window titles, and specifying window dimensions
::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::

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

    >>> vis.draw([sphere,line_set], show_ui=True, title="Sphere and AABB LineSet", width=700, height=700)

.. image:: https://user-images.githubusercontent.com/93158890/158281728-994ff828-53b0-485a-9feb-9b121d7354f7.jpg
    :width: 700px


At the bottom of the UI / control panel, you can see the section titled
"*Geometries*" (outlined in a dark grey box). This section contains a list of
rendered objects that can be individually turned on or off by clicking a
checkbox to the left of their names.





Assigning names to objects in the UI
::::::::::::::::::::::::::::::::::::

Object collections
""""""""""""""""""

In prior examples, we used the the ``draw()`` function to render 3D objects
explicitly. The ``draw()`` function is not limited to 3D Objects only. You can
create a collection of objects with their properties, mix them with
visualizer-specific options, and render the result. In the previous example, we
learned how to control a number of Open3D Visualizer display options that are
not shown by default. In this case, our goal is to rename the default-assigned
name of *Object 1* in the "Geometries" frame of the Visualizer UI to *sphere* .

We now declare the ``geoms`` collection which will contain a geometry object
``sphere`` (from previous examples), and we will name it *sphere* (``'name':
'sphere'``). This will serve as a signal to the Visualizer UI to replace its
default "Geometries" from *Object 1* to *sphere*:

.. code-block:: python

    >>> geoms = {'name': 'sphere', 'geometry': sphere}

We can now display the UI and confirm that our custom object is named
appropriately:

.. code-block:: python

    >>> vis.draw(geoms, show_ui=True)

And here is the named object:

.. image:: https://user-images.githubusercontent.com/93158890/159092908-a2462f6d-34fc-4703-9845-9b311a7f1630.jpg
    :width: 700px

So far, our ``geoms`` collection defined only a single object: *sphere*. But we
can turn it into a list and define multiple objects there:

1. Re-declare ``geoms`` object to contain a collection list of the ``sphere``
   and ``aabb`` bounding box from the :ref:`bounding_box_sphere` section.

2. Call ``draw(geoms, show_ui=True)``:

.. code-block:: python

    >>> geoms = [{'name': 'sphere', 'geometry': sphere}, {'name': 'Axis Aligned Bounding Box line_set', 'geometry': line_set}]
    >>> vis.draw(geoms, show_ui=True)

.. image:: https://user-images.githubusercontent.com/93158890/159094500-83ddd46f-0e71-40e1-9b97-ae46480cd860.jpg
    :width: 700px



More ``draw()`` options
:::::::::::::::::::::::

``show_skybox`` and ``bg_color``
""""""""""""""""""""""""""""""""

Aside from naming Open3D Visualizer status bar, geometries, and displaying the
UI, you also have options to programmatically turn the light blue *skybox* on or
off (``show_skybox=False/True``) as well as change the background color
(``bg_color=(x.x, x.x, x.x, x.x)``).

First, we'll demonstrate how to turn off the *skybox* using our *sphere*
example. At your Python prompt, enter:

.. code-block:: python

    >>> vis.draw(sphere, show_ui=True, show_skybox=False)

And the Visualizer window opens without the default *skybox* blue background:

.. image:: https://user-images.githubusercontent.com/93158890/159093215-31dcacf7-306f-4231-9155-0df474ce4828.jpg
    :width: 700px

Next, we will explore the *background color* (``bg_color``) parameter. At the
Python prompt, enter:

.. code-block:: python

    >>> vis.draw(sphere, show_ui=True, title="Green Background", show_skybox=False, bg_color=(0.56, 1.0, 0.69, 1.0))

Here, we have displayed the UI, renamed the title bar to *"Green Background"*,
turned off the default *skybox* background, and explicitly specified RGB-Alfa
values for the ``bg_color``:

.. image:: https://user-images.githubusercontent.com/93158890/160878317-a57755a0-8b8f-44db-b718-443aa435035a.jpg
    :width: 700px





