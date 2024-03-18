UV Maps
##########################

UV Maps
--------------------

********************
What is a UV Map ?
********************

[Ref: `Wikipedia <https://en.wikipedia.org/wiki/UV_mapping>`_] UV mapping is the 3D modeling process of projecting a 2D image to a 3D model's surface for texture mapping. The letters "U" and "V" denote the axes of the 2D texture because "X", "Y", and "Z" are already used to denote the axes of the 3D object in model space.
UV texturing permits polygons that make up a 3D object to be painted with color (and other surface attributes) from an ordinary image. The image is called a UV texture map.

*****************************
How to add custom UV maps ?
*****************************

* UV Map coordinates ``(U, V)`` are stored as ``std::vector<Eigen::Vector2d>`` of length ``3 x number of triangles``. So, there is a set of 3 (U, V) coordinates for each triangle, each associated with it's vertices.
* One may assume the UV map, maps a texture image of height and width of length 1.0 to the geometry. So, the range of U and V is from 0.0 to 1.0 (both inclusive).


Quick Reference to default UV Maps for some primitive shapes provided by Open3D
--------------------------------------------------------------------------------

The examples below all assume the following code preamble:

.. code-block:: python

    import open3d as o3d
    import open3d.visualization.rendering as rendering

    material = rendering.MaterialRecord()
    material.shader = 'defaultUnlit'
    material.albedo_img = o3d.io.read_image('/Users/renes/Downloads/uv1.png')


*****************************
Example Texture Map
*****************************

.. image:: ../../_static/geometry/uvmaps/uv1.png
    :width: 200px
    :align: center

************************************
Box (map uv to each face = false) 
************************************

.. code-block:: python


    box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True)
    o3d.visualization.draw({'name': 'box', 'geometry': box, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv2.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv3.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv4.png
    :width: 200px

**************************************
Box (map uv to each face = true)
**************************************

.. code-block:: python


    box = o3d.geometry.TriangleMesh.create_box(create_uv_map=True, map_texture_to_each_face=True)
    o3d.visualization.draw({'name': 'box', 'geometry': box, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv5.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv6.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv7.png
    :width: 200px


*************
Tetrahedral
*************

.. code-block:: python


    tetra = o3d.geometry.TriangleMesh.create_tetrahedron(create_uv_map=True)
    o3d.visualization.draw({'name': 'tetrahedron', 'geometry': tetra, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv8.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv9.png
    :width: 200px


.. image:: ../../_static/geometry/uvmaps/uv10.png
    :width: 200px

***************
Octahedral
***************

.. code-block:: python


    octo = o3d.geometry.TriangleMesh.create_octahedron(create_uv_map=True)
    o3d.visualization.draw({'name': 'octahedron', 'geometry': octo, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv11.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv12.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv13.png
    :width: 200px

**************
Icosahedron
**************

.. code-block:: python


    ico = o3d.geometry.TriangleMesh.create_icosahedron(create_uv_map=True)
    o3d.visualization.draw({'name': 'icosahedron', 'geometry': ico, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv14.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv15.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv16.png
    :width: 200px

*************
Cylinder
*************

.. code-block:: python


    cylinder = o3d.geometry.TriangleMesh.create_cylinder(create_uv_map=True)
    o3d.visualization.draw({'name': 'cylinder', 'geometry': cylinder, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv17.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv18.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv19.png
    :width: 200px

*******
Cone
*******

.. code-block:: python


    cone = o3d.geometry.TriangleMesh.create_cone(create_uv_map=True)
    o3d.visualization.draw({'name': 'cone', 'geometry': cone, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv20.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv21.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv22.png
    :width: 200px

*******
Sphere
*******

.. code-block:: python


    sphere = o3d.geometry.TriangleMesh.create_sphere(create_uv_map=True)
    o3d.visualization.draw({'name': 'sphere', 'geometry': sphere, 'material': material})

.. image:: ../../_static/geometry/uvmaps/uv23.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv24.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv25.png
    :width: 200px


