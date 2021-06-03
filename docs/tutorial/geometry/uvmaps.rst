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

*****************************
Example Texture Map
*****************************


.. image:: ../../_static/geometry/uvmaps/uv1.png
    :width: 200px
    :align: center

************************************
Box (map uv to each face = false) 
************************************


.. image:: ../../_static/geometry/uvmaps/uv2.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv3.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv4.png
    :width: 200px

**************************************
Box (map uv to each face = true)
**************************************

.. image:: ../../_static/geometry/uvmaps/uv5.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv6.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv7.png
    :width: 200px


*************
Tetrahedral
*************

.. image:: ../../_static/geometry/uvmaps/uv8.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv9.png
    :width: 200px


.. image:: ../../_static/geometry/uvmaps/uv10.png
    :width: 200px

***************
Octahedral
***************

.. image:: ../../_static/geometry/uvmaps/uv11.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv12.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv13.png
    :width: 200px

**************
Icosahedron
**************

.. image:: ../../_static/geometry/uvmaps/uv14.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv15.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv16.png
    :width: 200px

*************
Cylinder
*************

.. image:: ../../_static/geometry/uvmaps/uv17.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv18.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv19.png
    :width: 200px

*******
Cone
*******
.. image:: ../../_static/geometry/uvmaps/uv20.png
    :width: 200px
    
.. image:: ../../_static/geometry/uvmaps/uv21.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv22.png
    :width: 200px

*******
Sphere
*******


.. image:: ../../_static/geometry/uvmaps/uv23.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv24.png
    :width: 200px

.. image:: ../../_static/geometry/uvmaps/uv25.png
    :width: 200px


