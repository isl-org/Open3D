.. _scene_rendering:

Scene Rendering 
-------------------------------------

Open3D lets you render scenes that are heavily customized to any application using our ``rendering`` class. You can build a custom application using the ``gui`` module to host the rendering. These classes are built to be flexible and completely extensible while providing speed and complete interaction.

This tutorial take you through a python script that builds a rendering engine. The engine lets you manage all aspects of the render such as cameras, gradients, materials, and scenes. You must also review the ``GUI`` tutorial to understand how you can create a GUI to host the renderer. Please refer to Open3D/python/visualization/rendering/ for complete class definition. You can view some examples in Open3D/examples/python/rendering/.

Before you can render, you must have an application that can host the rendering. To understand how you can build an application using the ``gui`` module, see the GUI tutorial.


Overview
````````````````````````````````````````````````````
We assume that you have an application that can host the rendering. We will render a scene through the following steps:
#. Setting up the material for the scene
#. Add a sphere to the scene
#. Set up the camera
#. Initialize the window and rendering the scene


The full script of the this sample is show below:
.. literalinclude:: ../../../examples/python/visualization/render_scene.py
   :language: python
   :lineno-start: 07
   :lines: 07-45
   :linenos:

Setting up the material for the scene
=====================================
The following section of the script initializes the renderer and sets up the material.

.. literalinclude:: ../../../examples/python/visualization/render_scene.py
   :language: python
   :lineno-start: 13
   :lines: 13-17
   :linenos:

Add a sphere to scene
=====================================
The following section of the script sets up the properties for the sphere and add the sphere to the scene.

.. literalinclude:: ../../../examples/python/visualization/render_scene.py
   :language: python
   :lineno-start: 19
   :lines: 19-25
   :linenos:

Set up the camera for the scene
=====================================
The following section of the script set up the camera and lighting for the scene.

.. literalinclude:: ../../../examples/python/visualization/render_scene.py
   :language: python
   :lineno-start: 27
   :lines: 27-34
   :linenos:

Initialize the window and rendering the scene
==============================================
The following section of the script initiliazes the window, and adds the scene to the window.

.. literalinclude:: ../../../examples/python/visualization/render_scene.py
   :language: python
   :lineno-start: 36
   :lines: 36-46
   :linenos:
