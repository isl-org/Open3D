.. _customized_visualization:

Customized visualization
-------------------------------------

The usage of Open3D convenient visualization functions ``draw_geometries`` and ``draw_geometries_with_custom_animation`` is straightforward. Everything can be done with the GUI. Press :kbd:`h` inside the visualizer window to see helper information. Details see :ref:`visualization`.

This tutorial focuses on more advanced functionalities to customize the behavior of the visualizer window. Please refer examples/Python/Advanced/customized_visualization.py to try the following examples.


Mimic draw_geometries() with Visualizer class
````````````````````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/customized_visualization.py
   :language: python
   :lineno-start: 13
   :lines: 13-20
   :linenos:

This function produces exactly the same functionality of the convenient function ``draw_geometries``.

.. image:: ../../_static/Advanced/customized_visualization/custom.png
    :width: 400px

Class ``Visualizer`` has a couple of variables such as a ``ViewControl`` and a ``RenderOption``. The following function reads a predefined ``RenderOption`` stored in a json file.

.. literalinclude:: ../../../examples/Python/Advanced/customized_visualization.py
   :language: python
   :lineno-start: 46
   :lines: 46-52
   :linenos:

Outputs:

.. image:: ../../_static/Advanced/customized_visualization/normal.png
    :width: 400px


Change field of view
````````````````````````````````````
To change field of view of the camera, it is necessary to get an instance of visualizer control first. To modify modify field of view, use ``change_field_of_view``.

.. literalinclude:: ../../../examples/Python/Advanced/customized_visualization.py
   :language: python
   :lineno-start: 23
   :lines: 23-32
   :linenos:

The field of view can be set as [5,90] degree. Note that ``change_field_of_view`` adds specified FoV on the current FoV. By default, visualizer has 60 degrees of FoV. Calling the following code

.. code-block:: python

    custom_draw_geometry_with_custom_fov(pcd, 90.0)

will add the specified 90 degrees to the default 60 degrees. As it exceeds maximum allowable FoV, this will set FoV as 90 degrees.

.. image:: ../../_static/Advanced/customized_visualization/fov_90.png
    :width: 400px

The following code

.. code-block:: python

    custom_draw_geometry_with_custom_fov(pcd, -90.0)

will make FoV as 5 degrees, because 60 - 90 = -30 is smaller than 5 degrees.

.. image:: ../../_static/Advanced/customized_visualization/fov_5.png
    :width: 400px


Use callback functions
````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/customized_visualization.py
   :language: python
   :lineno-start: 35
   :lines: 35-43
   :linenos:

Function ``draw_geometries_with_animation_callback`` registers a Python callback function ``rotate_view`` as the idle function of the main loop. It rotates the view along the x-axis whenever the visualizer is idle. This defines an animation behavior.

.. image:: ../../_static/Advanced/customized_visualization/rotate_small.gif
    :width: 400px

.. literalinclude:: ../../../examples/Python/Advanced/customized_visualization.py
   :language: python
   :lineno-start: 55
   :lines: 55-84
   :linenos:

Callback functions can also be registered upon key press event. This script registered four keys. For example, pressing :kbd:`k` changes the background color to black.

.. image:: ../../_static/Advanced/customized_visualization/key_k.png
    :width: 400px

Capture images in a customized animation
`````````````````````````````````````````````````

.. literalinclude:: ../../../examples/Python/Advanced/customized_visualization.py
   :language: python
   :lineno-start: 87
   :lines: 87-134
   :linenos:

This function reads a camera trajectory, then defines an animation function ``move_forward`` to travel through the camera trajectory. In this animation function, both color image and depth image are captured using ``Visualizer.capture_depth_float_buffer`` and ``Visualizer.capture_screen_float_buffer`` respectively. They are saved in files.

The captured image sequence:

.. image:: ../../_static/Advanced/customized_visualization/image_small.gif
    :width: 400px

The captured depth sequence:

.. image:: ../../_static/Advanced/customized_visualization/depth_small.gif
    :width: 400px
