.. _customized_visualization:

Customized visualization
-------------------------------------

The usage of Open3D convenient visualization functions ``draw_geometries`` and ``draw_geometries_with_custom_animation`` is straightforward. Everything can be done with the GUI. Press :kbd:`h` inside the visualizer window to see helper information. For more details, see :any:`/tutorial/visualization/visualization.ipynb`.

This tutorial focuses on more advanced functionalities to customize the behavior of the visualizer window. Please refer to examples/python/visualization/customized_visualization.py to try the following examples.


Mimic draw_geometries() with Visualizer class
````````````````````````````````````````````````````

.. literalinclude:: ../../../examples/python/visualization/customized_visualization.py
   :language: python
   :lineno-start: 37
   :lines: 37-44
   :linenos:

This function produces exactly the same functionality as the convenience function ``draw_geometries``.

.. image:: ../../_static/visualization/customized_visualization/custom.png
    :width: 400px

Class ``Visualizer`` has a couple of variables such as a ``ViewControl`` and a ``RenderOption``. The following function reads a predefined ``RenderOption`` stored in a json file.

.. literalinclude:: ../../../examples/python/visualization/customized_visualization.py
   :language: python
   :lineno-start: 70
   :lines: 70-76
   :linenos:

Outputs:

.. image:: ../../_static/visualization/customized_visualization/normal.png
    :width: 400px


Change field of view
````````````````````````````````````
To change field of view of the camera, it is first necessary to get an instance of the visualizer control. To modify the field of view, use ``change_field_of_view``.

.. literalinclude:: ../../../examples/python/visualization/customized_visualization.py
   :language: python
   :lineno-start: 47
   :lines: 47-56
   :linenos:

The field of view (FoV) can be set to a degree in the range [5,90]. Note that ``change_field_of_view`` adds the specified FoV to the current FoV. By default, the visualizer has an FoV of 60 degrees. Calling the following code

.. code-block:: python

    custom_draw_geometry_with_custom_fov(pcd, 90.0)

will add the specified 90 degrees to the default 60 degrees. As it exceeds the maximum allowable FoV, the FoV is set to 90 degrees.

.. image:: ../../_static/visualization/customized_visualization/fov_90.png
    :width: 400px

The following code

.. code-block:: python

    custom_draw_geometry_with_custom_fov(pcd, -90.0)

will set FoV to 5 degrees, because 60 - 90 = -30 is less than 5 degrees.

.. image:: ../../_static/visualization/customized_visualization/fov_5.png
    :width: 400px


Callback functions
````````````````````````````````````

.. literalinclude:: ../../../examples/python/visualization/customized_visualization.py
   :language: python
   :lineno-start: 59
   :lines: 59-67
   :linenos:

Function ``draw_geometries_with_animation_callback`` registers a Python callback function ``rotate_view`` as the idle function of the main loop. It rotates the view along the x-axis whenever the visualizer is idle. This defines an animation behavior.

.. image:: ../../_static/visualization/customized_visualization/rotate_small.gif
    :width: 400px

.. literalinclude:: ../../../examples/python/visualization/customized_visualization.py
   :language: python
   :lineno-start: 79
   :lines: 79-108
   :linenos:

Callback functions can also be registered upon key press event. This script registered four keys. For example, pressing :kbd:`k` changes the background color to black.

.. image:: ../../_static/visualization/customized_visualization/key_k.png
    :width: 400px

Capture images in a customized animation
`````````````````````````````````````````````````

.. literalinclude:: ../../../examples/python/visualization/customized_visualization.py
   :language: python
   :lineno-start: 109
   :lines: 111-162
   :linenos:

This function reads a camera trajectory, then defines an animation function ``move_forward`` to travel through the camera trajectory. In this animation function, both color image and depth image are captured using ``Visualizer.capture_depth_float_buffer`` and ``Visualizer.capture_screen_float_buffer`` respectively. The images are saved as png files.

The captured image sequence:

.. image:: ../../_static/visualization/customized_visualization/image_small.gif
    :width: 400px

The captured depth sequence:

.. image:: ../../_static/visualization/customized_visualization/depth_small.gif
    :width: 400px
