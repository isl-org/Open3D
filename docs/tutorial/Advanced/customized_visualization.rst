.. _customized_visualization:

Customized visualization
-------------------------------------

The usage of Open3D convenient visualization functions ``draw_geometries`` and ``draw_geometries_with_custom_animation`` is straightforward. Everything can be done with the GUI. Press :kbd:`h` inside the visualizer window to see helper information. Details see :ref:`visualization`.

This tutorial focuses on more advanced functionalities to customize the behavior of the visualizer window. Please refer src/Python/Tutorial/Advanced/customized_visualization.py to try the following examples.


Mimic draw_geometries() with Visualizer class
````````````````````````````````````````````````````

.. code-block:: python

    def custom_draw_geometry(pcd):
        # The following code achieves the same effect as:
        # draw_geometries([pcd])
        vis = Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.run()
        vis.destroy_window()

This function produces exactly the same functionality of the convenient function ``draw_geometries``.

.. image:: ../../_static/Advanced/customized_visualization/custom.png
    :width: 400px

Class ``Visualizer`` has a couple of variables such as a ``ViewControl`` and a ``RenderOption``. The following function reads a predefined ``RenderOption`` stored in a json file.

.. code-block:: python

    def custom_draw_geometry_load_option(pcd):
        vis = Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().load_from_json(
                "../../TestData/renderoption.json")
        vis.run()
        vis.destroy_window()

Outputs:

.. image:: ../../_static/Advanced/customized_visualization/normal.png
    :width: 400px


Change field of view
````````````````````````````````````
To change field of view of the camera, it is necessary to get an instance of visualizer control first. To modify modify field of view, use ``change_field_of_view``.

.. code-block:: python

    def custom_draw_geometry_with_custom_fov(pcd, fov_step):
        vis = Visualizer()
        vis.create_window()
        vis.add_geometry(pcd)
        ctr = vis.get_view_control()
        print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
        ctr.change_field_of_view(step = fov_step)
        print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
        vis.run()
        vis.destroy_window()

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

.. code-block:: python

    def custom_draw_geometry_with_rotation(pcd):
        def rotate_view(vis):
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)
            return False
        draw_geometries_with_animation_callback([pcd], rotate_view)

Function ``draw_geometries_with_animation_callback`` registers a Python callback function ``rotate_view`` as the idle function of the main loop. It rotates the view along the x-axis whenever the visualizer is idle. This defines an animation behavior.

.. image:: ../../_static/Advanced/customized_visualization/rotate_small.gif
    :width: 400px

.. code-block:: python

    def custom_draw_geometry_with_key_callback(pcd):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False
        def load_render_option(vis):
            vis.get_render_option().load_from_json(
                    "../../TestData/renderoption.json")
            return False
        def capture_depth(vis):
            depth = vis.capture_depth_float_buffer()
            plt.imshow(np.asarray(depth))
            plt.show()
            return False
        def capture_image(vis):
            image = vis.capture_screen_float_buffer()
            plt.imshow(np.asarray(image))
            plt.show()
            return False
        key_to_callback = {}
        key_to_callback[ord("K")] = change_background_to_black
        key_to_callback[ord("R")] = load_render_option
        key_to_callback[ord(",")] = capture_depth
        key_to_callback[ord(".")] = capture_image
        draw_geometries_with_key_callbacks([pcd], key_to_callback)

Callback functions can also be registered upon key press event. This script registered four keys. For example, pressing :kbd:`k` changes the background color to black.

.. image:: ../../_static/Advanced/customized_visualization/key_k.png
    :width: 400px

Capture images in a customized animation
`````````````````````````````````````````````````

.. code-block:: python

    def custom_draw_geometry_with_camera_trajectory(pcd):
        custom_draw_geometry_with_camera_trajectory.index = -1
        custom_draw_geometry_with_camera_trajectory.trajectory =\
                read_pinhole_camera_trajectory(
                        "../../TestData/camera_trajectory.json")
        custom_draw_geometry_with_camera_trajectory.vis = Visualizer()
        if not os.path.exists("../../TestData/image/"):
            os.makedirs("../../TestData/image/")
        if not os.path.exists("../../TestData/depth/"):
            os.makedirs("../../TestData/depth/")
        def move_forward(vis):
            # This function is called within the Visualizer::run() loop
            # The run loop calls the function, then re-render
            # So the sequence in this function is to:
            # 1. Capture frame
            # 2. index++, check ending criteria
            # 3. Set camera
            # 4. (Re-render)
            ctr = vis.get_view_control()
            glb = custom_draw_geometry_with_camera_trajectory
            if glb.index >= 0:
                print("Capture image {:05d}".format(glb.index))
                depth = vis.capture_depth_float_buffer(False)
                image = vis.capture_screen_float_buffer(False)
                plt.imsave("../../TestData/depth/{:05d}.png".format(glb.index),\
                        np.asarray(depth), dpi = 1)
                plt.imsave("../../TestData/image/{:05d}.png".format(glb.index),\
                        np.asarray(image), dpi = 1)
                #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
                #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
            glb.index = glb.index + 1
            if glb.index < len(glb.trajectory.extrinsic):
                ctr.convert_from_pinhole_camera_parameters(glb.trajectory.intrinsic,\
                        glb.trajectory.extrinsic[glb.index])
            else:
                custom_draw_geometry_with_camera_trajectory.vis.\
                        register_animation_callback(None)
            return False
        vis = custom_draw_geometry_with_camera_trajectory.vis
        vis.create_window()
        vis.add_geometry(pcd)
        vis.get_render_option().load_from_json("../../TestData/renderoption.json")
        vis.register_animation_callback(move_forward)
        vis.run()
        vis.destroy_window()

This function reads a camera trajectory, then defines an animation function ``move_forward`` to travel through the camera trajectory. In this animation function, both color image and depth image are captured using ``Visualizer.capture_depth_float_buffer`` and ``Visualizer.capture_screen_float_buffer`` respectively. They are saved in files.

The captured image sequence:

.. image:: ../../_static/Advanced/customized_visualization/image_small.gif
    :width: 400px

The captured depth sequence:

.. image:: ../../_static/Advanced/customized_visualization/depth_small.gif
    :width: 400px
