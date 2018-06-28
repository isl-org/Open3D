.. _color_map_optimization:

Color Map Optimization
-------------------------------------

Consider color mapping to the geometry reconstructed from depth cameras. As color and depth frames are not perfectly aligned, the texture mapping using color images is subject to results in blurred color map. Open3D provides color map optimization method proposed by [Zhou2014]_. Before begin, download fountain dataset from `here <https://drive.google.com/open?id=1eT45y8qw3TLED2YY9-K1Ot6dQuF9GDPJ>`_. The following script shows an example of color map optimization.

.. code-block:: python

    from py3d import *
    from trajectory_io import *
    import os, sys
    sys.path.append("../Utility")
    from common import *

    path = "[set_this_path_to_fountain_dataset]"
    debug_mode = False

        if __name__ == "__main__":
            set_verbosity_level(VerbosityLevel.Debug)

            # Read RGBD images
            rgbd_images = []
            depth_image_path = get_file_list(
                    os.path.join(path, "depth/"), extension = ".png")
            color_image_path = get_file_list(
                    os.path.join(path, "image/"), extension = ".jpg")
            assert(len(depth_image_path) == len(color_image_path))
            for i in range(len(depth_image_path)):
                depth = read_image(os.path.join(depth_image_path[i]))
                color = read_image(os.path.join(color_image_path[i]))
                rgbd_image = create_rgbd_image_from_color_and_depth(color, depth,
                        convert_rgb_to_intensity = False)
                if debug_mode:
                    pcd = create_point_cloud_from_rgbd_image(rgbd_image,
                            PinholeCameraIntrinsic.get_prime_sense_default())
                    draw_geometries([pcd])
                rgbd_images.append(rgbd_image)

            # Read camera pose and mesh
            camera = read_pinhole_camera_trajectory(os.path.join(path, "scene/key.log"))
            mesh = read_triangle_mesh(os.path.join(path, "scene", "integrated.ply"))

            # Before full optimization, let's just visualize texture map
            # with given geometry, RGBD images, and camera poses.
            option = ColorMapOptmizationOption()
            option.maximum_iteration = 0
            color_map_optimization(mesh, rgbd_images, camera, option)
            draw_geometries([mesh])
            write_triangle_mesh(os.path.join(path, "scene",
                "color_map_before_optimization.ply"), mesh)

            # Optimize texture and save the mesh as texture_mapped.ply
            # This is implementation of following paper
            # Q.-Y. Zhou and V. Koltun,
            # Color Map Optimization for 3D Reconstruction with Consumer Depth Cameras,
            # SIGGRAPH 2014
            option.maximum_iteration = 500
            option.non_rigid_camera_coordinate = True
            color_map_optimization(mesh, rgbd_images, camera, option)
            draw_geometries([mesh])
            write_triangle_mesh(os.path.join(path, "scene",
                "color_map_after_optimization.ply"), mesh)

Input
````````````````````````

.. code-block:: python

    # read RGBD images
    rgbd_images = []
    depth_image_path = get_file_list(
            os.path.join(path, "depth/"), extension=".png")
    color_image_path = get_file_list(
            os.path.join(path, "image/"), extension=".jpg")
    assert(len(depth_image_path) == len(color_image_path))
    for i in range(len(depth_image_path)):
        depth = read_image(os.path.join(depth_image_path[i]))
        color = read_image(os.path.join(color_image_path[i]))
        rgbd_image = create_rgbd_image_from_color_and_depth(color, depth,
                convert_rgb_to_intensity=False)
        if debug_mode:
            pcd = create_point_cloud_from_rgbd_image(rgbd_image,
                    PinholeCameraIntrinsic.get_prime_sense_default())
            draw_geometries([pcd])
        rgbd_images.append(rgbd_image)

This script reads color and depth image pairs and makes ``rgbd_image``. Note that ``convert_rgb_to_intensity`` flag is ``False``. This is to preserve 8-bit color channels instead of using single channel float type image.

It is always good practice to visualize RGBD image before applying it to color map optimization. ``debug_mode`` switch is for visualizing RGBD image.

.. code-block:: python

    # read camera pose and mesh
    camera = read_pinhole_camera_trajectory(os.path.join(path, "scene/key.log"))
    mesh = read_triangle_mesh(os.path.join(path, "scene", "integrated.ply"))

The script reads camera trajectory and mesh.

.. code-block:: python

    option = ColorMapOptmizationOption()
    option.maximum_iteration = 0
    color_map_optimization(mesh, rgbd_images, camera, option)
    draw_geometries([mesh])
    write_triangle_mesh(os.path.join(path, "scene",
        "color_map_before_optimization.ply"), mesh)

To visualize how the camera poses are not good for color mapping, this script intentionally set the iteration number as 0, which means no optimization. ``color_map_optimization`` paints a mesh using corresponding RGBD images and camera poses. Without optimization, the texture map is blurred.

.. image:: ../../_static/Advanced/color_map_optimization/initial.png
    :width: 300px

.. image:: ../../_static/Advanced/color_map_optimization/initial_zoom.png
    :width: 300px

Rigid Optimization
```````````````````````````````

The next step is to optimize camera poses to get a sharp color map.

.. code-block:: python

    option.maximum_iteration = 500
    option.non_rigid_camera_coordinate = True
    color_map_optimization(mesh, rgbd_images, camera, option)
    draw_geometries([mesh])
    write_triangle_mesh(os.path.join(path, "scene",
        "color_map_after_optimization.ply"), mesh)

The script sets ``maximum_iteration = 500`` for actual iterations. The optimization displays the following energy profile.

.. code-block:: shell

    [ColorMapOptimization] :: Rigid Optimization
    [Iteration 0001] Residual error : 25777.372725 (avg : 0.004998)
    [Iteration 0002] Residual error : 25620.681829 (avg : 0.004967)
    [Iteration 0003] Residual error : 25463.806101 (avg : 0.004937)
    :
    [Iteration 0498] Residual error : 11550.014763 (avg : 0.002255)
    [Iteration 0499] Residual error : 11549.850827 (avg : 0.002255)
    [Iteration 0500] Residual error : 11550.062068 (avg : 0.002255)

Residual error implies inconsistency of image intensities. Lower residual leads better color map quality. By default, ``ColorMapOptmizationOption`` enables rigid optimization. It optimizes 6-dimentional pose of every cameras.

.. image:: ../../_static/Advanced/color_map_optimization/rigid.png
    :width: 300px

.. image:: ../../_static/Advanced/color_map_optimization/rigid_zoom.png
    :width: 300px

Non-rigid Optimization
```````````````````````````````````

For better alignment quality, there is an option for non-rigid optimization. To enable, simply add

.. code-block:: python

    option.non_rigid_camera_coordinate = True

before calling ``color_map_optimization``. Besides 6-dimentional camera poses, non-rigid optimization even consider local image warping represented by anchor points. This adds even more flexibility and leads higher quality color mapping. The residual error is smaller than the case of rigid optimization.

.. code-block:: shell

    [ColorMapOptimization] :: Non-Rigid Optimization
    [Iteration 0001] Residual error : 25777.372725, reg : 0.000000
    [Iteration 0002] Residual error : 25330.445704, reg : 13.005639
    [Iteration 0003] Residual error : 24885.912182, reg : 40.000765
    :
    [Iteration 0498] Residual error : 7585.606850, reg : 3294.124184
    [Iteration 0499] Residual error : 7585.274846, reg : 3294.887659
    [Iteration 0500] Residual error : 7583.972930, reg : 3294.634065

Results of non-rigid optimization follow.

.. image:: ../../_static/Advanced/color_map_optimization/non_rigid.png
    :width: 300px

.. image:: ../../_static/Advanced/color_map_optimization/non_rigid_zoom.png
    :width: 300px
