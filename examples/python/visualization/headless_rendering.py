# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/visualization/headless_rendering.py

import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt


def custom_draw_geometry_with_camera_trajectory(pcd):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
            o3d.io.read_pinhole_camera_trajectory(
                    "../../test_data/camera_trajectory.json")
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    if not os.path.exists("../../test_data/image/"):
        os.makedirs("../../test_data/image/")
    if not os.path.exists("../../test_data/depth/"):
        os.makedirs("../../test_data/depth/")

    def move_forward(vis):
        # This function is called within the o3d.visualization.Visualizer::run() loop
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
            plt.imsave("../../test_data/depth/{:05d}.png".format(glb.index),\
                    np.asarray(depth), dpi = 1)
            plt.imsave("../../test_data/image/{:05d}.png".format(glb.index),\
                    np.asarray(image), dpi = 1)
            #vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            #vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index])
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                    register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json("../../test_data/renderoption.json")
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("../../test_data/fragment.ply")

    print(
        "Customized visualization playing a camera trajectory. Ctrl+z to terminate"
    )
    custom_draw_geometry_with_camera_trajectory(pcd)
