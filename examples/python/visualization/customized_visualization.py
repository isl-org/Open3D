# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
test_data_path = os.path.join(os.path.dirname(pyexample_path), 'test_data')


def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # o3d.visualization.draw_geometries([pcd])
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_custom_fov(pcd, fov_step):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    ctr = vis.get_view_control()
    print("Field of view (before changing) %.2f" % ctr.get_field_of_view())
    ctr.change_field_of_view(step=fov_step)
    print("Field of view (after changing) %.2f" % ctr.get_field_of_view())
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_rotation(pcd):

    def rotate_view(vis):
        ctr = vis.get_view_control()
        ctr.rotate(10.0, 0.0)
        return False

    o3d.visualization.draw_geometries_with_animation_callback([pcd],
                                                              rotate_view)


def custom_draw_geometry_load_option(pcd, render_option_path):
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.run()
    vis.destroy_window()


def custom_draw_geometry_with_key_callback(pcd, render_option_path):

    def change_background_to_black(vis):
        opt = vis.get_render_option()
        opt.background_color = np.asarray([0, 0, 0])
        return False

    def load_render_option(vis):
        vis.get_render_option().load_from_json(render_option_path)
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
    o3d.visualization.draw_geometries_with_key_callbacks([pcd], key_to_callback)


def custom_draw_geometry_with_camera_trajectory(pcd, render_option_path,
                                                camera_trajectory_path):
    custom_draw_geometry_with_camera_trajectory.index = -1
    custom_draw_geometry_with_camera_trajectory.trajectory =\
        o3d.io.read_pinhole_camera_trajectory(camera_trajectory_path)
    custom_draw_geometry_with_camera_trajectory.vis = o3d.visualization.Visualizer(
    )
    image_path = os.path.join(test_data_path, 'image')
    if not os.path.exists(image_path):
        os.makedirs(image_path)
    depth_path = os.path.join(test_data_path, 'depth')
    if not os.path.exists(depth_path):
        os.makedirs(depth_path)

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
            plt.imsave(os.path.join(depth_path, '{:05d}.png'.format(glb.index)),
                       np.asarray(depth),
                       dpi=1)
            plt.imsave(os.path.join(image_path, '{:05d}.png'.format(glb.index)),
                       np.asarray(image),
                       dpi=1)
            # vis.capture_depth_image("depth/{:05d}.png".format(glb.index), False)
            # vis.capture_screen_image("image/{:05d}.png".format(glb.index), False)
        glb.index = glb.index + 1
        if glb.index < len(glb.trajectory.parameters):
            ctr.convert_from_pinhole_camera_parameters(
                glb.trajectory.parameters[glb.index], allow_arbitrary=True)
        else:
            custom_draw_geometry_with_camera_trajectory.vis.\
                register_animation_callback(None)
        return False

    vis = custom_draw_geometry_with_camera_trajectory.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.get_render_option().load_from_json(render_option_path)
    vis.register_animation_callback(move_forward)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    sample_data = o3d.data.DemoCustomVisualization()
    pcd_flipped = o3d.io.read_point_cloud(sample_data.point_cloud_path)
    # Flip it, otherwise the pointcloud will be upside down
    pcd_flipped.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0],
                           [0, 0, 0, 1]])

    print("1. Customized visualization to mimic DrawGeometry")
    custom_draw_geometry(pcd_flipped)

    print("2. Changing field of view")
    custom_draw_geometry_with_custom_fov(pcd_flipped, 90.0)
    custom_draw_geometry_with_custom_fov(pcd_flipped, -90.0)

    print("3. Customized visualization with a rotating view")
    custom_draw_geometry_with_rotation(pcd_flipped)

    print("4. Customized visualization showing normal rendering")
    custom_draw_geometry_load_option(pcd_flipped,
                                     sample_data.render_option_path)

    print("5. Customized visualization with key press callbacks")
    print("   Press 'K' to change background color to black")
    print("   Press 'R' to load a customized render option, showing normals")
    print("   Press ',' to capture the depth buffer and show it")
    print("   Press '.' to capture the screen and show it")
    custom_draw_geometry_with_key_callback(pcd_flipped,
                                           sample_data.render_option_path)

    pcd = o3d.io.read_point_cloud(sample_data.point_cloud_path)
    print("6. Customized visualization playing a camera trajectory")
    custom_draw_geometry_with_camera_trajectory(
        pcd, sample_data.render_option_path, sample_data.camera_trajectory_path)
