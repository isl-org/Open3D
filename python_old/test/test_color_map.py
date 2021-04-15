import open3d as o3d
import numpy as np
import re
import os
import sys
from open3d_test import download_fountain_dataset
import time


def load_fountain_dataset():

    def get_file_list(path, extension=None):

        def sorted_alphanum(file_list_ordered):
            convert = lambda text: int(text) if text.isdigit() else text
            alphanum_key = lambda key: [
                convert(c) for c in re.split('([0-9]+)', key)
            ]
            return sorted(file_list_ordered, key=alphanum_key)

        if extension is None:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f))
            ]
        else:
            file_list = [
                path + f
                for f in os.listdir(path)
                if os.path.isfile(os.path.join(path, f)) and
                os.path.splitext(f)[1] == extension
            ]
        file_list = sorted_alphanum(file_list)
        return file_list

    path = download_fountain_dataset()
    depth_image_path = get_file_list(os.path.join(path, "depth/"),
                                     extension=".png")
    color_image_path = get_file_list(os.path.join(path, "image/"),
                                     extension=".jpg")
    assert (len(depth_image_path) == len(color_image_path))

    rgbd_images = []
    for i in range(len(depth_image_path)):
        depth = o3d.io.read_image(os.path.join(depth_image_path[i]))
        color = o3d.io.read_image(os.path.join(color_image_path[i]))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, convert_rgb_to_intensity=False)
        rgbd_images.append(rgbd_image)

    camera_trajectory = o3d.io.read_pinhole_camera_trajectory(
        os.path.join(path, "scene/key.log"))
    mesh = o3d.io.read_triangle_mesh(
        os.path.join(path, "scene", "integrated.ply"))

    return mesh, rgbd_images, camera_trajectory


def test_color_map():
    # Load dataset
    mesh, rgbd_images, camera_trajectory = load_fountain_dataset()

    # Computes averaged color without optimization, for debugging
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory,
            o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=0))

    # Rigid Optimization
    start = time.time()
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_optimized = o3d.pipelines.color_map.run_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory,
            o3d.pipelines.color_map.RigidOptimizerOption(maximum_iteration=5))
    print(f"Rigid optimization takes {time.time() - start}")

    # Non-rigid Optimization
    start = time.time()
    with o3d.utility.VerbosityContextManager(
            o3d.utility.VerbosityLevel.Debug) as cm:
        mesh_optimized = o3d.pipelines.color_map.run_non_rigid_optimizer(
            mesh, rgbd_images, camera_trajectory,
            o3d.pipelines.color_map.NonRigidOptimizerOption(
                maximum_iteration=5))
    print(f"Non-rigid optimization takes {time.time() - start}")

    # Black box test with hard-coded result values. The results of
    # color_map_optimization are deterministic. This test ensures the refactored
    # code produces the same output. This is only valid for using exactly the
    # same inputs and optimization options.
    vertex_colors = np.asarray(mesh_optimized.vertex_colors)
    assert vertex_colors.shape == (536872, 3)

    # We need to account for the acceptable variation in the least significant
    # bit which can occur with different JPEG libraries. The test value is
    # pretty much exact with libjpeg-turbo, but not with the original libjpeg.
    np.testing.assert_allclose(np.mean(vertex_colors, axis=0),
                               [0.40307181, 0.37264626, 0.5436129],
                               rtol=1. / 256.)
