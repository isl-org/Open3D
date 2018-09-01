# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Advanced/color_map_optimization.py

from open3d import *
from trajectory_io import *
import os, sys
sys.path.append("../Utility")
from common import *

path = "[path_to_fountain_dataset]"
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
                    PinholeCameraIntrinsic(PinholeCameraIntrinsicParameters.PrimeSenseDefault))
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
    option.maximum_iteration = 300
    option.non_rigid_camera_coordinate = True
    color_map_optimization(mesh, rgbd_images, camera, option)
    draw_geometries([mesh])
    write_triangle_mesh(os.path.join(path, "scene",
        "color_map_after_optimization.ply"), mesh)
