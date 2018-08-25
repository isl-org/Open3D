# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import os

import numpy as np
from open3d import *

from trajectory_io import *

if __name__ == "__main__":
    data_dir = "../../TestData/RGBD"
    camera_poses = read_trajectory(os.path.join(data_dir, "odometry.log"))
    volume = ScalableTSDFVolume(voxel_length = 4.0 / 512.0,
            sdf_trunc = 0.04, color_type = TSDFVolumeColorType.RGB8)

    for i, camera_pose in enumerate(camera_poses):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = read_image(os.path.join(data_dir, "color/{:05d}.jpg".format(i)))
        depth = read_image(os.path.join(data_dir, "depth/{:05d}.png".format(i)))
        rgbd = create_rgbd_image_from_color_and_depth(color, depth,
                depth_trunc = 4.0, convert_rgb_to_intensity = False)
        volume.integrate(rgbd, PinholeCameraIntrinsic(
                PinholeCameraIntrinsicParameters.PrimeSenseDefault),
                np.linalg.inv(camera_pose.pose))

    print("Extract a triangle mesh from the volume and visualize it.")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    draw_geometries([mesh])
