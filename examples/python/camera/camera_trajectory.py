# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import numpy as np
import open3d as o3d

if __name__ == "__main__":

    print("Testing camera in open3d ...")
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    print(intrinsic.intrinsic_matrix)
    print(o3d.camera.PinholeCameraIntrinsic())
    x = o3d.camera.PinholeCameraIntrinsic(640, 480, 525, 525, 320, 240)
    print(x)
    print(x.intrinsic_matrix)
    o3d.io.write_pinhole_camera_intrinsic("test.json", x)
    y = o3d.io.read_pinhole_camera_intrinsic("test.json")
    print(y)
    print(np.asarray(y.intrinsic_matrix))

    print("Read a trajectory and combine all the RGB-D images.")
    pcds = []
    redwood_rgbd = o3d.data.SampleRedwoodRGBDImages()
    trajectory = o3d.io.read_pinhole_camera_trajectory(
        redwood_rgbd.trajectory_log_path)
    o3d.io.write_pinhole_camera_trajectory("test.json", trajectory)
    print(trajectory)
    print(trajectory.parameters[0].extrinsic)
    print(np.asarray(trajectory.parameters[0].extrinsic))
    for i in range(5):
        im1 = o3d.io.read_image(redwood_rgbd.depth_paths[i])
        im2 = o3d.io.read_image(redwood_rgbd.color_paths[i])
        im = o3d.geometry.RGBDImage.create_from_color_and_depth(
            im2, im1, 1000.0, 5.0, False)
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            im, trajectory.parameters[i].intrinsic,
            trajectory.parameters[i].extrinsic)
        pcds.append(pcd)
    o3d.visualization.draw_geometries(pcds)
