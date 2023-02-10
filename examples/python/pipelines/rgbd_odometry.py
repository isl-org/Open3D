# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Find camera movement between two consecutive RGBD image pairs"""

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    pinhole_camera_intrinsic = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    rgbd_data = o3d.data.SampleRedwoodRGBDImages()
    source_color = o3d.io.read_image(rgbd_data.color_paths[0])
    source_depth = o3d.io.read_image(rgbd_data.depth_paths[0])
    target_color = o3d.io.read_image(rgbd_data.color_paths[1])
    target_depth = o3d.io.read_image(rgbd_data.depth_paths[1])

    source_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    target_rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        target_color, target_depth)
    target_pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, pinhole_camera_intrinsic)

    option = o3d.pipelines.odometry.OdometryOption()
    odo_init = np.identity(4)
    print(option)

    [success_color_term, trans_color_term,
     info] = o3d.pipelines.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image,
         pinhole_camera_intrinsic, odo_init,
         o3d.pipelines.odometry.RGBDOdometryJacobianFromColorTerm(), option)
    [success_hybrid_term, trans_hybrid_term,
     info] = o3d.pipelines.odometry.compute_rgbd_odometry(
         source_rgbd_image, target_rgbd_image,
         pinhole_camera_intrinsic, odo_init,
         o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)

    if success_color_term:
        print("Using RGB-D Odometry")
        print(trans_color_term)
        source_pcd_color_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_color_term.transform(trans_color_term)
        o3d.visualization.draw([target_pcd, source_pcd_color_term])

    if success_hybrid_term:
        print("Using Hybrid RGB-D Odometry")
        print(trans_hybrid_term)
        source_pcd_hybrid_term = o3d.geometry.PointCloud.create_from_rgbd_image(
            source_rgbd_image, pinhole_camera_intrinsic)
        source_pcd_hybrid_term.transform(trans_hybrid_term)
        o3d.visualization.draw([target_pcd, source_pcd_hybrid_term])
