# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

import os, sys

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)

from open3d_example import read_trajectory

if __name__ == "__main__":
    rgbd_data = o3d.data.SampleRedwoodRGBDImages()
    camera_poses = read_trajectory(rgbd_data.odometry_log_path)
    camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    volume = o3d.pipelines.integration.UniformTSDFVolume(
        length=4.0,
        resolution=512,
        sdf_trunc=0.04,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for i in range(len(camera_poses)):
        print("Integrate {:d}-th image into the volume.".format(i))
        color = o3d.io.read_image(rgbd_data.color_paths[i])
        depth = o3d.io.read_image(rgbd_data.depth_paths[i])

        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color, depth, depth_trunc=4.0, convert_rgb_to_intensity=False)
        volume.integrate(
            rgbd,
            camera_intrinsics,
            np.linalg.inv(camera_poses[i].pose),
        )

    print("Extract triangle mesh")
    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    o3d.visualization.draw_geometries([mesh])

    print("Extract voxel-aligned debugging point cloud")
    voxel_pcd = volume.extract_voxel_point_cloud()
    o3d.visualization.draw_geometries([voxel_pcd])

    print("Extract voxel-aligned debugging voxel grid")
    voxel_grid = volume.extract_voxel_grid()
    # o3d.visualization.draw_geometries([voxel_grid])

    # print("Extract point cloud")
    # pcd = volume.extract_point_cloud()
    # o3d.visualization.draw_geometries([pcd])
