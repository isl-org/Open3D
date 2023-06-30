# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

from pathlib import Path


def test_get_set_view_control():
    ply_pointcloud = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_pointcloud.path)

    width = 480
    height = 360
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=width, height=height, visible=False)
    vis.add_geometry(pcd)

    # Set view control with hard-coded intrinsic and extrinsics.
    new_width = 100
    new_height = 100
    new_intrinsic = np.array([
        [10, 0, 50],
        [0, 10, 50],
        [0, 0, 1],
    ])
    new_extrinsic = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ])
    o3d_camera = o3d.camera.PinholeCameraParameters()
    o3d_camera.intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=new_width,
        height=new_height,
        fx=new_intrinsic[0, 0],
        fy=new_intrinsic[1, 1],
        cx=new_intrinsic[0, 2],
        cy=new_intrinsic[1, 2],
    )
    o3d_camera.extrinsic = new_extrinsic
    ctr = vis.get_view_control()
    ctr.convert_from_pinhole_camera_parameters(o3d_camera, allow_arbitrary=True)

    # Get view control and verify correctness.
    vis.update_geometry(pcd)
    vis.poll_events()
    vis.update_renderer()
    ctr = vis.get_view_control()
    o3d_camera = ctr.convert_to_pinhole_camera_parameters()
    np.testing.assert_allclose(o3d_camera.intrinsic.intrinsic_matrix,
                               new_intrinsic)
    np.testing.assert_allclose(o3d_camera.extrinsic, new_extrinsic)
