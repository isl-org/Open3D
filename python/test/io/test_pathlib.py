# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

from pathlib import Path

import open3d as o3d


def test_pathlib_support():
    pcd_pointcloud = o3d.data.PCDPointCloud()
    assert isinstance(pcd_pointcloud.path, str)

    pcd = o3d.io.read_point_cloud(pcd_pointcloud.path)
    assert pcd.has_points()

    pcd = o3d.io.read_point_cloud(Path(pcd_pointcloud.path))
    assert pcd.has_points()