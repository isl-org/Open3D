# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import numpy as np

if __name__ == "__main__":
    sample_ply_data = o3d.data.DemoCropPointCloud()
    pcd = o3d.io.read_point_cloud(sample_ply_data.point_cloud_path)
    vol = o3d.visualization.read_selection_polygon_volume(
        sample_ply_data.cropped_json_path)
    chair = vol.crop_point_cloud(pcd)

    chair.paint_uniform_color([0, 0, 1])
    pcd.paint_uniform_color([1, 0, 0])
    print("Displaying the two point clouds used for calculating distance ...")
    o3d.visualization.draw([pcd, chair])

    dists = pcd.compute_point_cloud_distance(chair)
    dists = np.asarray(dists)
    print("Printing average distance between the two point clouds ...")
    print(dists)
