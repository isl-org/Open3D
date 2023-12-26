# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

if __name__ == "__main__":
    pcd_data = o3d.data.PCDPointCloud()
    print(
        f"Reading pointcloud from file: fragment.pcd stored at {pcd_data.path}")
    pcd = o3d.io.read_point_cloud(pcd_data.path)
    print(pcd)
    print("Saving pointcloud to file: copy_of_fragment.pcd")
    o3d.io.write_point_cloud("copy_of_fragment.pcd", pcd)
