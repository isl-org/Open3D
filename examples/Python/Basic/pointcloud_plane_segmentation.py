# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Basic/pointcloud_plane_segmentation.py

import numpy as np
import open3d as o3d

if __name__ == "__main__":
    pcd = o3d.io.read_point_cloud("../../TestData/fragment.pcd")

    print(
        "Find the plane model and the inliers of the largest planar segment in the point cloud."
    )
    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                             ransac_n=3,
                                             num_iterations=250)

    [a, b, c, d] = plane_model
    print(f"Plane model: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    inlier_cloud = pcd.select_by_index(inliers)
    inlier_cloud.paint_uniform_color([1.0, 0, 0])

    outlier_cloud = pcd.select_by_index(inliers, invert=True)

    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud])
