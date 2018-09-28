# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Advanced/outlier_removal.py

from open3d import *

def display_inlier_outlier(cloud, ind):
    inlier_cloud = select_down_sample(cloud, ind)
    outlier_cloud = select_down_sample(cloud, ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0.8, 0.8, 0.8])
    draw_geometries([inlier_cloud, outlier_cloud])


if __name__ == "__main__":

    print("Load a ply point cloud, print it, and render it")
    pcd = read_point_cloud("../../TestData/ICP/cloud_bin_2.pcd")
    draw_geometries([pcd])

    print("Downsample the point cloud with a voxel of 0.02")
    voxel_down_pcd = voxel_down_sample(pcd, voxel_size = 0.02)
    draw_geometries([voxel_down_pcd])

    print("Every 5th points are selected")
    uni_down_pcd = uniform_down_sample(pcd, every_k_points = 5)
    draw_geometries([uni_down_pcd])

    print("Statistical oulier removal")
    cl,ind = statistical_outlier_removal(voxel_down_pcd,
            nb_neighbors=20, std_ratio=2.0)
    display_inlier_outlier(voxel_down_pcd, ind)

    print("Radius oulier removal")
    cl,ind = radius_outlier_removal(voxel_down_pcd,
            nb_points=16, radius=0.05)
    display_inlier_outlier(voxel_down_pcd, ind)
