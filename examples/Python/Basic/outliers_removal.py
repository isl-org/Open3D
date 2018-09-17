from open3d import *
import numpy as np


def display_inlier_outlier(cloud,ind):
    inlier_cloud = select_down_sample(cloud,ind)

    ind_inverted = np.arange(len(cloud.points))
    ind_inverted[ind] = -1
    ind_inverted = ind_inverted[ind_inverted > 0]
    outlier_cloud = select_down_sample(cloud,ind_inverted)

    outlier_cloud.paint_uniform_color([0.1, 0.9, 0.1])
    inlier_cloud.paint_uniform_color([0.9, 0.1, 0.1])
    print("Showing outliers: ")
    draw_geometries([outlier_cloud])

    print("Showing inliers: ")
    draw_geometries([inlier_cloud])

    print("Showing both sets together: ")
    draw_geometries([inlier_cloud,outlier_cloud])

pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")

print("------ Statistical Oulier Removal -------")
cl,ind = statistical_outlier_removal(pcd,50,2)
display_inlier_outlier(pcd,ind)

print("------ Radius Oulier Removal -------")
cl,ind = radius_outlier_removal(pcd,60,0.05)
display_inlier_outlier(pcd,ind)