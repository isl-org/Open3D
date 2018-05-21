# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import copy
from open3d import *


def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    draw_geometries([source_temp, target])


if __name__ == "__main__":

    set_verbosity_level(VerbosityLevel.Debug)

    print("1. Load two point clouds and show initial pose")
    source = read_point_cloud("/Users/jaesikpa/Downloads/testClouds/testclouda.ply")
    target = read_point_cloud("/Users/jaesikpa/Downloads/testClouds/testcloudb.ply")
    for i in range(len(source.points)):
        source.points[i] /= 30.0
    for i in range(len(target.points)):
        target.points[i] /= 30.0
    radius = 0.02
    estimate_normals(source, KDTreeSearchParamHybrid(
            radius = radius * 2, max_nn = 30))
    estimate_normals(target, KDTreeSearchParamHybrid(
            radius = radius * 2, max_nn = 30))
    # orient_normals_towards_camera_location(source)
    # orient_normals_towards_camera_location(target)

    # draw initial alignment
    draw_registration_result_original_color(
            source, target, np.identity(4))

    # point to plane ICP
    print("2. Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. Distance threshold 0.02.")
    result_icp = registration_icp(source, target, radius * 5,
            np.identity(4), TransformationEstimationPointToPoint())
    draw_registration_result_original_color(
            source, target, result_icp.transformation)
    result_icp = registration_icp(source, target, radius * 5,
            result_icp.transformation, TransformationEstimationPointToPlane())
    draw_registration_result_original_color(
            source, target, result_icp.transformation)

    # colored pointcloud registration
     # This is implementation of following paper
     # J. Park, Q.-Y. Zhou, V. Koltun,
     # Colored Point Cloud Registration Revisited, ICCV 2017
    voxel_radius = [ 0.04, 0.02, 0.01 ];
    max_iter = [ 50, 30, 14 ];
    current_transformation = result_icp.transformation
    print("3. Colored point cloud registration")
    for scale in range(3):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter,radius,scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        source_down = voxel_down_sample(source, radius)
        target_down = voxel_down_sample(target, radius)

        print("3-2. Estimate normal.")
        estimate_normals(source_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))
        estimate_normals(target_down, KDTreeSearchParamHybrid(
                radius = radius * 2, max_nn = 30))

        print("3-3. Applying colored point cloud registration")
        result_icp = registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = iter))
        current_transformation = result_icp.transformation
        print(result_icp)
    draw_registration_result_original_color(
            source_down, target_down, result_icp.transformation)
    draw_registration_result_original_color(
            source, target, result_icp.transformation)
