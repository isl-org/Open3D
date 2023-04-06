# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d

import numpy as np
from copy import deepcopy
import argparse


def visualize_registration(src, dst, transformation=np.eye(4)):
    src_trans = deepcopy(src)
    src_trans.transform(transformation)
    src_trans.paint_uniform_color([1, 0, 0])

    dst_clone = deepcopy(dst)
    dst_clone.paint_uniform_color([0, 1, 0])

    o3d.visualization.draw([src_trans, dst_clone])


def preprocess_point_cloud(pcd, voxel_size):
    # TODO(wei): another PR to do voxel down sample
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down = o3d.t.geometry.PointCloud.from_legacy(pcd_down).cuda()
    pcd_down.estimate_normals(radius=voxel_size * 2.0)
    pcd_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        max_nn=100,
        radius=voxel_size * 5.0,
    )
    return (pcd_down, pcd_fpfh)


def pcd_odometry(src, dst, voxel_size, init_transformation):
    # TODO(wei): add multi-scale icp
    result = o3d.t.pipelines.registration.icp(
        src,
        dst,
        voxel_size * 1.5,
        init_transformation,
        estimation_method=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
    )
    return result.transformation.numpy()


def pcd_global_registration(src, src_fpfh, dst, dst_fpfh, voxel_size):
    distance_threshold = voxel_size * 1.5

    corres = o3d.t.pipelines.registration.correspondences_from_features(
        src_fpfh, dst_fpfh, True
    )

    result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
        src.to_legacy(),
        dst.to_legacy(),
        o3d.utility.Vector2iVector(corres.cpu().numpy()),
        max_correspondence_distance=distance_threshold,
        estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
            False
        ),
        ransac_n=3,
        checkers=[
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold
            ),
        ],
        criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(1000000, 0.9999),
    )

    return result.transformation


if __name__ == "__main__":
    pcd_data = o3d.data.DemoICPPointClouds()

    # yapf: disable
    parser = argparse.ArgumentParser(
        "Global point cloud registration example with RANSAC"
    )
    parser.add_argument(
        "src", type=str, default=pcd_data.paths[0], nargs="?",
        help="path to src point cloud",
    )
    parser.add_argument(
        "dst", type=str, default=pcd_data.paths[1], nargs="?",
        help="path to dst point cloud",
    )
    parser.add_argument(
        "--voxel_size", type=float, default=0.05,
        help="voxel size in meter used to downsample inputs",
    )
    parser.add_argument(
        "--distance_multiplier", type=float, default=1.5,
        help="multipler used to compute distance threshold"
        "between correspondences."
        "Threshold is computed by voxel_size * distance_multiplier.",
    )
    parser.add_argument(
        "--max_iterations", type=int, default=100000,
        help="number of max RANSAC iterations",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.999, help="RANSAC confidence"
    )
    parser.add_argument(
        "--mutual_filter", action="store_true",
        help="whether to use mutual filter for putative correspondences",
    )
    parser.add_argument(
        "--method", choices=["from_features", "from_correspondences"], default="from_correspondences"
    )
    # yapf: enable

    args = parser.parse_args()

    voxel_size = args.voxel_size
    distance_threshold = args.distance_multiplier * voxel_size
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)

    print("Reading inputs")
    src = o3d.io.read_point_cloud(args.src)
    dst = o3d.io.read_point_cloud(args.dst)

    print("Downsampling inputs")
    src_down, src_fpfh = preprocess_point_cloud(src, voxel_size)
    dst_down, dst_fpfh = preprocess_point_cloud(dst, voxel_size)

    src2dst = pcd_global_registration(
        src_down, src_fpfh, dst_down, dst_fpfh, voxel_size
    )

    visualize_registration(src, dst, src2dst)

    src2dst = pcd_odometry(src_down, dst_down, voxel_size, src2dst)
    visualize_registration(src, dst, src2dst)
