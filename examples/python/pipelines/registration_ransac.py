# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
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
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 2.0,
                                             max_nn=30))
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=voxel_size * 5.0,
                                             max_nn=100),
    )
    return (pcd_down, pcd_fpfh)


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

    if args.method == "from_features":
        print("Running RANSAC from features")
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            src_down,
            dst_down,
            src_fpfh,
            dst_fpfh,
            mutual_filter=args.mutual_filter,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                args.max_iterations, args.confidence),
        )
        visualize_registration(src, dst, result.transformation)

    elif args.method == "from_correspondences":
        print("Running RANSAC from correspondences")
        # Mimic importing customized external features (e.g. learned FCGF features) in numpy
        # shape: (feature_dim, num_features)
        src_fpfh_np = np.asarray(src_fpfh.data).copy()
        dst_fpfh_np = np.asarray(dst_fpfh.data).copy()

        src_fpfh_import = o3d.pipelines.registration.Feature()
        src_fpfh_import.data = src_fpfh_np

        dst_fpfh_import = o3d.pipelines.registration.Feature()
        dst_fpfh_import.data = dst_fpfh_np

        corres = o3d.pipelines.registration.correspondences_from_features(
            src_fpfh_import, dst_fpfh_import, args.mutual_filter)
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src_down,
            dst_down,
            corres,
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.
            TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.
                CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                args.max_iterations, args.confidence),
        )
        visualize_registration(src, dst, result.transformation)
