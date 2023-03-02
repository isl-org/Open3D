# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d
import open3d.core as o3c

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


def preprocess_point_cloud(pcd, voxel_size, tensor_fpfh=False):
    pcd_down = pcd.voxel_down_sample(voxel_size)
    pcd_down.estimate_normals(max_nn=30, radius=voxel_size * 2.0)
    pcd_fpfh = o3d.t.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        max_nn=100,
        radius=voxel_size * 5.0,
    )
    return (pcd_down, pcd_fpfh)


if __name__ == "__main__":
    pcd_data = o3d.data.DemoICPPointClouds()
    parser = argparse.ArgumentParser(
        "Global point cloud registration example with RANSAC"
    )
    parser.add_argument(
        "src",
        type=str,
        default=pcd_data.paths[0],
        nargs="?",
        help="path to src point cloud",
    )
    parser.add_argument(
        "dst",
        type=str,
        default=pcd_data.paths[1],
        nargs="?",
        help="path to dst point cloud",
    )
    parser.add_argument(
        "--voxel_size",
        type=float,
        default=0.05,
        help="voxel size in meter used to downsample inputs",
    )
    parser.add_argument(
        "--distance_multiplier",
        type=float,
        default=1.5,
        help="multipler used to compute distance threshold"
        "between correspondences."
        "Threshold is computed by voxel_size * distance_multiplier.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        default=1000000,
        help="number of max RANSAC iterations",
    )
    parser.add_argument(
        "--confidence", type=float, default=0.999, help="RANSAC confidence"
    )
    parser.add_argument(
        "--mutual_filter",
        action="store_true",
        help="whether to use mutual filter for putative correspondences",
    )

    args = parser.parse_args()

    voxel_size = args.voxel_size
    distance_threshold = args.distance_multiplier * voxel_size

    # o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    print("Reading inputs")
    tsrc = o3d.t.io.read_point_cloud(args.src).cuda()
    tdst = o3d.t.io.read_point_cloud(args.dst).cuda()

    print("Downsampling inputs")
    tsrc_down, tsrc_fpfh = preprocess_point_cloud(tsrc, voxel_size)
    tdst_down, tdst_fpfh = preprocess_point_cloud(tdst, voxel_size)

    print("Running RANSAC from correspondences")
    # Mimic importing customized external features (e.g. learned FCGF features) in numpy
    # shape: (feature_dim, num_features)
    src_down = tsrc_down.to_legacy()
    dst_down = tdst_down.to_legacy()
    src_fpfh_np = np.asarray(tsrc_fpfh.cpu().numpy().T).copy()
    dst_fpfh_np = np.asarray(tdst_fpfh.cpu().numpy().T).copy()

    print(src_fpfh_np)

    src_fpfh_import = o3d.pipelines.registration.Feature()
    src_fpfh_import.data = src_fpfh_np

    dst_fpfh_import = o3d.pipelines.registration.Feature()
    dst_fpfh_import.data = dst_fpfh_np

    # Legacy CPU
    corres_legacy = o3d.pipelines.registration.correspondences_from_features(
        src_fpfh_import, dst_fpfh_import, args.mutual_filter
    )

    # Tensor CPU
    src_fpfh_cpu = o3c.Tensor(src_fpfh_np.T).contiguous()
    dst_fpfh_cpu = o3c.Tensor(dst_fpfh_np.T).contiguous()
    corres_tensor_cpu = o3d.t.pipelines.registration.correspondences_from_features(
        src_fpfh_cpu, dst_fpfh_cpu
    )

    src_fpfh_cuda = src_fpfh_cpu.cuda()
    dst_fpfh_cuda = dst_fpfh_cpu.cuda()
    corres_tensor_cuda = o3d.t.pipelines.registration.correspondences_from_features(
        src_fpfh_cuda, dst_fpfh_cuda
    )

    for corres in [
        corres_legacy,
        o3d.utility.Vector2iVector(corres_tensor_cpu.numpy()),
        o3d.utility.Vector2iVector(corres_tensor_cuda.cpu().numpy()),
    ]:
        print(np.asarray(corres))
        result = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            src_down,
            dst_down,
            corres,
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
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                args.max_iterations, args.confidence
            ),
        )
        print(result)
        visualize_registration(src_down, dst_down, result.transformation)
