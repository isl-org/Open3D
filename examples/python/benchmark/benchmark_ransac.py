# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/benchmark/benchmark_ransac.py

import os
import sys
sys.path.append("../pipelines")
sys.path.append("../geometry")
sys.path.append("../utility")
import numpy as np
from file import *
from visualization import *
from downloader import *
from fast_global_registration import *
from trajectory_io import *

do_visualization = False


def get_ply_path(dataset_name, id):
    return "%s/%s/cloud_bin_%d.ply" % (dataset_path, dataset_name, id)


def get_log_path(dataset_name):
    return "%s/ransac_%s.log" % (dataset_path, dataset_name)


if __name__ == "__main__":
    # data preparation
    get_redwood_dataset()
    voxel_size = 0.05

    # do RANSAC based alignment
    for dataset_name in dataset_names:
        ply_file_names = get_file_list("%s/%s/" % (dataset_path, dataset_name),
                                       ".ply")
        n_ply_files = len(ply_file_names)

        alignment = []
        for s in range(n_ply_files):
            for t in range(s + 1, n_ply_files):

                print("%s:: matching %d-%d" % (dataset_name, s, t))
                source = o3d.io.read_point_cloud(get_ply_path(dataset_name, s))
                target = o3d.io.read_point_cloud(get_ply_path(dataset_name, t))
                source_down, source_fpfh = preprocess_point_cloud(
                    source, voxel_size)
                target_down, target_fpfh = preprocess_point_cloud(
                    target, voxel_size)

                result = execute_global_registration(source_down, target_down,
                                                     source_fpfh, target_fpfh,
                                                     voxel_size)
                if (result.transformation.trace() == 4.0):
                    success = False
                else:
                    success = True

                # Note: we save inverse of result.transformation
                # to comply with http://redwood-data.org/indoor/fileformat.html
                if not success:
                    print("No reasonable solution.")
                else:
                    alignment.append(
                        CameraPose([s, t, n_ply_files],
                                   np.linalg.inv(result.transformation)))
                    print(np.linalg.inv(result.transformation))

                if do_visualization:
                    draw_registration_result(source_down, target_down,
                                             result.transformation)
        write_trajectory(alignment, get_log_path(dataset_name))

    # do evaluation
