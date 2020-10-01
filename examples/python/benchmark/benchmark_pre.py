# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/benchmark/benchmark_pre.py

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

import pickle

do_visualization = False


def get_ply_path(dataset_name, id):
    return "%s/%s/cloud_bin_%d.ply" % (dataset_path, dataset_name, id)


def get_log_path(dataset_name):
    return "%s/fgr_%s.log" % (dataset_path, dataset_name)


dataset_path = 'testdata'
dataset_names = ['livingroom1', 'livingroom2', 'office1', 'office2']

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
            source = o3d.io.read_point_cloud(get_ply_path(dataset_name, s))
            source_down, source_fpfh = preprocess_point_cloud(
                source, voxel_size)
            f = open('store.pckl', 'wb')
            pickle.dump([source_down, source_fpfh], f)
            f.close()
