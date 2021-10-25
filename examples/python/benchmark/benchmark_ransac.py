# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------

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

if __name__ == "__main__":
    # data preparation
    redwood = o3d.data.dataset.Redwood()
    voxel_size = 0.05

    # do RANSAC based alignment
    for name, ply_file_list in zip(redwood.names, redwood.ply_paths):
        n_ply_files = len(ply_file_list)
        alignment = []
        for s in range(n_ply_files):
            for t in range(s + 1, n_ply_files):

                print("%s:: matching %d-%d" % (name, s, t))
                source = o3d.io.read_point_cloud(ply_file_list[s])
                target = o3d.io.read_point_cloud(ply_file_list[t])
                source_down, source_fpfh = preprocess_point_cloud(
                    source, voxel_size)
                target_down, target_fpfh = preprocess_point_cloud(
                    target, voxel_size)

                result = execute_fast_global_registration(
                    source_down, target_down, source_fpfh, target_fpfh,
                    voxel_size)
                if (result.transformation.trace() == 4.0):
                    success = False
                else:
                    success = True

                # Note: we save inverse of result_ransac.transformation
                # to comply with http://redwood-data.org/indoor/fileformat.html
                alignment.append(
                    CameraPose([s, t, n_ply_files],
                               np.linalg.inv(result.transformation)))
                print(np.linalg.inv(result.transformation))

                if do_visualization:
                    draw_registration_result(source_down, target_down,
                                             result.transformation)
        write_trajectory(alignment, redwood.data_root+"/ransac_"+name+".log")

    # do evaluation