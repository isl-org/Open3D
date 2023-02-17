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

# examples/python/reconstruction_system/debug/visualize_fragment.py

import argparse
import json
import sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)
from open3d_example import *

sys.path.append(".")
from initialize_config import *

# test wide baseline matching
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize pose graph")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--start_id",
                        type=int,
                        default=0,
                        help="starting ID of fragment")
    parser.add_argument(
        "--estimate_normal",
        type=int,
        default=0,
        help="normal estimation for better visualization of point cloud")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)
        fragment_files = get_file_list(join(config["path_dataset"],
                                            config["folder_fragment"]),
                                       extension='.ply')
        for i in range(args.start_id, len(fragment_files)):
            print(fragment_files[i])
            pcd = o3d.io.read_point_cloud(fragment_files[i])
            if (args.estimate_normal):
                pcd.estimate_normals(
                    o3d.geometry.KDTreeSearchParamHybrid(
                        radius=config["voxel_size"] * 2.0, max_nn=30))
            draw_geometries_flip([pcd])
