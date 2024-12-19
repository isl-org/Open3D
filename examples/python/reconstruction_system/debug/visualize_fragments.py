# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/reconstruction_system/debug/visualize_fragments.py

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
