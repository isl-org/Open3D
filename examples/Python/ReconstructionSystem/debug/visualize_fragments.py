# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import argparse
import json
import os
import sys
from open3d import *
sys.path.append("../Utility")
from file import *
from visualization import *
sys.path.append(".")
from initialize_config import *


# test wide baseline matching
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize pose graph")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--start_id", type=int, default=0,
            help="starting ID of fragment")
    parser.add_argument("--estimate_normal", type=int, default=0,
            help="normal estimation for better visualization of point cloud")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)
        fragment_files = get_file_list(
                join(config["path_dataset"], config["folder_fragment"]),
                extension='.ply')
        for i in range(args.start_id, len(fragment_files)):
            print(fragment_files[i])
            pcd = read_point_cloud(fragment_files[i])
            if (args.estimate_normal):
                estimate_normals(pcd, KDTreeSearchParamHybrid(
                        radius = config["voxel_size"] * 2.0, max_nn = 30))
            draw_geometries_flip([pcd])
