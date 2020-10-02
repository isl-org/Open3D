# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/python/reconstruction_system/debug/pairwise_pc_alignment.py

import argparse
import json
import sys
sys.path.append("../utility")
from file import *
sys.path.append(".")
from initialize_config import *
from register_fragments import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mathching two point clouds")
    parser.add_argument("config", help="reading json file for initial pose")
    parser.add_argument("source_id", type=int, help="ID of source point cloud")
    parser.add_argument("target_id", type=int, help="ID of target point cloud")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)
        config['debug_mode'] = True
        ply_file_names = get_file_list(
            join(config["path_dataset"], config["folder_fragment"]), ".ply")
        register_point_cloud_pair(ply_file_names, args.source_id,
                                  args.target_id, config)
