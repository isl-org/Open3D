# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import argparse
import os
import sys
sys.path.append("..")
sys.path.append("../../Utility")
import json
from open3d import *
from common import *
from register_fragments import *


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mathching two point clouds")
    parser.add_argument("config", help="reading json file for initial pose")
    parser.add_argument("source_id", type=int, help="ID of source point cloud")
    parser.add_argument("target_id", type=int, help="ID of target point cloud")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        config['debug_mode'] = True
        ply_file_names = get_file_list(
                os.path.join(config["path_dataset"], folder_fragment), ".ply")
        register_point_cloud_pair(ply_file_names,
                args.source_id, args.target_id, config)
