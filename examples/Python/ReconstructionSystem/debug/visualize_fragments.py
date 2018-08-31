# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import argparse
import json
import os
import sys
sys.path.append("../../Utility")
from open3d import *
from common import *


# test wide baseline matching
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize pose graph")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--start_id", type=int, default=0,
            help="starting ID of fragment")
    args = parser.parse_args()
    
    with open(args.config) as json_file:
        config = json.load(json_file)
        fragment_files = get_file_list(
                os.path.join(config["path_dataset"], folder_fragment),
                extension='.ply')
        for i in range(args.start_id, len(fragment_files)):
            print(fragment_files[i])
            mesh = read_triangle_mesh(fragment_files[i])
            mesh.compute_vertex_normals()
            draw_geometries_flip([mesh])
