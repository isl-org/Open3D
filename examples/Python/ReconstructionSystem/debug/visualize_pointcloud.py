# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import argparse
import json
from open3d import *
import sys
import os
import math
sys.path.append("../../Utility")
sys.path.append("../")
from common import *
from make_fragments import *

# test wide baseline matching
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="visualize fragment or scene as a point cloud form")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--path_intrinsic", help="path to the RGBD camera intrinsic")
    parser.add_argument("--fragment", help="visualize nodes in form of point clouds") # need to take fragment
    parser.add_argument("--scene", help="visualize nodes in form of point clouds", action="store_true")
    args = parser.parse_args()
    if not args.fragment and not args.scene:
        parser.print_help(sys.stderr)
        sys.exit(1)

    with open(args.config) as json_file:
        config = json.load(json_file)
        if (args.scene):
            pose_graph = read_pose_graph(
                    os.path.join(config["path_dataset"],
                    template_global_posegraph_optimized))
            ply_file_names = get_file_list(
                    os.path.join(config["path_dataset"],
                     folder_fragment), ".ply")
            pcds = []
            for i in range(len(pose_graph.nodes)):
                pcd = read_point_cloud(ply_file_names[i])
                pcd_down = voxel_down_sample(pcd, config["voxel_size"])
                pcd.transform(pose_graph.nodes[i].pose)
                print(np.linalg.inv(pose_graph.nodes[i].pose))
                pcds.append(pcd)
            draw_geometries_flip(pcds)

        if (args.fragment):
            if (args.path_intrinsic):
                pinhole_camera_intrinsic = read_pinhole_camera_intrinsic(
                        args.path_intrinsic)
            else:
                pinhole_camera_intrinsic = \
                        PinholeCameraIntrinsic(
                        PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            pcds = []
            [color_files, depth_files] = \
                    get_rgbd_file_lists(config["path_dataset"])
            n_files = len(color_files)
            n_fragments = int(math.ceil(float(n_files) / n_frames_per_fragment))
            sid = int(args.fragment) * n_frames_per_fragment
            eid = min(sid + n_frames_per_fragment, n_files)
            pose_graph = read_pose_graph(os.path.join(config["path_dataset"],
                    template_fragment_posegraph_optimized % int(args.fragment)))

            for i in range(sid, eid):
                print("appending rgbd image %d" % i)
                rgbd_image = read_rgbd_image(color_files[i], depth_files[i],
                        False, config)
                pcd_i = create_point_cloud_from_rgbd_image(rgbd_image,
                        pinhole_camera_intrinsic,
                        np.linalg.inv(pose_graph.nodes[i-sid].pose))
                pcd_i_down = voxel_down_sample(pcd_i, config["voxel_size"])
                pcds.append(pcd_i_down)
            draw_geometries_flip(pcds)
