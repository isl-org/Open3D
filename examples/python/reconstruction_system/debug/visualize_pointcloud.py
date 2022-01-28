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

# examples/python/reconstruction_system/debug/visualize_pointcloud.py

import argparse
import json
import math
import sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)
from open3d_example import *

sys.path.append(".")
from initialize_config import *
from make_fragments import *

# test wide baseline matching
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="visualize fragment or scene as a point cloud form")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("--path_intrinsic",
                        help="path to the RGBD camera intrinsic")
    parser.add_argument("--fragment",
                        help="visualize nodes in form of point clouds")
    parser.add_argument("--scene",
                        help="visualize nodes in form of point clouds",
                        action="store_true")
    parser.add_argument("--before_optimized",
                        help="visualize posegraph edges that is not optimized",
                        action="store_true")
    args = parser.parse_args()
    if not args.fragment and not args.scene:
        parser.print_help(sys.stderr)
        sys.exit(1)

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)
        if (args.scene):
            if (args.before_optimized):
                global_pose_graph_name = join(
                    config["path_dataset"],
                    config["template_refined_posegraph"])
            else:
                global_pose_graph_name = join(
                    config["path_dataset"],
                    config["template_refined_posegraph_optimized"])
            pose_graph = o3d.io.read_pose_graph(global_pose_graph_name)
            ply_file_names = get_file_list(
                join(config["path_dataset"], config["folder_fragment"]), ".ply")
            pcds = []
            for i in range(len(pose_graph.nodes)):
                pcd = o3d.io.read_point_cloud(ply_file_names[i])
                pcd_down = pcd.voxel_down_sample(config["voxel_size"] / 2.0)
                pcd_down.transform(pose_graph.nodes[i].pose)
                print(np.linalg.inv(pose_graph.nodes[i].pose))
                pcds.append(pcd_down)
            draw_geometries_flip(pcds)

        if (args.fragment):
            if (args.path_intrinsic):
                pinhole_camera_intrinsic = o3d.io.read_pinhole_camera_intrinsic(
                    args.path_intrinsic)
            else:
                pinhole_camera_intrinsic = \
                        o3d.camera.PinholeCameraIntrinsic(
                        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
            pcds = []
            [color_files, depth_files] = \
                    get_rgbd_file_lists(config["path_dataset"])
            n_files = len(color_files)
            n_fragments = int(math.ceil(float(n_files) / \
                    config['n_frames_per_fragment']))
            sid = int(args.fragment) * config['n_frames_per_fragment']
            eid = min(sid + config['n_frames_per_fragment'], n_files)
            pose_graph = o3d.io.read_pose_graph(join(config["path_dataset"],
                    config["template_fragment_posegraph_optimized"] % \
                    int(args.fragment)))

            for i in range(sid, eid):
                print("appending rgbd image %d" % i)
                rgbd_image = read_rgbd_image(color_files[i], depth_files[i],
                                             False, config)
                pcd_i = o3d.geometry.PointCloud.create_from_rgbd_image(
                    rgbd_image, pinhole_camera_intrinsic,
                    np.linalg.inv(pose_graph.nodes[i - sid].pose))
                pcd_i_down = pcd_i.voxel_down_sample(config["voxel_size"])
                pcds.append(pcd_i_down)
            draw_geometries_flip(pcds)
