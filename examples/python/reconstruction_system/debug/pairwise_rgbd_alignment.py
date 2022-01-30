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

# examples/python/reconstruction_system/debug/pairwise_rgbd_alignment.py

import argparse
import json
import sys
import open3d as o3d

pyexample_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(pyexample_path)
from open3d_example import *

sys.path.append(".")
from initialize_config import *
from make_fragments import *


def test_single_pair(s, t, color_files, depth_files, intrinsic, with_opencv,
                     config):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
    [success, trans,
     info] = register_one_rgbd_pair(s, t, color_files, depth_files, intrinsic,
                                    with_opencv, config)
    print(trans)
    print(info)
    print(intrinsic)
    source_rgbd_image = read_rgbd_image(color_files[s], depth_files[s], False,
                                        config)
    target_rgbd_image = read_rgbd_image(color_files[t], depth_files[t], False,
                                        config)
    source = o3d.geometry.PointCloud.create_from_rgbd_image(
        source_rgbd_image, intrinsic)
    target = o3d.geometry.PointCloud.create_from_rgbd_image(
        target_rgbd_image, intrinsic)
    draw_geometries_flip([source, target])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mathching two RGBD images")
    parser.add_argument("config", help="path to the config file")
    parser.add_argument("source_id", type=int, help="ID of source RGBD image")
    parser.add_argument("target_id", type=int, help="ID of target RGBD image")
    parser.add_argument("--path_intrinsic",
                        help="path to the RGBD camera intrinsic")
    args = parser.parse_args()

    with open(args.config) as json_file:
        config = json.load(json_file)
        initialize_config(config)

        with_opencv = initialize_opencv()
        if with_opencv:
            from opencv_pose_estimation import pose_estimation

        [color_files, depth_files] = get_rgbd_file_lists(config["path_dataset"])
        if args.path_intrinsic:
            intrinsic = o3d.io.read_pinhole_camera_intrinsic(
                args.path_intrinsic)
        else:
            intrinsic = o3d.camera.PinholeCameraIntrinsic(
                o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
        test_single_pair(args.source_id, args.target_id, color_files,
                         depth_files, intrinsic, with_opencv, config)
