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

import open3d as o3d


def lounge_data_loader():
    print('Loading Stanford Lounge RGB-D Dataset')

    # Get the dataset.
    lounge_rgbd = o3d.data.LoungeRGBDImages()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = lounge_rgbd.extract_dir
    config['path_intrinsic'] = ""
    config['depth_max'] = 3.0
    config['voxel_size'] = 0.05
    config['depth_diff_max'] = 0.07
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 3.0
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def bedroom_data_loader():
    print('Loading Redwood Bedroom RGB-D Dataset')

    # Get the dataset.
    bedroom_rgbd = o3d.data.BedroomRGBDImages()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = bedroom_rgbd.extract_dir
    config['path_intrinsic'] = ""
    config['depth_max'] = 3.0
    config['voxel_size'] = 0.05
    config['depth_diff_max'] = 0.07
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 3.0
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config


def jackjack_data_loader():
    print('Loading RealSense L515 Jack-Jack RGB-D Bag Dataset')

    # Get the dataset.
    jackjack_bag = o3d.data.JackJackL515Bag()

    # Set dataset specific parameters.
    config = {}
    config['path_dataset'] = jackjack_bag.path
    config['path_intrinsic'] = ""
    config['depth_max'] = 0.85
    config['voxel_size'] = 0.025
    config['depth_diff_max'] = 0.03
    config['preference_loop_closure_odometry'] = 0.1
    config['preference_loop_closure_registration'] = 5.0
    config['tsdf_cubic_size'] = 0.75
    config['icp_method'] = "color"
    config['global_registration'] = "ransac"
    config['python_multi_threading'] = True

    return config
