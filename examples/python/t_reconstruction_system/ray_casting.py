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

# examples/python/t_reconstruction_system/ray_casting.py

# P.S. This example is used in documentation, so, please ensure the changes are
# synchronized.

import os
import numpy as np
import open3d as o3d
import open3d.core as o3c
import time
import matplotlib.pyplot as plt

from tqdm import tqdm
from config import ConfigParser
from common import load_depth_file_names, load_intrinsic, load_extrinsics, get_default_testdata
import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')

    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               required=True,
               help='path to the npz file that stores voxel block grid.')
    config = parser.get_config()

    if config.path_dataset == '':
        config.path_dataset = get_default_testdata()
        config.path_trajectory = os.path.join(config.path_dataset,
                                              'trajectory.log')

    vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
    depth_file_names = load_depth_file_names(config)
    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)
    device = o3d.core.Device(config.device)

    for i, extrinsic in tqdm(enumerate(extrinsics)):
        depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
        frustum_block_coords = vbg.compute_unique_block_coordinates(
            depth, intrinsic, extrinsic, config.depth_scale, config.depth_max)

        result = vbg.ray_cast(block_coords=frustum_block_coords,
                              intrinsic=intrinsic,
                              extrinsic=extrinsic,
                              width=depth.columns,
                              height=depth.rows,
                              render_attributes=[
                                  'depth', 'normal', 'color', 'index',
                                  'interp_ratio'
                              ],
                              depth_scale=config.depth_scale,
                              depth_min=config.depth_min,
                              depth_max=config.depth_max,
                              weight_threshold=1,
                              range_map_down_factor=8)

        fig, axs = plt.subplots(2, 2)

        # Colorized depth
        colorized_depth = o3d.t.geometry.Image(result['depth']).colorize_depth(
            config.depth_scale, config.depth_min, config.depth_max)

        # Render color via indexing
        vbg_color = vbg.attribute('color').reshape((-1, 3))
        nb_indices = result['index'].reshape((-1))
        nb_interp_ratio = result['interp_ratio'].reshape((-1, 1))
        nb_colors = vbg_color[nb_indices] * nb_interp_ratio
        sum_colors = nb_colors.reshape((depth.rows, depth.columns, 8, 3)).sum(
            (2)) / 255.0

        axs[0, 0].imshow(colorized_depth.as_tensor().cpu().numpy())
        axs[0, 0].set_title('depth')

        axs[0, 1].imshow(result['normal'].cpu().numpy())
        axs[0, 1].set_title('normal')

        axs[1, 0].imshow(result['color'].cpu().numpy())
        axs[1, 0].set_title('color via kernel')

        axs[1, 1].imshow(sum_colors.cpu().numpy())
        axs[1, 1].set_title('color via indexing')

        plt.tight_layout()
        plt.show()
