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

from config import ConfigParser
from common import load_rgbd_file_names, load_depth_file_names, save_poses, load_intrinsic, load_extrinsics, get_default_testdata

from tqdm import tqdm


def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    if os.path.exists(config.path_npz):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.path_npz))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.path_npz)
        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.path_npz))

        n_files = len(depth_file_names)
        device = o3d.core.Device(config.device)

        voxel_size = 3.0 / 512
        trunc = voxel_size * 4
        res = 8

        if config.integrate_color:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                ('tsdf', 'weight', 'color'),
                (o3c.float32, o3c.float32, o3c.float32), ((1), (1), (3)),
                3.0 / 512, 8, 100000, device)
        else:
            vbg = o3d.t.geometry.VoxelBlockGrid(
                ('tsdf', 'weight'), (o3c.float32, o3c.float32), ((1), (1)),
                3.0 / 512, 8, 100000, device)

        start = time.time()
        for i in tqdm(range(n_files)):
            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            extrinsic = extrinsics[i]

            start = time.time()
            # Get active frustum block coordinates from input
            frustum_block_coords = vbg.compute_unique_block_coordinates(
                depth, intrinsic, extrinsic, config.depth_scale,
                config.depth_max)
            # Activate them in the underlying hash map (may have been inserted)
            vbg.hashmap().activate(frustum_block_coords)

            # Find buf indices in the underlying engine
            buf_indices, masks = vbg.hashmap().find(frustum_block_coords)
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
                buf_indices)
            o3d.core.cuda.synchronize()
            end = time.time()

            # Now project them to the depth and find association
            # (3, N) -> (2, N)
            start = time.time()
            extrinsic_dev = extrinsic.to(device, o3c.float32)
            xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3,
                                                                           3:]

            intrinsic_dev = intrinsic.to(device, o3c.float32)
            uvd = intrinsic_dev @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3c.int64)
            v = (uvd[1] / d).round().to(o3c.int64)
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (
                v < depth.rows)

            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]
            depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(
                o3c.float32) / config.depth_scale
            sdf = depth_readings - d_proj

            mask_inlier = (depth_readings > 0) \
                & (depth_readings < config.depth_max) \
                & (sdf >= -trunc)

            sdf[sdf >= trunc] = trunc
            sdf = sdf / trunc
            o3d.core.cuda.synchronize()
            end = time.time()

            start = time.time()
            weight = vbg.attribute('weight').reshape((-1, 1))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            wp = w + 1

            tsdf[valid_voxel_indices] \
                = (tsdf[valid_voxel_indices] * w +
                   sdf[mask_inlier].reshape(w.shape)) / (wp)
            if config.integrate_color:
                color = o3d.t.io.read_image(color_file_names[i]).to(device)
                color_readings = color.as_tensor()[v_proj,
                                                   u_proj].to(o3c.float32)

                color = vbg.attribute('color').reshape((-1, 3))
                color[valid_voxel_indices] \
                    = (color[valid_voxel_indices] * w +
                             color_readings[mask_inlier]) / (wp)

            weight[valid_voxel_indices] = wp
            o3d.core.cuda.synchronize()
            end = time.time()

        print('Saving to {}...'.format(config.path_npz))
        vbg.save(config.path_npz)
        print('Saving finished')

    return vbg


if __name__ == '__main__':

    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--integrate_color', action='store_true')
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--path_npz',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    if config.path_dataset == '':
        config.path_dataset = get_default_testdata()
        config.path_trajectory = os.path.join(config.path_dataset,
                                              'trajectory.log')

    if config.integrate_color:
        depth_file_names, color_file_names = load_rgbd_file_names(config)
    else:
        depth_file_names = load_depth_file_names(config)
        color_file_names = None

    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])
