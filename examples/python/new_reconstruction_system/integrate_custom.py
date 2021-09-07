# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import os
import numpy as np
import open3d as o3d
import time
import matplotlib.pyplot as plt

from config import ConfigParser
from common import load_image_file_names, save_poses, load_intrinsic, load_extrinsics
import imageio

import torch
from torch.utils.dlpack import from_dlpack

from tqdm import tqdm


def to_torch(o3d_tensor):
    return from_dlpack(o3d_tensor.to_dlpack())


def integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
              config):
    if os.path.exists(config.npz_file):
        print('Voxel block grid npz file {} found, trying to load...'.format(
            config.npz_file))
        vbg = o3d.t.geometry.VoxelBlockGrid.load(config.npz_file)
        print('Loading finished.')
    else:
        print('Voxel block grid npz file {} not found, trying to integrate...'.
              format(config.npz_file))

        n_files = len(color_file_names)
        device = o3d.core.Device(config.device)

        voxel_size = 3.0 / 512
        trunc = voxel_size * 4
        res = 8

        vbg = o3d.t.geometry.VoxelBlockGrid(
            ('tsdf', 'weight'),
            (o3d.core.Dtype.Float32, o3d.core.Dtype.Float32), ((1), (1)),
            voxel_size, res, 100000, o3d.core.Device('CUDA:0'))

        start = time.time()
        for i in tqdm(range(n_files)):
            print('Integrating frame {}/{}'.format(i, n_files))

            depth = o3d.t.io.read_image(depth_file_names[i]).to(device)
            color = o3d.t.io.read_image(color_file_names[i]).to(device)
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
            torch.cuda.synchronize()
            end = time.time()
            print('hash map preparation: {}'.format(end - start))

            start = time.time()
            voxel_coords, voxel_indices = vbg.voxel_coordinates_and_flattened_indices(
                buf_indices)
            torch.cuda.synchronize()
            end = time.time()
            print('enumerate voxels: {}'.format(end - start))

            # Now project them to the depth and find association
            # (3, N) -> (2, N)
            start = time.time()
            extrinsic_dev = extrinsic.to(device, o3d.core.Dtype.Float32)
            xyz = extrinsic_dev[:3, :3] @ voxel_coords.T() + extrinsic_dev[:3,
                                                                           3:]

            # o3d.visualization.draw(
            #     [o3d.t.geometry.PointCloud(xyz.T())])

            intrinsic_dev = intrinsic.to(device, o3d.core.Dtype.Float32)
            uvd = intrinsic_dev @ xyz
            d = uvd[2]
            u = (uvd[0] / d).round().to(o3d.core.Dtype.Int64)
            v = (uvd[1] / d).round().to(o3d.core.Dtype.Int64)
            torch.cuda.synchronize()
            end = time.time()
            print('geometry transformation: {}'.format(end - start))

            start = time.time()
            mask_proj = (d > 0) & (u >= 0) & (v >= 0) & (u < depth.columns) & (
                v < depth.rows)

            v_proj = v[mask_proj]
            u_proj = u[mask_proj]
            d_proj = d[mask_proj]
            depth_readings = depth.as_tensor()[v_proj, u_proj, 0].to(
                o3d.core.Dtype.Float32) / config.depth_scale
            color_readings = color.as_tensor()[v_proj, u_proj].to(
                o3d.core.Dtype.Float32)
            sdf = depth_readings - d_proj

            mask_inlier = (depth_readings > 0) \
                & (depth_readings < config.depth_max) \
                & (sdf >= -trunc)

            sdf[sdf >= trunc] = trunc
            sdf = sdf / trunc
            torch.cuda.synchronize()
            end = time.time()
            print('association: {}'.format(end - start))

            start = time.time()
            weight = vbg.attribute('weight').reshape((-1, 1))
            # color = vbg.attribute('color').reshape((-1, 3))
            tsdf = vbg.attribute('tsdf').reshape((-1, 1))

            valid_voxel_indices = voxel_indices[mask_proj][mask_inlier]
            w = weight[valid_voxel_indices]
            wp = w + 1

            tsdf[valid_voxel_indices] \
                = (tsdf[valid_voxel_indices] * w +
                   sdf[mask_inlier].reshape(w.shape)) / (wp)
            # color[valid_voxel_indices] \
            #     = (color[valid_voxel_indices] * w +
            #              color_readings[mask_inlier]) / (wp)
            weight[valid_voxel_indices] = wp
            torch.cuda.synchronize()
            end = time.time()
            print('update: {}'.format(end - start))

        print('Saving to {}...'.format(config.npz_file))
        vbg.save(config.npz_file)
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
    parser.add('--path_trajectory',
               help='path to the trajectory .log or .json file.')
    parser.add('--npz_file',
               help='path to the npz file that stores voxel block grid.',
               default='vbg.npz')
    config = parser.get_config()

    depth_file_names, color_file_names = load_image_file_names(config)
    intrinsic = load_intrinsic(config)
    extrinsics = load_extrinsics(config.path_trajectory, config)

    vbg = integrate(depth_file_names, color_file_names, intrinsic, extrinsics,
                    config)

    mesh = vbg.extract_triangle_mesh()
    o3d.visualization.draw([mesh.to_legacy()])

    pcd = vbg.extract_point_cloud()
    o3d.visualization.draw([pcd])
