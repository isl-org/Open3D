# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import os, sys
import open3d as o3d
import numpy as np
import time

from config import ConfigParser
from split_fragments import split_fragments, load_fragments
from rgbd_odometry import rgbd_odometry, rgbd_loop_closure
from pose_graph_optim import PoseGraphWrapper
from integrate import integrate

from common import load_intrinsic

if __name__ == '__main__':
    parser = ConfigParser()
    parser.add(
        '--config',
        is_config_file=True,
        help='YAML config file path. Please refer to default_config.yml as a '
        'reference. It overrides the default config file, but will be '
        'overridden by other command line inputs.')
    parser.add('--default_dataset',
               help='Default dataset is used when config file is not provided. '
               'Default dataset may be selected from the following options: '
               '[lounge, jack_jack]',
               default='lounge')
    parser.add('--path_npz', type=str, default='test.npz')
    parser.add('--split_fragments', action='store_true')
    parser.add('--fragment_odometry', action='store_true')
    parser.add('--fragment_integration', action='store_true')
    config = parser.get_config()

    if config.split_fragments:
        split_fragments(config)

    if config.fragment_odometry:
        start = time.time()
        intrinsic = load_intrinsic(config)
        depth_lists, color_lists = load_fragments(config)

        for frag_id, (depth_list,
                      color_list) in enumerate(zip(depth_lists, color_lists)):
            pose_graph = PoseGraphWrapper()

            # Odometry: (i, i+1), trans(i, i+1), info(i, i+1)
            edges, trans, infos = rgbd_odometry(depth_list, color_list,
                                                intrinsic, config)
            pose_i2w = np.eye(4)
            pose_graph.add_node(0, pose_i2w.copy())

            for i in range(len(trans)):
                trans_i2j = trans[i]
                info_i2j = infos[i]

                trans_j2i = np.linalg.inv(trans_i2j)
                pose_j2w = pose_i2w @ trans_j2i

                pose_graph.add_node(i + 1, pose_j2w.copy())
                pose_graph.add_edge(i, i + 1, trans_i2j.copy(), info_i2j.copy(),
                                    False)
                pose_i2w = pose_j2w

            # Loop closure: (i, j), trans(i, j), info(i, j) where i, j are multipliers of intervals
            edges, trans, infos = rgbd_loop_closure(depth_list, color_list,
                                                    intrinsic, config)
            for i in range(len(edges)):
                ki, kj = edges[i]
                trans_i2j = trans[i]
                info_i2j = infos[i]

                pose_graph.add_edge(ki, kj, trans_i2j.copy(), info_i2j.copy(),
                                    True)

            pose_graph.solve_()
            pose_graph.save(
                os.path.join(config.path_dataset, 'fragments',
                             'fragment_posegraph_{:03d}.json'.format(frag_id)))
        end = time.time()
        print(
            'Pose graph generation and optimization takes {:.3f}s for {} fragments.'
            .format(end - start, len(depth_lists)))

    if config.fragment_integration:
        start = time.time()
        intrinsic = load_intrinsic(config)
        depth_lists, color_lists = load_fragments(config)

        for frag_id, (depth_list,
                      color_list) in enumerate(zip(depth_lists, color_lists)):
            pose_graph = PoseGraphWrapper.load(
                os.path.join(config.path_dataset, 'fragments',
                             'fragment_posegraph_{:03d}.json'.format(frag_id)))
            extrinsics = pose_graph.export_extrinsics()

            vbg = integrate(depth_list,
                            color_list,
                            intrinsic,
                            intrinsic,
                            extrinsics,
                            config=config)

            pcd = vbg.extract_point_cloud()

            # Float color does not load correctly in the t.io mode
            o3d.io.write_point_cloud(
                os.path.join(config.path_dataset, 'fragments',
                             'fragment_pcd_{:03d}.ply'.format(frag_id)),
                pcd.to_legacy())

        end = time.time()
        print('TSDF integration takes {:.3f}s for {} fragments.'.format(
            end - start, len(depth_lists)))
