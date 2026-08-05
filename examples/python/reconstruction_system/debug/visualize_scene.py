# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import argparse
import os
import numpy as np
import open3d as o3d

POINTS_PER_FRUSTUM = 5
EDGES_PER_FRUSTUM = 8


def lineset_from_pose_graph(pose_graph):
    points = []
    colors = []
    lines = []

    cnt = 0
    for node in pose_graph.nodes:
        pose = np.array(node.pose)

        l = 0.1
        points.append((pose @ np.array([0, 0, 0, 1]).T)[:3])
        points.append((pose @ np.array([l, l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-l, -l, 2 * l, 1]).T)[:3])
        points.append((pose @ np.array([-l, l, 2 * l, 1]).T)[:3])

        lines.append([cnt + 0, cnt + 1])
        lines.append([cnt + 0, cnt + 2])
        lines.append([cnt + 0, cnt + 3])
        lines.append([cnt + 0, cnt + 4])
        lines.append([cnt + 1, cnt + 2])
        lines.append([cnt + 2, cnt + 3])
        lines.append([cnt + 3, cnt + 4])
        lines.append([cnt + 4, cnt + 1])

        for i in range(0, EDGES_PER_FRUSTUM):
            colors.append(np.array([1, 0, 0]))

        cnt += POINTS_PER_FRUSTUM

    for edge in pose_graph.edges:
        s = edge.source_node_id
        t = edge.target_node_id
        lines.append([POINTS_PER_FRUSTUM * s, POINTS_PER_FRUSTUM * t])
        colors.append(
            np.array([0, 1, 0]) if edge.uncertain else np.array([0, 0, 1]))

    lineset = o3d.geometry.LineSet()
    lineset.points = o3d.utility.Vector3dVector(np.vstack(points))
    lineset.lines = o3d.utility.Vector2iVector(np.vstack(lines).astype(int))
    lineset.colors = o3d.utility.Vector3dVector(np.vstack(colors))

    return lineset


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('scene', help='path to reconstructed scene (.ply)')
    parser.add_argument('--poses', type=str, help='path to pose graph')
    args = parser.parse_args()

    path_to_scene = args.scene
    scene = o3d.io.read_point_cloud(path_to_scene)

    geometries = [scene]
    if args.poses:
        path_to_posegraph = args.poses
        pose_graph = o3d.io.read_pose_graph(path_to_posegraph)

        lineset = lineset_from_pose_graph(pose_graph)
        geometries.append(lineset)

    o3d.visualization.draw_geometries(geometries)
