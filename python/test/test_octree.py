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
import numpy as np
import os

_eight_cubes_colors = np.array([
    [0.0, 0.0, 0.0],
    [0.1, 0.0, 0.0],
    [0.0, 0.1, 0.0],
    [0.1, 0.1, 0.0],
    [0.0, 0.0, 0.1],
    [0.1, 0.0, 0.1],
    [0.0, 0.1, 0.1],
    [0.1, 0.1, 0.1],
])

_eight_cubes_points = np.array([
    [0.5, 0.5, 0.5],
    [1.5, 0.5, 0.5],
    [0.5, 1.5, 0.5],
    [1.5, 1.5, 0.5],
    [0.5, 0.5, 1.5],
    [1.5, 0.5, 1.5],
    [0.5, 1.5, 1.5],
    [1.5, 1.5, 1.5],
])


def test_octree_OctreeNodeInfo():
    origin = [0, 0, 0]
    size = 2.0
    depth = 5
    child_index = 7

    node_info = o3d.geometry.OctreeNodeInfo(origin, size, depth, child_index)
    np.testing.assert_equal(node_info.origin, origin)
    np.testing.assert_equal(node_info.size, size)
    np.testing.assert_equal(node_info.depth, depth)
    np.testing.assert_equal(node_info.child_index, child_index)


def test_octree_OctreeColorLeafNode():
    color_leaf_node = o3d.geometry.OctreeColorLeafNode()
    color = [0.1, 0.2, 0.3]
    color_leaf_node.color = color
    np.testing.assert_equal(color_leaf_node.color, color)

    # Test copy constructor
    color_leaf_node_copy = o3d.geometry.OctreeColorLeafNode(color_leaf_node)
    np.testing.assert_equal(color_leaf_node_copy.color, color)

    # Test OctreeLeafNode's inherited operator== function
    assert color_leaf_node == color_leaf_node_copy
    assert color_leaf_node_copy == color_leaf_node

    # Test OctreeLeafNode's inherited clone() function
    color_leaf_node_clone = color_leaf_node.clone()
    np.testing.assert_equal(color_leaf_node_clone.color, color)
    assert color_leaf_node == color_leaf_node_clone
    assert color_leaf_node_clone == color_leaf_node


def test_octree_init():
    octree = o3d.geometry.Octree(1, [0, 0, 0], 2)


def test_octree_convert_from_point_cloud():
    octree = o3d.geometry.Octree(1, [0, 0, 0], 2)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(_eight_cubes_points)
    pcd.colors = o3d.utility.Vector3dVector(_eight_cubes_colors)
    octree.convert_from_point_cloud(pcd)


def test_octree_insert_point():
    octree = o3d.geometry.Octree(1, [0, 0, 0], 2)
    for point, color in zip(_eight_cubes_points, _eight_cubes_colors):
        f_init = o3d.geometry.OctreeColorLeafNode.get_init_function()
        f_update = o3d.geometry.OctreeColorLeafNode.get_update_function(color)
        octree.insert_point(point, f_init, f_update)


def test_octree_node_access():
    octree = o3d.geometry.Octree(1, [0, 0, 0], 2)
    for point, color in zip(_eight_cubes_points, _eight_cubes_colors):
        f_init = o3d.geometry.OctreeColorLeafNode.get_init_function()
        f_update = o3d.geometry.OctreeColorLeafNode.get_update_function(color)
        octree.insert_point(point, f_init, f_update)
    for i in range(8):
        np.testing.assert_equal(octree.root_node.children[i].color,
                                _eight_cubes_colors[i])


def test_octree_visualize():
    pcd_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)
    octree = o3d.geometry.Octree(8)
    octree.convert_from_point_cloud(pcd)
    # Enable the following line to test visualization
    # o3d.visualization.draw_geometries([octree])


def test_octree_voxel_grid_convert():
    pcd_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)
    octree = o3d.geometry.Octree(8)
    octree.convert_from_point_cloud(pcd)

    voxel_grid = octree.to_voxel_grid()
    octree_copy = voxel_grid.to_octree(max_depth=8)

    # Enable the following line to test visualization
    # o3d.visualization.draw_geometries([octree])
    # o3d.visualization.draw_geometries([voxel_grid])
    # o3d.visualization.draw_geometries([octree_copy])


def test_locate_leaf_node():
    pcd_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(pcd_data.path)

    max_depth = 5
    octree = o3d.geometry.Octree(max_depth)
    octree.convert_from_point_cloud(pcd, 0.01)

    # Try locating a few points
    for idx in range(0, len(pcd.points), 200):
        point = pcd.points[idx]
        node, node_info = octree.locate_leaf_node(np.array(point))
        # The located node must be in bound
        assert octree.is_point_in_bound(point, node_info.origin, node_info.size)
        # Leaf node must be located
        assert node_info.depth == max_depth
        # Leaf node's size must match
        assert node_info.size == octree.size / np.power(2, max_depth)
