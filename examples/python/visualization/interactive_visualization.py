# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

# examples/python/visualization/interactive_visualization.py

import numpy as np
import copy
import open3d as o3d


def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print(
        "1) Press 'Y' twice to align geometry with negative direction of y-axis"
    )
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry")
    print("5) Press 'S' to save the selected geometry")
    print("6) Press 'F' to switch to freeview mode")
    pcd_data = o3d.data.DemoICPPointClouds()
    pcd = o3d.io.read_point_cloud(pcd_data.paths[0])
    o3d.visualization.draw_geometries_with_editing([pcd])


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def prepare_data():
    pcd_data = o3d.data.DemoICPPointClouds()
    source = o3d.io.read_point_cloud(pcd_data.paths[0])
    target = o3d.io.read_point_cloud(pcd_data.paths[2])
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))
    return source, target


def pick_points(pcd):
    print("")
    print(
        "1) Please pick at least three correspondences using [shift + left click]"
    )
    print("   Press [shift + right click] to undo point picking")
    print("2) After picking points, press 'Q' to close the window")
    vis = o3d.visualization.VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()  # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()


def register_via_correspondences(source, target, source_points, target_points):
    corr = np.zeros((len(source_points), 2))
    corr[:, 0] = source_points
    corr[:, 1] = target_points
    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = o3d.pipelines.registration.TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
                                            o3d.utility.Vector2iVector(corr))
    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03  # 3cm distance threshold
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)


def demo_manual_registration():
    print("Demo for manual ICP")
    source, target = prepare_data()

    # pick points from two point clouds and builds correspondences
    source_points = pick_points(source)
    target_points = pick_points(target)
    assert (len(source_points) >= 3 and len(target_points) >= 3)
    assert (len(source_points) == len(target_points))
    register_via_correspondences(source, target, source_points, target_points)
    print("")


if __name__ == "__main__":
    demo_crop_geometry()
    demo_manual_registration()
