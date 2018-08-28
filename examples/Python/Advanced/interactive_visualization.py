# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import numpy as np
import copy
from open3d import *

def demo_crop_geometry():
    print("Demo for manual geometry cropping")
    print("1) Press 'Y' twice to align geometry with negative direction of y-axis")
    print("2) Press 'K' to lock screen and to switch to selection mode")
    print("3) Drag for rectangle selection,")
    print("   or use ctrl + left click for polygon selection")
    print("4) Press 'C' to get a selected geometry and to save it")
    print("5) Press 'F' to switch to freeview mode")
    pcd = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    draw_geometries_with_editing([pcd])

def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    draw_geometries([source_temp, target_temp])

def pick_points(pcd):
    print("")
    print("1) Please pick at least three correspondences using [shift + left click]")
    print("   Press [shift + right click] to undo point picking")
    print("2) Afther picking points, press q for close the window")
    vis = VisualizerWithEditing()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run() # user picks points
    vis.destroy_window()
    print("")
    return vis.get_picked_points()

def demo_manual_registration():
    print("Demo for manual ICP")
    source = read_point_cloud("../../TestData/ICP/cloud_bin_0.pcd")
    target = read_point_cloud("../../TestData/ICP/cloud_bin_2.pcd")
    print("Visualization of two point clouds before manual alignment")
    draw_registration_result(source, target, np.identity(4))

    # pick points from two point clouds and builds correspondences
    picked_id_source = pick_points(source)
    picked_id_target = pick_points(target)
    assert(len(picked_id_source)>=3 and len(picked_id_target)>=3)
    assert(len(picked_id_source) == len(picked_id_target))
    corr = np.zeros((len(picked_id_source),2))
    corr[:,0] = picked_id_source
    corr[:,1] = picked_id_target

    # estimate rough transformation using correspondences
    print("Compute a rough transform using the correspondences given by user")
    p2p = TransformationEstimationPointToPoint()
    trans_init = p2p.compute_transformation(source, target,
             Vector2iVector(corr))

    # point-to-point ICP for refinement
    print("Perform point-to-point ICP refinement")
    threshold = 0.03 # 3cm distance threshold
    reg_p2p = registration_icp(source, target, threshold, trans_init,
            TransformationEstimationPointToPoint())
    draw_registration_result(source, target, reg_p2p.transformation)
    print("")

if __name__ == "__main__":
    demo_crop_geometry()
    demo_manual_registration()
