# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

# examples/Python/Tutorial/Advanced/camera_trajectory.py

import numpy as np
from open3d import *


def save_view_point(pcd, filename):
    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run() # user changes the view and press "q" to terminate
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    write_pinhole_camera_parameters(filename, param)
    vis.destroy_window()


def load_view_point(pcd, filename):
    vis = Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    param = read_pinhole_camera_parameters(filename)
    vis.add_geometry(pcd)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    pcd = read_point_cloud("../../TestData/fragment.pcd")
    save_view_point(pcd, "viewpoint.json")
    load_view_point(pcd, "viewpoint.json")
