#!/usr/bin/env python3

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys, os
sys.path.append("../..")
from py3d import *
import numpy as np
import matplotlib.pyplot as plt

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # draw_geometries([pcd])
    print("1")
    vis = Visualizer()
    print("2")
    vis.create_window()
    print("3")
    vis.add_geometry(pcd)
    print("4")
    vis.run()
    print("5")
    vis.destroy_window()
    print("6")

if __name__ == "__main__":
    pcd = read_point_cloud("../../TestData/fragment.ply")

    print("1. Customized visualization to mimic DrawGeometry")
    custom_draw_geometry(pcd)

