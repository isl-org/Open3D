#!/usr/bin/env python3

# Open3D: www.open3d.org
# The MIT License (MIT)
# See license file or visit www.open3d.org for details

import sys, os
sys.path.append("../..")
from py3d import *
import numpy as np
import matplotlib.pyplot as plt

import faulthandler
faulthandler.enable()

def custom_draw_geometry(pcd):
    # The following code achieves the same effect as:
    # draw_geometries([pcd])
    vis = Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()
    vis.destroy_window()

if __name__ == "__main__":
    pcd = read_point_cloud("../../TestData/fragment.ply")

    print("1. Customized visualization to mimic DrawGeometry")
    custom_draw_geometry(pcd)

