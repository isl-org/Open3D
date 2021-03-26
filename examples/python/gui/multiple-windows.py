#!/usr/bin/env python
import numpy as np
import open3d as o3d
import os
import threading
import time

SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

CLOUD_NAME = "points"


def add_draw_window(geometries, window_name, width, height):
    vis = o3d.visualization.O3DVisualizer(title=window_name,
                                          width=width,
                                          height=height)
    count = 0
    for geometry in geometries:
        vis.add_geometry(f"geometry_{count}", geometry)
        count += 1
    vis.reset_camera_to_default()
    o3d.visualization.gui.Application.instance.add_window(vis)


def empty_box():
    pc_rad = 1.0
    r = 0.4
    big_bbox = o3d.geometry.AxisAlignedBoundingBox((-pc_rad, -3, -pc_rad),
                                                   (6.0 + r, 1.0 + r, pc_rad))

    add_draw_window([big_bbox], "Open3D empty_box", 640, 480)


def multi_objects():
    pc_rad = 1.0
    r = 0.4
    sphere_unlit = o3d.geometry.TriangleMesh.create_sphere(r)
    sphere_unlit.translate((0, 1, 0))
    sphere_colored_unlit = o3d.geometry.TriangleMesh.create_sphere(r)
    sphere_colored_unlit.paint_uniform_color((1.0, 0.0, 0.0))
    sphere_colored_unlit.translate((2, 1, 0))
    sphere_lit = o3d.geometry.TriangleMesh.create_sphere(r)
    sphere_lit.compute_vertex_normals()
    sphere_lit.translate((4, 1, 0))
    sphere_colored_lit = o3d.geometry.TriangleMesh.create_sphere(r)
    sphere_colored_lit.compute_vertex_normals()
    sphere_colored_lit.paint_uniform_color((0.0, 1.0, 0.0))
    sphere_colored_lit.translate((6, 1, 0))
    big_bbox = o3d.geometry.AxisAlignedBoundingBox((-pc_rad, -3, -pc_rad),
                                                   (6.0 + r, 1.0 + r, pc_rad))
    sphere_bbox = sphere_unlit.get_axis_aligned_bounding_box()
    sphere_bbox.color = (1.0, 0.5, 0.0)
    lines = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        sphere_lit.get_axis_aligned_bounding_box())
    lines_colored = o3d.geometry.LineSet.create_from_axis_aligned_bounding_box(
        sphere_colored_lit.get_axis_aligned_bounding_box())
    lines_colored.paint_uniform_color((0.0, 0.0, 1.0))

    add_draw_window([
        sphere_unlit, sphere_colored_unlit, sphere_lit, sphere_colored_lit,
        big_bbox, sphere_bbox, lines, lines_colored
    ], "Open3D multi_objects", 640, 480)


def run():
    app = o3d.visualization.gui.Application.instance
    app.initialize()

    empty_box()
    multi_objects()

    app.run()


if __name__ == "__main__":
    o3d.visualization.gui.Application.instance.enable_webrtc()
    run()
