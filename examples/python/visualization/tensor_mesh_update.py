# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Animate a fixed-topology tensor mesh with in-place GPU buffer updates."""

import numpy as np
import open3d as o3d


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    visualizer = o3d.visualization.O3DVisualizer(
        "Fixed-topology tensor mesh update", 1024, 768)

    legacy_mesh = o3d.geometry.TriangleMesh.create_sphere(resolution=40)
    mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    mesh.compute_vertex_normals()
    initial_positions = mesh.vertex.positions.numpy().copy()

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "defaultLit"
    visualizer.add_geometry("deforming sphere", mesh, material)
    visualizer.reset_camera_to_default()

    update_flags = (o3d.visualization.rendering.Scene.UPDATE_POINTS_FLAG |
                    o3d.visualization.rendering.Scene.UPDATE_NORMALS_FLAG)

    def on_animation_tick(vis, _delta_time, total_time):
        positions = initial_positions.copy()
        radial_scale = 1.0 + 0.15 * np.sin(3.0 * total_time +
                                           4.0 * positions[:, 2])
        positions *= radial_scale[:, None]
        mesh.vertex.positions = o3d.core.Tensor(positions)
        mesh.compute_vertex_normals()
        vis.update_geometry("deforming sphere", mesh, update_flags)
        return o3d.visualization.O3DVisualizer.TickResult.REDRAW

    visualizer.set_on_animation_tick(on_animation_tick)
    visualizer.animation_duration = 3600.0
    visualizer.is_animating = True
    app.add_window(visualizer)
    app.run()


if __name__ == "__main__":
    main()
