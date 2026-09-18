# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------
"""Animate a fixed-topology tensor line set with in-place buffer updates."""

import numpy as np
import open3d as o3d


def main():
    app = o3d.visualization.gui.Application.instance
    app.initialize()
    visualizer = o3d.visualization.O3DVisualizer(
        "Fixed-topology tensor line update", 1024, 768)

    x = np.linspace(-2.0, 2.0, 200, dtype=np.float32)
    points = np.column_stack((x, np.zeros_like(x), np.zeros_like(x)))
    lines = np.column_stack(
        (np.arange(len(x) - 1), np.arange(1, len(x)))).astype(np.int64)
    colors = np.tile(np.array([[0.1, 0.6, 1.0]], dtype=np.float32),
                     (len(lines), 1))
    line_set = o3d.t.geometry.LineSet(o3d.core.Tensor(points),
                                      o3d.core.Tensor(lines))
    line_set.line.colors = o3d.core.Tensor(colors)

    material = o3d.visualization.rendering.MaterialRecord()
    material.shader = "unlitLine"
    material.line_width = 4.0
    visualizer.add_geometry("wave", line_set, material)
    visualizer.reset_camera_to_default()

    update_flags = (o3d.visualization.rendering.Scene.UPDATE_POINTS_FLAG |
                    o3d.visualization.rendering.Scene.UPDATE_COLORS_FLAG)

    def on_animation_tick(vis, _delta_time, total_time):
        updated = points.copy()
        updated[:, 1] = 0.5 * np.sin(3.0 * updated[:, 0] - 4.0 * total_time)
        animated_colors = colors.copy()
        animated_colors[:, 0] = 0.5 + 0.5 * np.sin(total_time)
        line_set.point.positions = o3d.core.Tensor(updated)
        line_set.line.colors = o3d.core.Tensor(animated_colors)
        vis.update_geometry("wave", line_set, update_flags)
        return o3d.visualization.O3DVisualizer.TickResult.REDRAW

    visualizer.set_on_animation_tick(on_animation_tick)
    visualizer.animation_duration = 3600.0
    visualizer.is_animating = True
    app.add_window(visualizer)
    app.run()


if __name__ == "__main__":
    main()
