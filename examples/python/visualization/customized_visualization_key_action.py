# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2023 www.open3d.org
# SPDX-License-Identifier: MIT
# ----------------------------------------------------------------------------

import open3d as o3d


def custom_key_action_without_kb_repeat_delay(pcd):
    rotating = False

    vis = o3d.visualization.VisualizerWithKeyCallback()

    def key_action_callback(vis, action, mods):
        nonlocal rotating
        print(action)
        if action == 1:  # key down
            rotating = True
        elif action == 0:  # key up
            rotating = False
        elif action == 2:  # key repeat
            pass
        return True

    def animation_callback(vis):
        nonlocal rotating
        if rotating:
            ctr = vis.get_view_control()
            ctr.rotate(10.0, 0.0)

    # key_action_callback will be triggered when there's a keyboard press, release or repeat event
    vis.register_key_action_callback(32, key_action_callback)  # space

    # animation_callback is always repeatedly called by the visualizer
    vis.register_animation_callback(animation_callback)

    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


if __name__ == "__main__":
    ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_data.path)

    print(
        "Customized visualization with smooth key action (without keyboard repeat delay)"
    )
    custom_key_action_without_kb_repeat_delay(pcd)
