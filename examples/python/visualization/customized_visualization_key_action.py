# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2024 www.open3d.org
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


def custom_mouse_action(pcd):

    vis = o3d.visualization.VisualizerWithKeyCallback()
    buttons = ['left', 'right', 'middle']
    actions = ['up', 'down']
    mods_name = ['shift', 'ctrl', 'alt', 'cmd']

    def on_key_action(vis, action, mods):
        print("on_key_action", action, mods)

    vis.register_key_action_callback(ord("A"), on_key_action)

    def on_mouse_move(vis, x, y):
        print(f"on_mouse_move({x:.2f}, {y:.2f})")

    def on_mouse_scroll(vis, x, y):
        print(f"on_mouse_scroll({x:.2f}, {y:.2f})")

    def on_mouse_button(vis, button, action, mods):
        pressed_mods = " ".join(
            [mods_name[i] for i in range(4) if mods & (1 << i)])
        print(f"on_mouse_button: {buttons[button]}, {actions[action]}, " +
              pressed_mods)

    vis.register_mouse_move_callback(on_mouse_move)
    vis.register_mouse_scroll_callback(on_mouse_scroll)
    vis.register_mouse_button_callback(on_mouse_button)

    vis.create_window()
    vis.add_geometry(pcd)
    vis.run()


if __name__ == "__main__":
    ply_data = o3d.data.PLYPointCloud()
    pcd = o3d.io.read_point_cloud(ply_data.path)

    print("Customized visualization with smooth key action "
          "(without keyboard repeat delay). Press the space-bar.")
    custom_key_action_without_kb_repeat_delay(pcd)
    print("Customized visualization with mouse action.")
    custom_mouse_action(pcd)
