# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
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
