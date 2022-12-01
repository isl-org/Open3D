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

from . import gui
from . import O3DVisualizer


def draw(geometry=None,
         title="Open3D",
         width=1024,
         height=768,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         ibl=None,
         ibl_intensity=None,
         show_skybox=None,
         show_ui=None,
         raw_mode=False,
         point_size=None,
         line_width=None,
         animation_time_step=1.0,
         animation_duration=None,
         rpc_interface=False,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None,
         non_blocking_and_return_uid=False):
    gui.Application.instance.initialize()
    w = O3DVisualizer(title, width, height)
    w.set_background(bg_color, bg_image)

    if actions is not None:
        for a in actions:
            w.add_action(a[0], a[1])

    if point_size is not None:
        w.point_size = point_size

    if line_width is not None:
        w.line_width = line_width

    def add(g, n):
        if isinstance(g, dict):
            w.add_geometry(g)
        else:
            w.add_geometry("Object " + str(n), g)

    n = 1
    if isinstance(geometry, list):
        for g in geometry:
            add(g, n)
            n += 1
    elif geometry is not None:
        add(geometry, n)

    w.reset_camera_to_default()  # make sure far/near get setup nicely
    if lookat is not None and eye is not None and up is not None:
        w.setup_camera(field_of_view, lookat, eye, up)

    w.animation_time_step = animation_time_step
    if animation_duration is not None:
        w.animation_duration = animation_duration

    if show_ui is not None:
        w.show_settings = show_ui

    if ibl is not None:
        w.set_ibl(ibl)

    if ibl_intensity is not None:
        w.set_ibl_intensity(ibl_intensity)

    if show_skybox is not None:
        w.show_skybox(show_skybox)

    if rpc_interface:
        w.start_rpc_interface(address="tcp://127.0.0.1:51454", timeout=10000)

        def stop_rpc():
            w.stop_rpc_interface()
            return True

        w.set_on_close(stop_rpc)

    if raw_mode:
        w.enable_raw_mode(True)

    if on_init is not None:
        on_init(w)
    if on_animation_frame is not None:
        w.set_on_animation_frame(on_animation_frame)
    if on_animation_tick is not None:
        w.set_on_animation_tick(on_animation_tick)

    gui.Application.instance.add_window(w)
    if non_blocking_and_return_uid:
        return w.uid
    else:
        gui.Application.instance.run()
