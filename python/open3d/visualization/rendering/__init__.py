# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2020 www.open3d.org
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

if "@BUILD_GUI@" == "ON":
    import open3d
    if open3d.__DEVICE_API__ == 'cuda':
        from open3d.cuda.pybind.visualization.rendering import *
    else:
        from open3d.cpu.pybind.visualization.rendering import *
else:
    print("Open3D was not compiled with BUILD_GUI, but script is importing "
          "open3d.visualization.rendering")

import math
from .. import gui

class RenderToImage:
    def __init__(self, width, height):
        """Creates an object that will handle all the necessary parts of
           rendering to an image. Takes two arguments: width, height."""
        open3d.visualization.gui.Application.instance.initialize()
        self._width = width
        self._height = height
        self._window = open3d.visualization.gui.Window("Open3D.RenderToImage",
                                                       width, height, 0, 0)
        # The rendering backend can only render from the same size as the
        # window. On some operating systems, notably macOS, the window is
        # specified in "virtual pixels", not device pixels, and the number of
        # pixels you actually get depends on the scaling factor. Handle that
        # here.
        if self._window.content_rect.width > width:
            scaling = self._window.scaling
            self._window.size = open3d.visualization.gui.Size(int(math.ceil(width / scaling)),
                                          int(math.ceil(height / scaling)))

    def set_background_color(self, color4):
        """Sets the background color. Takes one argument: [r, g, b, a]"""
        self._window.renderer.set_clear_color(color4)

    def create_scene(self):
        """Creates and returns the open3d.visualization.rendering.Open3DScene
           that can be used to construct the scene to be rendered. Note that
           the caller needs to ensure that the scene is destroyed prior to
           calling done() otherwise we will crash. This can most easily be
           accomplished by assigning None to the variable after you are finished
           using the scene."""
        scene = Open3DScene(self._window.renderer)
        scene.set_view_size(self._width, self._height)
        return scene

    def render(self, open3dscene):
        """Renders the scene and returns an open3d.geometry.Image of the scene"""
        # If we call render() a again, we need to do a draw to clear the
        # remnants of the last image from the window's buffer.
        self._window.post_redraw()

        # Getting the clear color actually set in Filament seems to require
        # running a tick for everything to propagate through.
        open3d.visualization.gui.Application.instance.run_one_tick()

        return open3d.visualization.gui.Application.instance.render_to_image(self._window,
                                                         open3dscene)

    def done(self):
        """Cleans up the internal rendering objects. Failure to call this will
           result in a crash on exit."""
        # We need to close the window and cleanup filament. Calling quit() will
        # close the window, and then we run for a tick, which will see that
        # no more windows are left and cleanup filament.
        open3d.visualization.gui.Application.instance.quit()
        open3d.visualization.gui.Application.instance.run_one_tick()
