# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018 www.open3d.org
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

import ipywidgets as widgets
from traitlets import Unicode, Bool, validate, TraitError
from IPython.display import display
import open3d as o3
import time


@widgets.register
class WebVisualizer(widgets.DOMWidget):
    _view_name = Unicode('WebVisualizerView').tag(sync=True)
    _view_module = Unicode('open3d').tag(sync=True)
    _view_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)

    # Attributes: ipython traitlets
    value = Unicode('example@example.com',
                    help="The email value.").tag(sync=True)

    def show(self):
        display(self)

        # TODO: Remove this hack. This is to trigger WebRtcStreamer connection
        # AFTER the <video> tag has established.
        time.sleep(2)
        if self.value == "x@gmail.com":
            self.value = "y@gmail.com"
        else:
            self.value = "x@gmail.com"
