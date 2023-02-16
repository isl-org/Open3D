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

import ipywidgets
import traitlets
import IPython
import json
import functools
import open3d as o3d
# Note: the _AsyncEventLoop is started whenever this module is imported.
from open3d.visualization.async_event_loop import async_event_loop

from open3d._build_config import _build_config
if not _build_config["BUILD_JUPYTER_EXTENSION"]:
    raise RuntimeError(
        "Open3D WebVisualizer Jupyter extension is not available. To use "
        "WebVisualizer, build Open3D with -DBUILD_JUPYTER_EXTENSION=ON.")


@ipywidgets.register
class WebVisualizer(ipywidgets.DOMWidget):
    """Open3D Web Visualizer based on WebRTC."""

    # Name of the widget view class in front-end.
    _view_name = traitlets.Unicode('WebVisualizerView').tag(sync=True)

    # Name of the widget model class in front-end.
    _model_name = traitlets.Unicode('WebVisualizerModel').tag(sync=True)

    # Name of the front-end module containing widget view.
    _view_module = traitlets.Unicode('open3d').tag(sync=True)

    # Name of the front-end module containing widget model.
    _model_module = traitlets.Unicode('open3d').tag(sync=True)

    # Version of the front-end module containing widget view.
    # @...@ is configured by cpp/pybind/make_python_package.cmake.
    _view_module_version = traitlets.Unicode(
        '~@PROJECT_VERSION_THREE_NUMBER@').tag(sync=True)
    # Version of the front-end module containing widget model.
    _model_module_version = traitlets.Unicode(
        '~@PROJECT_VERSION_THREE_NUMBER@').tag(sync=True)

    # Widget specific property. Widget properties are defined as traitlets. Any
    # property tagged with `sync=True` is automatically synced to the frontend
    # *any* time it changes in Python. It is synced back to Python from the
    # frontend *any* time the model is touched.
    window_uid = traitlets.Unicode("window_UNDEFINED",
                                   help="Window UID").tag(sync=True)

    # Two-way communication channels.
    pyjs_channel = traitlets.Unicode(
        "Empty pyjs_channel.",
        help="Python->JS message channel.").tag(sync=True)
    jspy_channel = traitlets.Unicode(
        "Empty jspy_channel.",
        help="JS->Python message channel.").tag(sync=True)

    def show(self):
        IPython.display.display(self)

    def _call_http_api(self, entry_point, query_string, data):
        return o3d.visualization.webrtc_server.call_http_api(
            entry_point, query_string, data)

    @traitlets.validate('window_uid')
    def _valid_window_uid(self, proposal):
        if proposal['value'][:7] != "window_":
            raise traitlets.TraitError('window_uid must be "window_xxx".')
        return proposal['value']

    @traitlets.observe('jspy_channel')
    def _on_jspy_channel(self, change):
        # self.result_map = {"0": "result0",
        #                    "1": "result1", ...};
        if not hasattr(self, "result_map"):
            self.result_map = dict()

        jspy_message = change["new"]
        try:
            jspy_requests = json.loads(jspy_message)

            for call_id, payload in jspy_requests.items():
                if "func" not in payload or payload["func"] != "call_http_api":
                    raise ValueError(f"Invalid jspy function: {jspy_requests}")
                if "args" not in payload or len(payload["args"]) != 3:
                    raise ValueError(
                        f"Invalid jspy function arguments: {jspy_requests}")

                # Check if already in result.
                if not call_id in self.result_map:
                    json_result = self._call_http_api(payload["args"][0],
                                                      payload["args"][1],
                                                      payload["args"][2])
                    self.result_map[call_id] = json_result
        except:
            print(
                f"jspy_message is not a function call, ignored: {jspy_message}")
        else:
            self.pyjs_channel = json.dumps(self.result_map)


def draw(geometry=None,
         title="Open3D",
         width=640,
         height=480,
         actions=None,
         lookat=None,
         eye=None,
         up=None,
         field_of_view=60.0,
         bg_color=(1.0, 1.0, 1.0, 1.0),
         bg_image=None,
         show_ui=None,
         point_size=None,
         animation_time_step=1.0,
         animation_duration=None,
         rpc_interface=False,
         on_init=None,
         on_animation_frame=None,
         on_animation_tick=None):
    """Draw in Jupyter Cell"""

    window_uid = async_event_loop.run_sync(
        functools.partial(o3d.visualization.draw,
                          geometry=geometry,
                          title=title,
                          width=width,
                          height=height,
                          actions=actions,
                          lookat=lookat,
                          eye=eye,
                          up=up,
                          field_of_view=field_of_view,
                          bg_color=bg_color,
                          bg_image=bg_image,
                          show_ui=show_ui,
                          point_size=point_size,
                          animation_time_step=animation_time_step,
                          animation_duration=animation_duration,
                          rpc_interface=rpc_interface,
                          on_init=on_init,
                          on_animation_frame=on_animation_frame,
                          on_animation_tick=on_animation_tick,
                          non_blocking_and_return_uid=True))
    visualizer = WebVisualizer(window_uid=window_uid)
    visualizer.show()
