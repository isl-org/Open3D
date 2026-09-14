# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# Copyright (c) 2018-2026 www.open3d.org
# SPDX-License-Identifier: MIT
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


def draw(*args, **kwargs):
    """Draw in Jupyter Cell.

    See :func:`open3d.visualization.draw` for the available arguments.
    """

    kwargs["non_blocking_and_return_uid"] = True
    window_uid = async_event_loop.run_sync(
        functools.partial(o3d.visualization.draw, *args, **kwargs))
    visualizer = WebVisualizer(window_uid=window_uid)
    visualizer.show()
