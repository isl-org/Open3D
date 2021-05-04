import ipywidgets as widgets
from traitlets import validate, observe, Unicode, TraitError
from IPython.display import display
import json
import open3d as o3d
import threading
import open3d.visualization.gui as gui
from open3d.visualization import O3DVisualizer


@widgets.register
class WebVisualizer(widgets.DOMWidget):
    """Open3D Web Visualizer based on WebRTC."""

    # Name of the widget view class in front-end.
    _view_name = Unicode('WebVisualizerView').tag(sync=True)

    # Name of the widget model class in front-end.
    _model_name = Unicode('WebVisualizerModel').tag(sync=True)

    # Name of the front-end module containing widget view.
    _view_module = Unicode('open3d').tag(sync=True)

    # Name of the front-end module containing widget model.
    _model_module = Unicode('open3d').tag(sync=True)

    # Version of the front-end module containing widget view.
    # @...@ is configured by cpp/pybind/make_python_package.cmake.
    _view_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)
    # Version of the front-end module containing widget model.
    _model_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)

    # Widget specific property. Widget properties are defined as traitlets. Any
    # property tagged with `sync=True` is automatically synced to the frontend
    # *any* time it changes in Python. It is synced back to Python from the
    # frontend *any* time the model is touched.
    window_uid = Unicode("window_UNDEFINED", help="Window UID").tag(sync=True)

    # Two-way communication channels.
    pyjs_channel = Unicode("Empty pyjs_channel.",
                           help="Python->JS message channel.").tag(sync=True)
    jspy_channel = Unicode("Empty jspy_channel.",
                           help="JS->Python message channel.").tag(sync=True)

    def show(self):
        display(self)

    def _call_http_request(self, entry_point, query_string, data):
        webrtc_server = o3d.visualization.webrtc_server.WebRTCServer.instance
        result = webrtc_server.call_http_request(entry_point, query_string,
                                                 data)
        print(
            f"call_http_request({entry_point}, {query_string}, {query_string})"
            f"->{result}")
        return result

    @validate('window_uid')
    def _valid_window_uid(self, proposal):
        if proposal['value'][:7] != "window_":
            raise TraitError('window_uid must be "window_xxx".')
        return proposal['value']

    @observe('jspy_channel')
    def _on_jspy_channel(self, change):
        # self.result_map = {"0": "result0",
        #                    "1": "result1", ...};
        if not hasattr(self, "result_map"):
            self.result_map = dict()

        jspy_message = change["new"]
        print(f"jspy_message received: {jspy_message}")
        try:
            jspy_requests = json.loads(jspy_message)

            for call_id, payload in jspy_requests.items():
                if "func" not in payload or payload[
                        "func"] != "call_http_request":
                    raise ValueError(f"Invalid jspy function: {jspy_requests}")
                if "args" not in payload or len(payload["args"]) != 3:
                    raise ValueError(
                        f"Invalid jspy function arguments: {jspy_requests}")

                # Check if already in result.
                if not call_id in self.result_map:
                    json_result = self._call_http_request(
                        payload["args"][0], payload["args"][1],
                        payload["args"][2])
                    self.result_map[call_id] = json_result
        except:
            print(
                f"jspy_message is not a function call, ignored: {jspy_message}")
        else:
            print(f"pyjs_channel sending: {self.result_map}")
            self.pyjs_channel = json.dumps(self.result_map)


def _jupyter_labextension_paths():
    """Called by Jupyter Lab Server to detect if it is a valid labextension and
    to install the widget

    Returns
    =======
    src: Source directory name to copy files from. Webpack outputs generated files
        into this directory and Jupyter Lab copies from this directory during
        widget installation
    dest: Destination directory name to install widget files to. Jupyter Lab copies
        from `src` directory into <jupyter path>/labextensions/<dest> directory
        during widget installation
    """
    return [{
        'src': 'labextension',
        'dest': 'open3d',
    }]


def _jupyter_nbextension_paths():
    """Called by Jupyter Notebook Server to detect if it is a valid nbextension and
    to install the widget

    Returns
    =======
    section: The section of the Jupyter Notebook Server to change.
        Must be 'notebook' for widget extensions
    src: Source directory name to copy files from. Webpack outputs generated files
        into this directory and Jupyter Notebook copies from this directory during
        widget installation
    dest: Destination directory name to install widget files to. Jupyter Notebook copies
        from `src` directory into <jupyter path>/nbextensions/<dest> directory
        during widget installation
    require: Path to importable AMD Javascript module inside the
        <jupyter path>/nbextensions/<dest> directory
    """
    return [{
        'section': 'notebook',
        'src': 'nbextension',
        'dest': 'open3d',
        'require': 'open3d/extension'
    }]
