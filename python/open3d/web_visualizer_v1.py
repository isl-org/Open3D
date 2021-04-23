import ipywidgets as widgets
from traitlets import validate, observe, Unicode, TraitError
from IPython.display import display
import json


@widgets.register
class WebVisualizer(widgets.DOMWidget):
    """An example widget."""

    # Name of the widget view class in front-end
    _view_name = Unicode('WebVisualizerView').tag(sync=True)

    # Name of the widget model class in front-end
    _model_name = Unicode('WebVisualizerModel').tag(sync=True)

    # Name of the front-end module containing widget view
    _view_module = Unicode('open3d').tag(sync=True)

    # Name of the front-end module containing widget model
    _model_module = Unicode('open3d').tag(sync=True)

    # Version of the front-end module containing widget view
    _view_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)
    # Version of the front-end module containing widget model
    _model_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)

    # Widget specific property.
    # Widget properties are defined as traitlets. Any property tagged with `sync=True`
    # is automatically synced to the frontend *any* time it changes in Python.
    # It is synced back to Python from the frontend *any* time the model is touched.
    window_uid = Unicode("window_UNDEFINED", help="Window UID").tag(sync=True)

    # Two-way communication channels. It is possible to just use one channel.3
    pyjs_channel = Unicode("Empty pyjs_channel.",
                           help="Python->JS message channel.").tag(sync=True)
    jspy_channel = Unicode("Empty jspy_channel.",
                           help="JS->Python message channel.").tag(sync=True)

    def show(self):
        display(self)

    def pyjs_send(self, message):
        self.pyjs_channel = message

    # TODO: Forward call to WebRTC server's call_http_request.
    def call_http_request(self, entry_point, query_string, data):
        return f"Called Http Request: {entry_point}, {query_string}, {data}!"

    @validate('window_uid')
    def _valid_window_uid(self, proposal):
        if proposal['value'][:7] != "window_":
            raise TraitError('window_uid must be "window_xxx".')
        return proposal['value']

    @observe('jspy_channel')
    def on_jspy_message(self, change):
        jspy_message = change["new"]
        print(f"js->py message received: {jspy_message}")
        try:
            # Hard-coded to call call_http_request.
            jspy_request = json.loads(jspy_message)
            if "func" not in jspy_request or jspy_request[
                    "func"] != "call_http_request":
                raise ValueError(f"Invalid jspy function: {jspy_message}")
            if "args" not in jspy_request or len(jspy_request["args"]) != 3:
                raise ValueError(
                    f"Invalid jspy function arguments: {jspy_message}")
            result = self.call_http_request(jspy_request["args"][0],
                                            jspy_request["args"][1],
                                            jspy_request["args"][2])
            print(f"py->js sending: {result}")
            self.pyjs_send(result)
        except:
            print(f"jspy message is not a valid function call: {jspy_message}")
