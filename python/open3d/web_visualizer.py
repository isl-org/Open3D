import ipywidgets as widgets
from traitlets import validate, observe, Unicode, TraitError
from IPython.display import display
import json
import open3d as o3d


@widgets.register
class WebVisualizer(widgets.DOMWidget):
    """An example widget."""

    # Name of the widget view class in front-end.
    _view_name = Unicode('WebVisualizerView').tag(sync=True)

    # Name of the widget model class in front-end.
    _model_name = Unicode('WebVisualizerModel').tag(sync=True)

    # Name of the front-end module containing widget view.
    _view_module = Unicode('open3d').tag(sync=True)

    # Name of the front-end module containing widget model.
    _model_module = Unicode('open3d').tag(sync=True)

    # Version of the front-end module containing widget view.
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

    def pyjs_send(self, message):
        # Parse input
        message_dict = json.loads(message)
        if "call_id" not in message_dict:
            raise ValueError(f"pyjs_send call_id not in message: {message}")
        call_id = message_dict["call_id"]
        if "json_result" not in message_dict:
            raise ValueError(f"json_result call_id not in message: {message}")
        json_result = message_dict["json_result"]

        # Insert new entry to channel_dict
        channel_dict = json.loads(self.pyjs_channel)
        channel_dict[call_id] = json_result

        self.pyjs_channel = json.dumps(channel_dict)

    def call_http_request(self, entry_point, query_string, data):
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
    def on_jspy_message(self, change):

        # self.result_map = {"0": "result0", "1": "result1"};
        if not hasattr(self, "result_map"):
            self.result_map = dict()

        jspy_message = change["new"]
        print(f"js->py message received: {jspy_message}")
        new_call = False
        try:
            jspy_requests = json.loads(jspy_message)
            print(f"!!! jspy_message: {jspy_message}")
            print(f"!!! jspy_requests: {jspy_requests}")
            print(f"!!! type(jspy_requests): {type(jspy_requests)}")

            for call_id, payload in jspy_requests.items():
                print(f"!!! ONE call_id: {payload}")
                print(f"!!! ONE payload: {payload}")
                if "func" not in payload or payload[
                        "func"] != "call_http_request":
                    raise ValueError(f"Invalid jspy function: {jspy_requests}")
                if "args" not in payload or len(payload["args"]) != 3:
                    raise ValueError(
                        f"Invalid jspy function arguments: {jspy_requests}")

                # Check if already in result
                if not call_id in self.result_map:
                    json_result = self.call_http_request(
                        payload["args"][0], payload["args"][1],
                        payload["args"][2])
                    self.result_map[call_id] = json_result
                    new_call = True
        except:
            print(
                f"js->py message is not a function call, ignored: {jspy_message}"
            )
        else:
            print(f"py->js sending: {self.result_map}")
            self.pyjs_channel = json.dumps(self.result_map)
