import ipywidgets as widgets
from traitlets import validate, observe, Unicode, TraitError
from IPython.display import display
import json


@widgets.register
class WebVisualizer(widgets.DOMWidget):
    _view_name = Unicode('WebVisualizerView').tag(sync=True)
    _view_module = Unicode('open3d').tag(sync=True)
    _view_module_version = Unicode('~@PROJECT_VERSION_THREE_NUMBER@').tag(
        sync=True)

    # Attributes
    pyjs_channel = Unicode("Empty pyjs_channel.",
                           help="Python->JS message channel.").tag(sync=True)
    jspy_channel = Unicode("Empty jspy_channel.",
                           help="JS->Python message channel.").tag(sync=True)

    def pyjs_send(self, message):
        self.pyjs_channel = message

    def call_http_request(self, entry_point, query_string, data):
        return f"Called Http Request: {entry_point}, {query_string}, {data}!"

    @observe('jspy_channel')
    def on_jspy_message(self, change):
        jspy_message = change["new"]
        print(f"jspy message: {jspy_message}")
        try:
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
            self.pyjs_send(result)
        except:
            print(f"jspy message is not a valid function call: {jspy_message}")
