var widgets = require("@jupyter-widgets/base");
var _ = require("lodash");
require("webrtc-adapter");
var WebRtcStreamer = require("./webrtcstreamer");

// See web_visualizer.py for the kernel counterpart to this file.

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including
//
//  - `_view_name`
//  - `_view_module`
//  - `_view_module_version`
//
//  - `_model_name`
//  - `_model_module`
//  - `_model_module_version`
//
//  when different from the base class.

// When serialiazing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
var WebVisualizerModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _model_name: "WebVisualizerModel",
    _view_name: "WebVisualizerView",
    _model_module: "open3d",
    _view_module: "open3d",
    _model_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
    _view_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
  }),
});

// Custom View. Renders the widget model.
var WebVisualizerView = widgets.DOMWidgetView.extend({
  sleep: function (time_ms) {
    return new Promise((resolve) => setTimeout(resolve, time_ms));
  },

  jspy_send: function (message) {
    this.model.set("jspy_channel", message);
    this.touch();
  },

  // args must be all strings.
  // TODO: kwargs and sanity check
  callPython: async function (func, args = []) {
    var message = {
      func: func,
      args: args,
    };
    this.jspy_send(JSON.stringify(message));
    var count = 0;
    while (!this.new_pyjs_message) {
      console.log("callPython await:", count++);
      await this.sleep(10);
    }
    console.log("callPython await done");
    this.new_pyjs_message = false;
    var message = this.model.get("pyjs_channel");
    return message;
  },

  callPythonWrapper: function (func, args = []) {
    return callPython(func, args).then((result) => result);
  },

  render: function () {
    console.log("Entered render() function.");
    this.new_pyjs_message = false;

    this.videoElt = document.createElement("video");
    this.videoElt.id = "video_tag";
    this.videoElt.muted = true;
    this.videoElt.controls = false;
    this.videoElt.playsinline = true;

    // The `el` property is the DOM element associated with the view
    this.el.appendChild(this.videoElt);

    // Listen for py->js message.
    this.model.on("change:pyjs_channel", this.on_pyjs_message, this);

    // Send js->py message for testing.
    this.callPython("call_http_request", [
      "my_entry_point",
      "my_query_string",
      "my_data",
    ]).then((result) => {
      console.log("callPython.then()", result);
    });
  },

  on_pyjs_message: function () {
    var message = this.model.get("pyjs_channel");
    console.log("pyjs_message received: " + message);
    this.new_pyjs_message = true;
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
