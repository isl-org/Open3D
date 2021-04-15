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
    value: "Hello World!",
  }),
});

// Custom View. Renders the widget model.
var WebVisualizerView = widgets.DOMWidgetView.extend({
  onGetMediaList: function (mediaList) {
    console.log("!!!!onGetMediaList mediaList: ", mediaList);
  },

  /**
   * https://stackoverflow.com/a/52347011/1255535
   */
  executePython: function (python_code) {
    return new Promise((resolve, reject) => {
      var callbacks = {
        iopub: {
          output: (data) => resolve(data.content.text.trim()),
        },
      };
      Jupyter.notebook.kernel.execute(python_code, callbacks);
    });
  },

  /**
   * https://stackoverflow.com/a/736970/1255535
   * parseUrl(url).hostname
   * parseUrl(url).entryPoint
   */
  parseUrl: function (url) {
    var l = document.createElement("a");
    l.href = url;
    return l;
  },

  logAndReturn: function (value) {
    console.log("!!! logAndReturn: ", value);
    return value;
  },

  /**
   * Similar to WebRtcStreamer.remoteCall() but instead uses Jupyter's COMMS
   * interface.
   */
  commsCall: function (url, data = {}) {
    entryPoint = this.parseUrl(url).pathname;
    queryString = this.parseUrl(url).search;
    console.log("WebVisualizerView.commsCall with url: ", url, " data: ", data);
    console.log("WebVisualizerView.commsCall with entryPoint: ", entryPoint);
    console.log("WebVisualizerView.commsCall with queryString: ", queryString);
    console.log(
      'WebVisualizerView.commsCall with data["body"]: ',
      data["body"]
    );
    var command_prefix =
      "import open3d; print(open3d.visualization.webrtc_server.WebRTCServer.instance.call_http_request(";
    // entry_point
    var command_args = '"' + entryPoint + '"';
    // query_string
    if (queryString) {
      command_args += ', "' + queryString + '"';
    } else {
      command_args += ', ""';
    }
    // data
    var dataStr = data["body"];
    command_args += ", '" + dataStr + "'";
    var command_suffix = "))";
    var command = command_prefix + command_args + command_suffix;
    console.log("commsCall command: " + command);
    return this.executePython(command)
      .then((jsonStr) => JSON.parse(jsonStr))
      .then((val) => this.logAndReturn(val))
      .then(
        (jsonObj) =>
          new Response(
            new Blob([JSON.stringify(jsonObj)], {
              type: "application/json",
            })
          )
      )
      .then((val) => this.logAndReturn(val));
  },

  // Defines how the widget gets rendered into the DOM
  render: function () {
    this.value_changed();

    // Observe changes in the value traitlet in Python, and define
    // a custom callback.
    this.model.on("change:value", this.value_changed, this);
  },

  value_changed: function () {
    this.el.textContent = this.model.get("value");
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
