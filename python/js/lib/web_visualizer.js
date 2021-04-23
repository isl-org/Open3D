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
  commsCallLegacy: function (url, data = {}) {
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

  commsCall: function (url, data = {}) {
    var entryPoint = this.parseUrl(url).pathname;
    var supportedAPI = [
      "/api/getMediaList",
      "/api/getIceServers",
      // "/api/hangup",
      // "/api/call",
      // "/api/getIceCandidate",
      // "/api/addIceCandidate",
    ];
    if (supportedAPI.indexOf(entryPoint) >= 0) {
      var queryString = this.parseUrl(url).search;
      if (!queryString) {
        queryString = "";
      }
      var dataStr = data["body"];
      if (!queryString) {
        dataStr = "";
      }

      console.log(
        "WebVisualizerView.commsCall with url: ",
        url,
        " data: ",
        data
      );
      console.log("WebVisualizerView.commsCall with entryPoint: ", entryPoint);
      console.log(
        "WebVisualizerView.commsCall with queryString: ",
        queryString
      );
      console.log('WebVisualizerView.commsCall with data["body"]: ', dataStr);

      return this.callPython("call_http_request", [
        entryPoint,
        queryString,
        dataStr,
      ])
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
    } else {
      return this.commsCallLegacy(url, data);
    }
  },

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
    var callId = this.callId.toString();
    this.callId++;
    var message = {
      func: func,
      args: args,
      call_id: callId,
    };
    this.jspy_send(JSON.stringify(message));
    var count = 0;
    while (!(callId in this.callResultMap)) {
      console.log("callPython await, id: " + callId + ", count: " + count++);
      await this.sleep(100);
    }
    var json_result = this.callResultMap[callId];
    delete this.callResultMap[callId];
    console.log(
      "callPython await done, id:",
      callId,
      "json_result:",
      json_result
    );
    return json_result;
  },

  callPythonWrapper: function (func, args = []) {
    return callPython(func, args).then((result) => result);
  },

  render: function () {
    console.log("Entered render() function.");

    // Python call registry
    this.callId = 0;
    this.callResultMap = {};
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
    // this.callPython("call_http_request", [
    //   "my_entry_point",
    //   "my_query_string",
    //   "my_data",
    // ]).then((result) => {
    //   console.log("callPython.then()", result);
    // });

    // TODO: remove this after switching to purely comms-based communication.
    var http_server =
      location.protocol + "//" + window.location.hostname + ":" + 8888;

    // TODO: remove this since the media name should be given by Python
    // directly. This is only used for developing the pipe.
    WebRtcStreamer.remoteCall(http_server + "/api/getMediaList", true, {}, this)
      .then((response) => response.json())
      .then((jsonObj) => this.onGetMediaList(jsonObj));

    // Create WebRTC stream
    this.webRtcClient = new WebRtcStreamer(
      this.videoElt,
      location.protocol + "//" + window.location.hostname + ":" + 8888,
      /*useComms(when supported)=*/ true,
      /*webVisualizer=*/ this
    );
    this.webRtcClient.connect(this.model.get("window_uid"));
  },

  on_pyjs_message: function () {
    var message = this.model.get("pyjs_channel");
    console.log("pyjs_message received: " + message);

    var result_obj = JSON.parse(message);
    var call_id = result_obj["call_id"];
    var json_result = result_obj["json_result"];
    this.callResultMap[call_id] = json_result;
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
