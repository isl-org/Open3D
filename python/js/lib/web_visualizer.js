// ----------------------------------------------------------------------------
// -                         Open3D: www.open3d.org                           -
// ----------------------------------------------------------------------------
// Copyright(c) 2018-2023 www.open3d.org
// SPDX - License - Identifier: MIT
// ----------------------------------------------------------------------------

// Jupyter widget for Open3D WebRTC visualizer. See web_visualizer.py for the
// kernel counterpart to this file.

let widgets = require("@jupyter-widgets/base");
let _ = require("lodash");
require("webrtc-adapter");
let WebRtcStreamer = require("./webrtcstreamer");

// Custom Model. Custom widgets models must at least provide default values
// for model attributes, including:
//  - _view_name
//  - _view_module
//  - _view_module_version
//  - _model_name
//  - _model_module
//  - _model_module_version
// when different from the base class.
//
// When serializing the entire widget state for embedding, only values that
// differ from the defaults will be specified.
class WebVisualizerModel extends widgets.DOMWidgetModel {
  defaults() {
    return _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
      _model_name: "WebVisualizerModel",
      _view_name: "WebVisualizerView",
      _model_module: "open3d",
      _view_module: "open3d",
      // @...@ is configured by cpp/pybind/make_python_package.cmake.
      _model_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
      _view_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
    });
  }
}

// Custom View. Renders the widget model.
class WebVisualizerView extends widgets.DOMWidgetView {
  sleep(time_ms) {
    return new Promise((resolve) => setTimeout(resolve, time_ms));
  }

  logAndReturn(value) {
    console.log("logAndReturn: ", value);
    return value;
  }

  callResultReady(callId) {
    let pyjs_channel = this.model.get("pyjs_channel");
    console.log("Current pyjs_channel:", pyjs_channel);
    let callResultMap = JSON.parse(this.model.get("pyjs_channel"));
    return callId in callResultMap;
  }

  extractCallResult(callId) {
    if (!this.callResultReady(callId)) {
      throw "extractCallResult not ready yet.";
    }
    let callResultMap = JSON.parse(this.model.get("pyjs_channel"));
    return callResultMap[callId];
  }

  /**
   * Hard-coded to call "call_http_api". Args and return value are all
   * strings.
   */
  async callPython(func, args = []) {
    let callId = this.callId.toString();
    this.callId++;
    let message = {
      func: func,
      args: args,
      call_id: callId,
    };

    // Append message to current jspy_channel.
    let jspyChannel = this.model.get("jspy_channel");
    let jspyChannelObj = JSON.parse(jspyChannel);
    jspyChannelObj[callId] = message;
    jspyChannel = JSON.stringify(jspyChannelObj);
    this.model.set("jspy_channel", jspyChannel);
    this.touch();

    let count = 0;
    while (!this.callResultReady(callId)) {
      console.log("callPython await, id: " + callId + ", count: " + count++);
      await this.sleep(100);
    }
    let json_result = this.extractCallResult(callId);
    console.log(
      "callPython await done, id:",
      callId,
      "json_result:",
      json_result
    );
    return json_result;
  }

  commsCall(url, data = {}) {
    // https://stackoverflow.com/a/736970/1255535
    // parseUrl(url).hostname
    // parseUrl(url).entryPoint
    // parseUrl(url).search
    let parseUrl = function (url) {
      let l = document.createElement("a");
      l.href = url;
      return l;
    };

    let entryPoint = parseUrl(url).pathname;
    let supportedAPI = [
      "/api/getMediaList",
      "/api/getIceServers",
      "/api/hangup",
      "/api/call",
      "/api/getIceCandidate",
      "/api/addIceCandidate",
    ];
    if (supportedAPI.indexOf(entryPoint) >= 0) {
      let queryString = parseUrl(url).search;
      if (!queryString) {
        queryString = "";
      }
      let dataStr = data["body"];
      if (!dataStr) {
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

      return this.callPython("call_http_api", [
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
      throw "Unsupported entryPoint: " + entryPoint;
    }
  }

  render() {
    let windowUID = this.model.get("window_uid");
    let onClose = function () {
      console.log("onClose() called for window_uid:", windowUID);
    };

    console.log("Entered render() function.");
    this.model.set("pyjs_channel", "{}");
    this.model.set("jspy_channel", "{}");
    this.touch();

    // Python call registry.
    this.callId = 0;

    this.videoElt = document.createElement("video");
    this.videoElt.id = "video_tag";
    this.videoElt.muted = true;
    this.videoElt.controls = false;
    this.videoElt.playsinline = true;
    this.videoElt.innerText = "Your browser does not support HTML5 video.";

    // this.el is the DOM element associated with the view.
    this.el.appendChild(this.videoElt);

    // Create WebRTC stream.
    this.webRtcClient = new WebRtcStreamer(
      this.videoElt,
      "",
      onClose,
      this.commsCall.bind(this)
    );
    this.webRtcClient.connect(windowUID);
  }
}

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
