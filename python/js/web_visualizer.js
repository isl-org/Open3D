// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018 www.open3d.org
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
// IN THE SOFTWARE.
// ----------------------------------------------------------------------------

var widgets = require("@jupyter-widgets/base");
var path = require("path");
require("webrtc-adapter");
var WebRtcStreamer = require("./webrtcstreamer");

var WebVisualizerModel = widgets.DOMWidgetModel.extend({
  defaults: _.extend(widgets.DOMWidgetModel.prototype.defaults(), {
    _view_name: "WebVisualizerView",
    _view_module: "open3d",
    _view_module_version: "@PROJECT_VERSION_THREE_NUMBER@",
  }),
});

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
   * parseUrl(url).pathname
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
    api_url = this.parseUrl(url).pathname;
    console.log("WebVisualizerView.commsCall with api_url: ", api_url);
    console.log("WebVisualizerView.commsCall with url: ", url, " data: ", data);
    var command_prefix =
      "import open3d; print(open3d.visualization.webrtc_server.WebRTCServer.instance.call_web_request_api(";
    var command_suffix = "))";
    var command_args = '"' + api_url + '"';
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

  /**
   * Entry point for Jupyter widgets. Renders the view.
   */
  render: function () {
    console.log("render");

    this.videoElt = document.createElement("video");
    this.videoElt.id = "video_tag";
    this.videoElt.muted = true;
    this.videoElt.controls = false;
    this.videoElt.playsinline = true;

    // The `el` property is the DOM element associated with the view
    this.el.appendChild(this.videoElt);

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
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
