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
  // Render the view.
  render: function () {
    console.log("render");

    this.videoElt = document.createElement("video");
    this.videoElt.id = "video_tag";
    this.videoElt.muted = true;
    this.videoElt.controls = false;
    this.videoElt.playsinline = true;

    // The `el` property is the DOM element associated with the view
    this.el.appendChild(this.videoElt);

    // Create WebRTC stream
    this.webRtcClient = new WebRtcStreamer(
      this.videoElt,
      location.protocol + "//" + window.location.hostname + ":" + 8888,
      /*use_comms=*/ false
    );
    this.webRtcClient.connect(this.model.get("window_uid"));
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};

