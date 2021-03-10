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
require('webrtc-adapter');
var WebRtcStreamer = require('./webrtcstreamer');

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
    this.videoElt.id = "video";
    this.videoElt.muted = true;
    this.videoElt.controls = false;
    this.videoElt.playsinline = true;

    // The `el` property is the DOM element associated with the view
    this.el.appendChild(this.videoElt);

    // Python -> JavaScript update
    this.model.on("change:value", this.value_changed, this);
  },

  value_changed: function () {
    console.log("value_changed");
    this.webRtcServer = new WebRtcStreamer(
      "video",
      "http://localhost:8888/"
    );
    this.webRtcServer.connect("image://Open3D");

    function getModifiers(event) {
      // See open3d/visualization/gui/Events.h.
      var modNone = 0;
      var modShift = 1 << 0;
      var modCtrl = 1 << 1;
      var modAlt = 1 << 2;
      var modMeta = 1 << 3;
      var mod = modNone;
      if (event.getModifierState("Shift")) {
        mod = mod | modShift;
      }
      if (event.getModifierState("Control")) {
        mod = mod | modCtrl;
      }
      if (event.getModifierState("Alt")) {
        mod = mod | modAlt;
      }
      if (event.getModifierState("Meta")) {
        mod = mod | modMeta;
      }
      return mod;
    }

    // Register callbacks for videoElt.
    var videoElt = document.getElementById("video");
    if (videoElt) {
      videoElt.addEventListener("mousedown", (event) => {
        var msg =
          "mousedown " + event.offsetX +
          " " +
          event.offsetY +
          " " +
          getModifiers(event);
        console.log(msg);
        this.webRtcServer.dataChannel.send(msg);
      });
      videoElt.addEventListener("mouseup", (event) => {
        var msg =
          "mouseup " +
          event.offsetX +
          " " +
          event.offsetY +
          " " +
          getModifiers(event);
        console.log(msg);
        this.webRtcServer.dataChannel.send(msg);
      });
      videoElt.addEventListener("mousemove", (event) => {
        var msg =
          "mousemove " +
          event.offsetX +
          " " +
          event.offsetY +
          " " +
          getModifiers(event);
        console.log(msg);
        this.webRtcServer.dataChannel.send(msg);
      });
      videoElt.addEventListener(
        "wheel",
        (event) => {
          // Prevent propagating the wheel event to the browser.
          // https://stackoverflow.com/a/23606063/1255535
          event.preventDefault();
          var msg =
            "wheel " +
            event.offsetX +
            " " +
            event.offsetY +
            " " +
            getModifiers(event) +
            " " +
            event.deltaX +
            " " +
            event.deltaY;
          console.log(msg);
          this.webRtcServer.dataChannel.send(msg);
        },
        { passive: false }
      );
    }
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView
};
