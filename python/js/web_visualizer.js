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
      "http://localhost:8888/"
    );
    this.webRtcClient.connect("image://Open3D");

    // Register callbacks for videoElt.
    this.videoElt.addEventListener("mousedown", (event) => {
      var open3dMouseEvent = {
        class_name: "MouseEvent",
        type: "BUTTON_DOWN",
        x: event.offsetX,
        y: event.offsetY,
        modifiers: getModifiers(event),
        button: {
          button: "LEFT", // Fix me.
          count: 1,
        },
      };
      this.webRtcClient.dataChannel.send(JSON.stringify(open3dMouseEvent));
    });
    this.videoElt.addEventListener("mouseup", (event) => {
      var open3dMouseEvent = {
        class_name: "MouseEvent",
        type: "BUTTON_UP",
        x: event.offsetX,
        y: event.offsetY,
        modifiers: getModifiers(event),
        button: {
          button: "LEFT", // Fix me.
          count: 1,
        },
      };
      this.webRtcClient.dataChannel.send(JSON.stringify(open3dMouseEvent));
    });
    this.videoElt.addEventListener("mousemove", (event) => {
      // TODO: Known differences. Currently only left-key drag works.
      // - Open3D: L=1, M=2, R=4
      // - JavaScript: L=1, R=2, M=4
      var open3dMouseEvent = {
        class_name: "MouseEvent",
        type: event.buttons == 0 ? "MOVE" : "DRAG",
        x: event.offsetX,
        y: event.offsetY,
        modifiers: getModifiers(event),
        move: {
          buttons: event.buttons, // MouseButtons ORed together
        },
      };
      this.webRtcClient.dataChannel.send(JSON.stringify(open3dMouseEvent));
    });
    this.videoElt.addEventListener("mouseleave", (event) => {
      var open3dMouseEvent = {
        class_name: "MouseEvent",
        type: "BUTTON_UP",
        x: event.offsetX,
        y: event.offsetY,
        modifiers: getModifiers(event),
        button: {
          button: "LEFT", // Fix me.
          count: 1,
        },
      };
      this.webRtcClient.dataChannel.send(JSON.stringify(open3dMouseEvent));
    });
    this.videoElt.addEventListener(
      "wheel",
      (event) => {
        // Prevent propagating the wheel event to the browser.
        // https://stackoverflow.com/a/23606063/1255535
        event.preventDefault();

        // https://stackoverflow.com/a/56948026/1255535.
        var isTrackpad = event.wheelDeltaY
          ? event.wheelDeltaY === -3 * event.deltaY
          : event.deltaMode === 0;

        // TODO: set better scaling.
        // Flip the sign and set abaolute value to 1.
        var dx = event.deltaX;
        var dy = event.deltaY;
        dx = dx == 0 ? dx : (-dx / Math.abs(dx)) * 1;
        dy = dy == 0 ? dy : (-dy / Math.abs(dy)) * 1;

        var open3dMouseEvent = {
          class_name: "MouseEvent",
          type: "WHEEL",
          x: event.offsetX,
          y: event.offsetY,
          modifiers: getModifiers(event),
          wheel: {
            dx: dx,
            dy: dy,
            isTrackpad: isTrackpad ? 1 : 0,
          },
        };
        this.webRtcClient.dataChannel.send(JSON.stringify(open3dMouseEvent));
      },
      { passive: false }
    );
  },
});

module.exports = {
  WebVisualizerModel: WebVisualizerModel,
  WebVisualizerView: WebVisualizerView,
};
