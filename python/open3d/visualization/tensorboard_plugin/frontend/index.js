// ----------------------------------------------------------------------------
// -                        Open3D: www.open3d.org                            -
// ----------------------------------------------------------------------------
// The MIT License (MIT)
//
// Copyright (c) 2018-2021 www.open3d.org
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

import "./adapter.min.js";
import "./webrtcstreamer.js";

// [128, 128, 128, 255] -> "#808080"
function rgbToHex(rgb) {
  return '#' +
    (rgb[0] >= 16 ? '' : '0') + rgb[0].toString(16).slice(0,2) +
    (rgb[1] >= 16 ? '' : '0') + rgb[1].toString(16).slice(0,2) +
    (rgb[2] >= 16 ? '' : '0') + rgb[2].toString(16).slice(0,2);
}

function hexToRgb(hex) {
  if (hex[0] != '#') {
    console.error('Bad hex value:', hex);
    return [0, 0, 0];
  }
  const rgb = [parseInt(hex.slice(1, 3), 16),
    parseInt(hex.slice(3, 5), 16),
    parseInt(hex.slice(5, 7), 16)];
  return rgb;
}

function setColorTitle(evt) {
  const color = hexToRgb(evt.target.value);
  evt.target.title = `R:${color[0]}, G:${color[1]}, B:${color[2]}`;
}

/** Show / hide an element. Calling this function with a target id "toggle-ID"
 * will show / hide an element with id ID.
 */
function showHideDiv(evt) {
  let elt = document.getElementById(evt.currentTarget.id.slice(7));
  if (elt.style.display === "none") {
    elt.style.display = "block";
  } else {
    elt.style.display = "none";
  }
}

// Check if a tag (geometry series) contains a property as part of any map
// (point / vertex/ triangle)
function haveProperty(propertiesShapes, property) {
    for (const prefix of ['', 'vertex_', 'point_', 'line_', 'triangle_']) {
      if ((prefix + property) in propertiesShapes) {
        return prefix + property;
      }
    }
  return null;
}

class TensorboardOpen3DPluginClient {

  URL_ROUTE_PREFIX = "/data/plugin/Open3D";
  webRtcOptions = "rtptransport=tcp&timeout=60";
  full_width = 1280;   // video stream width, height
  full_height = 960;
  width = 640;
  height = 480;
  messageId = 0;
  webRtcClientList = new Map(); // {windowUId -> webRtcStreamer}
  windowState = new Map();     // {windowUId -> Geometry state (run, tags, batch_idx, step)}
  runWindow = new Map();  // {run -> windowUId}
  selectedTags = new Set();
  tagsPropertiesShapes = new Map();
  tagLabelsNames = new Map();
  COLORMAPS = null;     // Default colormaps. Get as response to get_run_tags message
  LabelLUTColors = null;   // Default LabelLUT colormap. Get as response to get_run_tags message
  renderState = new Map();  // {tag : {'property':__, 'index': 0, 'shader': __, 'colormap': []}}
  commonStep = null;
  commonBatchIdx = null;

  /**
   * Entry point for the TensorBoard Open3D plugin client
   * @constructor
   */
  constructor() {
    const DASHBOARD_HTML =
      `<link  href="style.css" rel="stylesheet">

            <div id="open3d-dashboard">
                <div id="options-selector">
                  <div class="sel-1button">
                      <h3>Options</h3>
                      <div class="selector">
                        <input type="checkbox" id="ui-options-view">
                        <label for="ui-options-view"> Sync view </label>
                        <input type="checkbox" id="ui-options-step">
                        <label for="ui-options-step"> Sync step </label>
                        <input type="checkbox" id="ui-options-bidx">
                        <label for="ui-options-bidx"> Sync batch index </label>
                        <input type="checkbox" id="ui-options-axes">
                        <label for="ui-options-axes"> Show axes </label>
                        <input type="checkbox" id="ui-options-ground">
                        <label for="ui-options-ground"> Show ground </label>
                      </div>
                    </div>

                    <h3>Runs</h3>
                    <div id="logdir" title="logdir"></div>
                    <div class="sel-1button" id="run-selector"></div>

                    <h3>Tags</h3>
                    <div class="sel-2button" id="tag-selector"></div>
                </div>

                <div id="widget-view"> </div>
            </div>
            `;
    document.body.insertAdjacentHTML("beforeend", DASHBOARD_HTML);
    const ICONS =
      `<svg style="display: none">
        <defs>
          <symbol id="zoom" viewBox="0 0 36 36" width="100%">
             <path d="m 10,16 2,0 0,-4 4,0 0,-2 L 10,10 l 0,6 0,0 z"></path>
             <path d="m 20,10 0,2 4,0 0,4 2,0 L 26,10 l -6,0 0,0 z"></path>
             <path d="m 24,24 -4,0 0,2 L 26,26 l 0,-6 -2,0 0,4 0,0 z"></path>
             <path d="M 12,20 10,20 10,26 l 6,0 0,-2 -4,0 0,-4 0,0 z"></path>
           </symbol>
           <symbol id="settings" viewBox="0 0 24 24">
             <path d="M19.43 12.98c.04-.32.07-.64.07-.98s-.03-.66-.07-.98l2.11-1.65c.19-.15.24-.42.12-.64l-2-3.46c-.12-.22-.39-.3-.61-.22l-2.49 1c-.52-.4-1.08-.73-1.69-.98l-.38-2.65C14.46 2.18 14.25 2 14 2h-4c-.25 0-.46.18-.49.42l-.38 2.65c-.61.25-1.17.59-1.69.98l-2.49-1c-.23-.09-.49 0-.61.22l-2 3.46c-.13.22-.07.49.12.64l2.11 1.65c-.04.32-.07.65-.07.98s.03.66.07.98l-2.11 1.65c-.19.15-.24.42-.12.64l2 3.46c.12.22.39.3.61.22l2.49-1c.52.4 1.08.73 1.69.98l.38 2.65c.03.24.24.42.49.42h4c.25 0 .46-.18.49-.42l.38-2.65c.61-.25 1.17-.59 1.69-.98l2.49 1c.23.09.49 0 .61-.22l2-3.46c.12-.22.07-.49-.12-.64l-2.11-1.65zM12 15.5c-1.93 0-3.5-1.57-3.5-3.5s1.57-3.5 3.5-3.5 3.5 1.57 3.5 3.5-1.57 3.5-3.5 3.5z"></path>
           </symbol>
         </defs>
      </svg>`
    document.body.insertAdjacentHTML("beforeend", ICONS);
    this.requestNewWindow();
  }

  /*  Request Open3D for a new window.
   *  @param {string} run Populate the window with data from this run. If
   *  undefined, the first run is chosen by default.
   */
  requestNewWindow = (run) => {
    // Ask Open3D for a new window
    console.info("Requesting window with size: (" + this.full_width +
      "," + this.full_height + ")");
    fetch(this.URL_ROUTE_PREFIX + "/new_window?width=" + this.full_width + "&height="
      + this.full_height, null)
      .then((response) => response.json())
      .then((response) => this.addConnection(response.window_id,
        response.logdir, run))
      .then(this.addAppEventListeners)
      .catch((err) => console.error("Error: /new_window failed:" + err));
  };

  /** Add App level event listeners.
  */
  addAppEventListeners = () => {
    window.addEventListener("beforeunload", this.closeWindow);
    // Listen for the user clicking on the main TB reload button
    let tbReloadButton =
      parent.document.querySelector(".reload-button");
    if (tbReloadButton != null) {
      tbReloadButton.addEventListener("click", this.reloadRunTags);
    }
    // App option handling
    document.getElementById("ui-options-view").addEventListener("change", (evt) => {
      if (evt.target.checked) {   // sync future viewpoint changes
        for (let [windowUId, webRtcClient] of this.webRtcClientList) {
          webRtcClient.sendJsonData = (jsonData) => {
            if (jsonData.class_name === "MouseEvent") {
              jsonData.class_name = "SyncMouseEvent";
              jsonData.window_uid_list = Array.from(
                this.webRtcClientList.keys());
            }
            webRtcClient.dataChannel.send(JSON.stringify(jsonData));
          }
        }
        this.messageId += 1;
        const syncViewMessage = {   // sync current viewpoint
          messageId: this.messageId,
          window_uid_list: Array.from(this.windowState.keys()),
          class_name: "tensorboard/sync_view",
        };
        console.info("Sending syncViewMessage:", syncViewMessage);
        this.webRtcClientList.values().next().value.sendJsonData(syncViewMessage);
      } else {
        for (let [windowUId, webRtcClient] of this.webRtcClientList) {
          webRtcClient.sendJsonData = WebRtcStreamer.prototype.sendJsonData;
        }
      }
    });

    document.getElementById("ui-options-step").addEventListener("change", (evt) => {
      if (evt.target.checked) {
        this.commonStep = this.windowState.values().next().value.step;
        for (let [windowUId, currentState] of this.windowState) {
          currentState.step = this.commonStep;
          this.requestGeometryUpdate(windowUId);
        }
      } else {
        this.commonStep = null;
      }
      console.debug("this.commonStep = ", this.commonStep);
    });

    document.getElementById("ui-options-bidx").addEventListener("change", (evt) => {
      if (evt.target.checked) {
        this.commonBatchIdx = this.windowState.values().next().value.batch_idx;
        for (let [windowUId, currentState] of this.windowState) {
          currentState.batch_idx = this.commonBatchIdx;
          this.requestGeometryUpdate(windowUId);
        }
      } else {
        this.commonBatchIdx = null;
      }
      console.debug("this.commonBatchIdx = ", this.commonBatchIdx);
    });

    document.getElementById("ui-options-axes").addEventListener("change", (evt) => {
      this.messageId += 1;
      const showHideAxesMessage = {
        messageId: this.messageId,
        window_uid_list: Array.from(this.windowState.keys()),
        class_name: "tensorboard/show_hide_axes",
        show: evt.target.checked
      };
      console.info("Sending showHideAxesMessage:", showHideAxesMessage);
      this.webRtcClientList.values().next().value.sendJsonData(showHideAxesMessage);
    });

    document.getElementById("ui-options-ground").addEventListener("change", (evt) => {
      this.messageId += 1;
      const showHideGroundMessage = {
        messageId: this.messageId,
        window_uid_list: Array.from(this.windowState.keys()),
        class_name: "tensorboard/show_hide_ground",
        show: evt.target.checked
      };
      console.info("Sending showHideGroundMessage:", showHideGroundMessage);
      this.webRtcClientList.values().next().value.sendJsonData(showHideGroundMessage);
    });
  };

  /**
   * Create a video element to display geometry and initiate WebRTC connection
   * with server. Attach listeners to process data channel messages.
   *
   * @param {string} windowUId  Window ID from the server, e.g.: "window_1"
   * @param {string} logdir TensorBoard log directory to be displayed to the
   *      user.
   * @param {string} run TB run to display here. Can be "undefined" in which
   *      case the server will assign a run.
   */
  addConnection = (windowUId, logdir, run) => {
    const videoId = "video_" + windowUId;
    let logdirElt = document.getElementById("logdir");
    logdirElt.innerText = logdir;

    // Add a video element to display WebRTC stream.
    const widgetTemplate = `
        <table class="webrtc" id="widget_${videoId}">
          <tr>
          <td><button id="zoom-${windowUId}" type="button"
            title="Make window larger / smaller">
             <svg width=36 height=36 ><use href="#zoom" /></svg>
          </button></td>
          <td class="batchidx-step-selector">
            <div id="batch-idx-selector-div-${windowUId}"></div>
            <div id="step-selector-div-${windowUId}"></div>
          </td>
          <td><button id="settings-${windowUId}" type="button"
            title="Show / hide settings">
             <svg width=24 height=24 ><use href="#settings" /></svg>
          </button></td>
          </tr>
          <tr><td colspan=3><div id="loader_${videoId}"></div>
          <video id="${videoId}" muted="true" playsinline="true"
            width=${this.width} height=${this.height}>
            Your browser does not support HTML5 video.
          </video>
          </td></tr>
        </table>
        `;
    let widgetView = document.getElementById("widget-view");
    widgetView.insertAdjacentHTML("beforeend", widgetTemplate);

    let videoElt = document.getElementById(videoId);
    let client = new WebRtcStreamer(videoElt, this.URL_ROUTE_PREFIX, null, null);
    console.info("[addConnection] videoId: " + videoId);

    client.connect(windowUId, /*audio*/ null, this.webRtcOptions);
    this.webRtcClientList.set(windowUId, client);
    console.info("[addConnection] windowUId: " + windowUId);
    videoElt.addEventListener("LocalDataChannelOpen", (evt) => {
      evt.detail.channel.addEventListener("message",
        this.processDCMessage.bind(this, windowUId));
    });
    if (this.webRtcClientList.size === 1) {
      // Initial Run Tag reload only needed for first window
      videoElt.addEventListener("RemoteDataChannelOpen", this.reloadRunTags);
    } else { // Initialize state from first webrtcWindow
      // deep copy: All objects and sub-objects need to be copied with JS
      // spread syntax.
      let newWindowState = {...this.windowState.values().next().value};
      newWindowState.tags = [...newWindowState.tags];
      this.windowState.set(windowUId, newWindowState);
      if (run) {
        this.runWindow.set(run, windowUId);
        this.windowState.get(windowUId).run = run;
        console.debug("After addConnection: this.runWindow = ",
          this.runWindow, "this.windowState = ", this.windowState);
      }
      videoElt.addEventListener("RemoteDataChannelOpen",
        this.requestGeometryUpdate.bind(this, windowUId));
    }
    document.getElementById("zoom-"+ windowUId).addEventListener(
      "click", this.toggleZoom.bind(this, windowUId));
    document.getElementById("settings-"+ windowUId).addEventListener(
      "click", this.toggleSettings.bind(this, windowUId));
  };

  /* Callback to toggle WebRTC window size in the browser.
  */
  toggleZoom = (windowUId) => {
    let elem = document.getElementById("video_" + windowUId);
    if (elem.width <= this.width) {        // zoom
      elem.width = this.full_width;
      elem.height = this.full_height;
    } else {                                // original
      elem.width = this.width;
      elem.height = this.height;
    }
    console.debug(`New ${windowUId} size: (${elem.width}, ${elem.height})`);
  };

  /* Callback to toggle O3DVisualizer settings panel.
  */
  toggleSettings = (windowUId) => {
    this.messageId += 1;
    const toggleSettingsMessage = {
      messageId: this.messageId,
      window_uid: windowUId,
      class_name: "tensorboard/" + windowUId + "/toggle_settings",
    };
    console.info("Sending toggleSettingsMessage:", toggleSettingsMessage);
    this.webRtcClientList.get(windowUId).sendJsonData(toggleSettingsMessage);
  };

  /**
   * Create generic checkbox and radio button selectors used for Run and Tag
   * selectors respectively.
   * @param {string} name Selector name attribute.
   * @param {string} parentId Id of parent element.
   * @param {array} options Array of string options to create
   * @param {string} type "checkbox" or "checkbox-button"
   * @param {array} initialCheckedOptions Array of options to be shown
   * selected initially.
   */
  createSelector(name, parentId, options, type, initialCheckedOptions) {
    console.assert(type === "checkbox" || type === "checkbox-button",
      "type must be checkbox or checkbox-button");

    let parentElement = document.getElementById(parentId);
    parentElement.replaceChildren();  /* Remove existing children */
    initialCheckedOptions = new Set(initialCheckedOptions);
    const selTypes = type.split('-');
    let BUTTON_TEMPLATE = "";
    let PROP_DIV_TEMPLATE = "";
    for (const option of options) {
      const checked = initialCheckedOptions.has(option) ? "checked" : "";
      if (selTypes.length==2) {
        BUTTON_TEMPLATE = `
              <button id="toggle-property-${option}" type="button" disabled>
                  <svg> <use href="#settings" /> </svg>
              </button>`
        PROP_DIV_TEMPLATE = `
          <div id="property-${option}" style="display: none;"></div>`;
      }
      const OPTION_TEMPLATE=`
        <div class="selector">
            <input type="${selTypes[0]}" id="${option}" name="${name}" ${checked}>
            ${BUTTON_TEMPLATE}
            <label for="${name}">${option}</label>
        </div>
        ${PROP_DIV_TEMPLATE}`;
      parentElement.insertAdjacentHTML("beforeend", OPTION_TEMPLATE);
      document.getElementById(option).addEventListener("change",
        this.onRunTagselect);
      if (selTypes.length==2) {
        document.getElementById(`toggle-property-${option}`).addEventListener(
          "click", showHideDiv);
      }
    }
  }

  /**
   * Create slider (range input element) for selecting a geometry in a batch,
   * or for selecting a step / iteration.
   * @param {int} windowUId Window ID associated with slider.
   * @param {string} name Selector name/id attribute.
   * @param {string} displayName Show this label in the UI.
   * @param {string} parentId Id of parent element.
   * @param {number} min Min value of range.
   * @param {number} max Max value of range.
   * @param {number} value Initial value of range.
   */
  createSlider(windowUId, name, displayName, parentId, min, max, value) {

    let parentElement = document.getElementById(parentId);
    parentElement.replaceChildren();  /* Remove existing children */
    const nameWid = name + "-" + windowUId;
    if (max > min) { // Don't create slider if no choice
      const sliderTemplate=`
            <form oninput="document.getElementById("${nameWid}_output").value
                = document.getElementById("${nameWid}").valueAsNumber;">
                <label> <span> ${displayName + ": [" + min.toString() + "-" +
                    max.toString() + "] "} </span>
                    <input type="range" id="${nameWid}" name="${nameWid}" min="${min}"
                        max="${max}" value="${value}">
                    </input>
                </label>
                <output for="${nameWid}" id="${nameWid + "_output"}"> ${value}
                </output>
            </form>
            `;
      parentElement.insertAdjacentHTML("beforeend", sliderTemplate);
      document.getElementById(nameWid).addEventListener("change",
        this.onStepBIdxSelect.bind(this, windowUId));
    }
  }

  createPropertyPanel = (tag) => {

    let tagPropEl = document.getElementById(`property-${tag}`);
    tagPropEl.replaceChildren();  // clean up
    const PROPERTY_TEMPLATE =
      `<label class="property-ui">Data
          <select name="property" id="ui-options-${tag}-property">
          </select>
      </label>
      <label class="property-ui">Index
          <input type="number" name="index" id="ui-options-${tag}-index" min="0" value="0">
      </label>
      <label class="property-ui">Shader
          <select name="shader" id="ui-options-${tag}-shader">
            <option value="unlitSolidColor">Solid Color</option>
            <option value="unlitGradient.LUT">Label Colormap</option>
            <option value="unlitGradient.GRADIENT.RAINBOW">Colormap (Rainbow)</option>
            <option value="unlitGradient.GRADIENT.GREYSCALE">Colormap (Greyscale)</option>
            <option value="defaultUnlit">RGB</option>
            <option value="defaultLit">With lighting</option>
          </select>
      </label>
      <div class="property-ui" id="ui-options-${tag}-colormap"> </div>
      <button class="property-ui" type="button" id="ui-render-${tag}-button">Update
      </button>`;
    // tagPropEl.replaceChildren();   // remove existing childen, if any
    tagPropEl.insertAdjacentHTML('beforeend', PROPERTY_TEMPLATE);

    this.tagsPropertiesShapes = new Map();
    for (const [windowUId, state] of this.windowState) {
      this.tagsPropertiesShapes = new Map([...this.tagsPropertiesShapes,
        ...Object.entries(state.tags_properties_shapes)]);
    }
    const propertiesShapes = this.tagsPropertiesShapes.get(tag);
    let propListEl = document.getElementById(`ui-options-${tag}-property`);
    let idxListEl = document.getElementById(`ui-options-${tag}-index`);
    let shaderListEl = document.getElementById(`ui-options-${tag}-shader`);
    let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
    let submitEl = document.getElementById(`ui-render-${tag}-button`);

    // Disable incompatible options
    /* shader : requiredProperties
     * Assume "positions" always present.
     * unlitSolidColor: ("positions",)
     * unlitGradient.LUT: ("labels", "positions")
     * unlitGradient.GRADIENT: ("positions",)
     * defaultUnlit: ("positions", "colors")
     * defaultLit: ("positions", "normals")
     */

    // unlitGradient.LUT
    const propLabels = haveProperty(propertiesShapes, 'labels');
    shaderListEl.children[1].hidden = (propLabels === null);
    // defaultlit
    const propNormals = haveProperty(propertiesShapes, 'normals');
    shaderListEl.children[5].hidden = (propNormals === null);
    const propPositions = haveProperty(propertiesShapes, 'positions');

    const renderStateTag = this.renderState.get(tag);
    shaderListEl.value = renderStateTag.shader;
    const selectedProperty = renderStateTag.property;

    for (const property of Object.keys(this.tagsPropertiesShapes.get(tag))) {
      const selected = (property === selectedProperty ? "selected" : "");
      propListEl.insertAdjacentHTML("beforeend",
        `<option ${selected}>${property}</option>`);
    }
    idxListEl.value = renderStateTag.index;
    idxListEl.max = propertiesShapes[selectedProperty]-1;
    propListEl.addEventListener('change', (evt) => {
      idxListEl.value = 0;
      idxListEl.max = this.tagsPropertiesShapes.get(tag)[evt.target.value]-1;
    });
    this.onShaderChanged(tag, /* onCreate=*/true);
    shaderListEl.addEventListener('change', this.onShaderChanged.bind(this, tag));
    submitEl.addEventListener('click', this.requestRenderUpdate.bind(this, tag));
    document.getElementById(`toggle-property-${tag}`).disabled = false;
  };

  onShaderChanged = (tag, onCreate) => {
    let shaderListEl = document.getElementById(`ui-options-${tag}-shader`);
    let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
    const renderStateTag = this.renderState.get(tag);

    cmapEl.replaceChildren();  // delete all children
    if (shaderListEl.value === "unlitSolidColor") {
      // this.colormap.set(tag, new Map([[0, [128, 128, 128, 255]]]));
      // renderStateTag.colormap = new Map(Object.entries(renderStateTag.colormap));
      let color = [128, 128, 128, 255];
      if (typeof onCreate != "undefined" && renderStateTag.colormap.size == 1) {
        color = renderStateTag.colormap.values().next().value;
      }
      cmapEl.innerHTML =
        `<label class="property-ui-colormap">
          <input type="checkbox" id="ui-cmap-${tag}-alpha-0" checked hidden>
          <input type="text" id="ui-cmap-${tag}-val-0" value="0" hidden>
          <input type="color" id="ui-cmap-${tag}-col-0"
            value="${rgbToHex(color)}"
            title="R:${color[0]} G:${color[1]} B:${color[2]}">
        </label>`;
        document.getElementById(`ui-cmap-${tag}-col-0`).addEventListener(
          "input", setColorTitle);
    } else if (shaderListEl.value === "unlitGradient.LUT") {
      // create LabelLUT
      if (this.tagLabelsNames.has(tag)) {
        const labelNames = this.tagLabelsNames.get(tag);
        if (typeof onCreate === "undefined") {
          renderStateTag.colormap = new Map(
            Object.entries(this.LabelLUTColors).slice(0, Object.entries(labelNames).length)
          );
        }
        let idx=0;
        for (const [label, name] of Object.entries(labelNames).sort(
        (l1, l2) => parseInt(l1) > parseInt(l2))) {
          const color = renderStateTag.colormap.get(label);
          const checked = color[3] > 0 ? "checked" : "";
          cmapEl.insertAdjacentHTML("beforeend",
            `<label class="property-ui-colormap">
              <input type="checkbox" id="ui-cmap-${tag}-alpha-${idx}" ${checked}>
              <input type="text" id="ui-cmap-${tag}-val-${idx}" value="${label}: ${name}" readonly>
              <input type="color" id="ui-cmap-${tag}-col-${idx}"
                title="R:${color[0]}, G:${color[1]}, B:${color[2]}"
                value="${rgbToHex(color)}">
            </label>`);
          document.getElementById(`ui-cmap-${tag}-col-${idx}`).addEventListener(
            "input", setColorTitle);
          idx = idx+1;
        }
      }
    } else if (shaderListEl.value.startsWith("unlitGradient.GRADIENT.")) {
        if (typeof onCreate === "undefined") {
          renderStateTag.colormap = new Map(
            Object.entries(this.COLORMAPS[shaderListEl.value.slice(23)]));
        }
      let currentRange = renderStateTag.range;
      cmapEl.insertAdjacentHTML("beforeend",
        `<p>Range: [${currentRange[0].toPrecision(4)}, ${currentRange[1].toPrecision(4)}]</p>`);
      if(!currentRange) { currentRange = [0.0, 1.0]; }
      const step = (currentRange[1] - currentRange[0])/16;
      let idx=0;
      for (const [value, color] of renderStateTag.colormap) {
        const valueFlt = currentRange[0] + parseFloat(value) *
          (currentRange[1] - currentRange[0]);
        const checked = color[3] > 0 ? "checked" : "";
        cmapEl.insertAdjacentHTML("beforeend",
          `<label class="property-ui-colormap">
            <input type="checkbox" id="ui-cmap-${tag}-alpha-${idx}" ${checked}>
            <input type="number" id="ui-cmap-${tag}-val-${idx}"
            value=${valueFlt.toPrecision(4)} step=${step} min=${currentRange[0]}
            max=${currentRange[1]}>
            <input type="color" id="ui-cmap-${tag}-col-${idx}"
              title="R:${color[0]}, G:${color[1]}, B:${color[2]}"
              value="${rgbToHex(color)}">
          </label>`);
        document.getElementById(`ui-cmap-${tag}-col-${idx}`).addEventListener(
          "input", setColorTitle);
        idx = idx+1;
      }
    } else if (shaderListEl.value === "defaultUnlit") {
      ;
    }
  };

  requestRenderUpdate = (tag) => {
    let propListEl = document.getElementById(`ui-options-${tag}-property`);
    let idxListEl = document.getElementById(`ui-options-${tag}-index`);
    let shaderListEl = document.getElementById(`ui-options-${tag}-shader`);
    let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);

    let updated = ['colormap'];
    let renderStateTag = this.renderState.get(tag);
    if (typeof renderStateTag == 'undefined') {
      renderStateTag = {
        property: '',
        index: 0,
        shader: '',
        colormap: null,
        range: [0.0, 1.0]
      };
    }
    if (propListEl.value != renderStateTag.property ||
      idxListEl.value != renderStateTag.index) {
      updated.push('property');
    }
    renderStateTag.property = propListEl.value;
    renderStateTag.index = idxListEl.value;
    if (shaderListEl.value != renderStateTag.shader) {
      updated.push('shader');
    }
    renderStateTag.shader = shaderListEl.value;
    const currentRange = renderStateTag["range"];
    let colormap_updated = false;
    let cmap = renderStateTag.colormap;
    let n_cols = 0;
    if (renderStateTag.shader == "unlitSolidColor") {
      n_cols = 1;
    } else if (renderStateTag.shader.startsWith("unlitGradient")) {
      n_cols = cmap.size;
    }
    cmap.clear();
    for (let idx=0; idx<n_cols; ++idx) {
      const alpha = document.getElementById(`ui-cmap-${tag}-alpha-${idx}`)
        .checked ? 255 : 0;
      let label_value = document.getElementById(`ui-cmap-${tag}-val-${idx}`)
        .value;
      if (label_value.includes(':')) {   // LabelLUT
        label_value = label_value.split(':')[0];
      } else {                          // Colormap
        label_value = (parseFloat(label_value) - currentRange[0]) /
          (currentRange[1]-currentRange[0]);
      }
      let color = hexToRgb(document.getElementById(`ui-cmap-${tag}-col-${idx}`).value);
      color.push(alpha);
      cmap.set(label_value, color);
    }

    this.messageId += 1;
    let updateRenderingMessage = {
      messageId: this.messageId,
      class_name: "tensorboard/update_rendering",
      window_uid_list: Array.from(this.windowState.keys()),
      tag: tag,
      render_state: JSON.parse(JSON.stringify(renderStateTag)),
      updated: updated
    };
    // Need colormap Object for JSON
    updateRenderingMessage.render_state.colormap = Object.fromEntries(renderStateTag.colormap.entries());
    console.log("Before update_rendering:", this.renderState);
    console.info("Sending updateRenderingMessage:", updateRenderingMessage);
    this.webRtcClientList.values().next().value.sendJsonData(updateRenderingMessage);
    // add busy indicator
    document.getElementById("loader_video_" + updateRenderingMessage.window_uid_list[0])
      .classList.add("loader");
  };

  /**
   * Event handler for browser tab / window close. Disconnect all server data
   * channel connections and close all server windows.
   */
  closeWindow = (evt) => {
    for (let [windowUId, webRtcClient] of this.webRtcClientList) {
      navigator.sendBeacon(this.URL_ROUTE_PREFIX + "/close_window?window_id=" +
        windowUId, null);
    }
    this.webRtcClientList.clear();
  };

  /**
   * Send a data channel message to the server to reload runs and tags from
   * the event file. Automatically run on loading.
   */
  reloadRunTags = () => {
    if (this.webRtcClientList.size > 0) {
      let client = this.webRtcClientList.values().next().value;
      // Any _open_ window_id may be used.
      const windowUId = this.webRtcClientList.keys().next().value;
      this.messageId += 1;
      const getRunTagsMessage = {
        messageId: this.messageId,
        window_id: windowUId,
        class_name: "tensorboard/" + windowUId + "/get_run_tags"
      };
      console.info("Sending getRunTagsMessage: ", getRunTagsMessage);
      client.dataChannel.send(JSON.stringify(getRunTagsMessage));
    } else {
      console.warn("No webRtcStreamer object initialized!");
    }
  };

  /**
   * Send a data channel message to the server to request an update to the
   * geometry display.
   */
  requestGeometryUpdate = (windowUId) => {
    this.messageId += 1;
    let updateGeometryMessage = JSON.parse(JSON.stringify({
      messageId: this.messageId,
      window_uid: windowUId,
      class_name: "tensorboard/" + windowUId + "/update_geometry",
      run: this.windowState.get(windowUId).run,
      tags: Array.from(this.selectedTags),
      render_state: Object.fromEntries(Array.from(this.renderState.entries())
        .filter(([tag, rs]) => this.selectedTags.has(tag))),
      batch_idx: this.commonBatchIdx || this.windowState.get(windowUId).batch_idx,
      step: this.commonStep || this.windowState.get(windowUId).step
    }));

    // Need colormap Object for JSON (not Map)
    for (const tag of this.selectedTags) {
      let rst = updateGeometryMessage.render_state[tag];
      if (typeof rst != "undefined")
        rst.colormap = Object.fromEntries(this.renderState.get(tag).colormap.entries());
    }
    console.log("Before update_geometry:", this.renderState);
    console.info("Sending updateGeometryMessage:", updateGeometryMessage);
    this.webRtcClientList.get(windowUId).sendJsonData(updateGeometryMessage);
    // add busy indicator
    document.getElementById("loader_video_" + windowUId).classList.add("loader");
  };

  /**
   * Event handler for Run and Tags selector update. Triggers a geometry
   * update message.
   */
  onRunTagselect = (evt) => {
    if (evt.target.name === "run-selector-checkboxes") {
      if (this.runWindow.has(evt.target.id)) {
        const windowUId = this.runWindow.get(evt.target.id);
        let windowWidget = document.getElementById("widget_video_" + windowUId);
        if (evt.target.checked) { // display window
          windowWidget.style.display = "flex";
          console.info("Showing window " + windowUId + " with run " + evt.target.id);
        } else {    // hide window
          windowWidget.style.display = "none";
          console.info("Hiding window " + windowUId + " with run " + evt.target.id);
        }
      } else {    // create new window
        this.requestNewWindow(evt.target.id);
      }
    } else if (evt.target.name === "tag-selector-checkboxes") {
      if (evt.target.checked) {
        this.selectedTags.add(evt.target.id);
      } else {
        this.selectedTags.delete(evt.target.id);
      }
      for (const windowUId of this.windowState.keys()) {
        this.requestGeometryUpdate(windowUId);
      }
      let tagPropButtonEl = document.getElementById(`toggle-property-${evt.target.id}`);
      tagPropButtonEl.disabled = !evt.target.checked;
      let tagPropEl = document.getElementById(`property-${evt.target.id}`);
      tagPropEl.hidden = !evt.target.checked;
    }
  };

  /**
   * Event handler for Step and Batch Index selector update. Triggers a
   * geometry update message.
   */
  onStepBIdxSelect = (windowUId, evt) => {
    console.debug("[onStepBIdxSelect] this.windowState: ",
      this.windowState, "this.commonStep", this.commonStep,
      "this.commonBatchIdx", this.commonBatchIdx);
    if (evt.target.name.startsWith("batch-idx-selector")) {
      if (this.commonBatchIdx != null) {
        this.commonBatchIdx = evt.target.value;
        for (const windowUId of this.windowState.keys()) {
          this.requestGeometryUpdate(windowUId);
        }
      } else {
        this.windowState.get(windowUId).batch_idx = evt.target.value;
        this.requestGeometryUpdate(windowUId);
      }
    } else if (evt.target.name.startsWith("step-selector")) {
      if (this.commonStep != null) {
        this.commonStep = evt.target.value;
        for (const windowUId of this.windowState.keys()) {
          this.requestGeometryUpdate(windowUId);
        }
      } else {
        this.windowState.get(windowUId).step = evt.target.value;
        this.requestGeometryUpdate(windowUId);
      }
    }
  };


  /**
   * Data channel message handler. Updates UI controls based on server state.
   */
  processDCMessage = (windowUId, evt) => {
    let message = null;
    try {
      message = JSON.parse(evt.data);
    } catch (err) {
      if (err.name === "SyntaxError") {
        if (evt.data.endsWith("DataChannel open")) {
          return;
        }
        if (evt.data.startsWith("[Open3D WARNING]")) {
          console.warn(evt.data);
          return;
        }
        console.error(err.name, err.message, evt.data);
      }
    }
    if (message.status != 'OK') {
      console.error(message.status);
    }
    if (message.class_name.endsWith("get_run_tags")) {
      const runToTags = message.run_to_tags;
      this.createSelector("run-selector-checkboxes", "run-selector",
        Object.getOwnPropertyNames(runToTags), "checkbox",
        [message.current.run]);
      this.selectedTags = new Set(message.current.tags);
      let allTags = new Set();
      for (const run in runToTags) {
        for (const tag of runToTags[run]) {
          allTags.add(tag);
        }
      }
      this.createSelector("tag-selector-checkboxes", "tag-selector",
        allTags, "checkbox-button", this.selectedTags);

      if (this.windowState.size === 0) { // First load
        this.runWindow.set(message.current.run, windowUId);
        this.windowState.set(windowUId, message.current);
        console.debug("[After get_run_tags] this.runWindow: ",
          this.runWindow, "this.windowState:", this.windowState);
        this.COLORMAPS = message.colormaps;  // Object (not Map)
        this.LabelLUTColors = message.LabelLUTColors;  // Object (not Map)
      }
      for (const [windowUId, state] of this.windowState) {
        if (runToTags.hasOwnProperty(state.run)) { // update window
          this.requestGeometryUpdate(windowUId);
        } else {   // stale run: close window
          fetch(this.URL_ROUTE_PREFIX +  "/close_window?window_id=" + windowUId, null)
            .then((response) => response.json())
            .then((response) => console.log(response))
            .then(this.runWindow.delete(state.run))
            .then(this.windowState.delete(windowUId))
            .then(this.webRtcClientList.delete(windowUId))
            .then(document.getElementById("widget_video_" + windowUId).remove())
            .catch((err) => console.error("Error closing widget:", err));
        }
      }
    } else if (message.class_name.endsWith("update_geometry")) {
      // Sync state with server
      console.assert(message.window_uid === windowUId,
        `windowUId mismatch: received ${message.window_uid} !== ${windowUId}`);
      this.windowState.set(windowUId, message.current);
      console.debug("[After update_geometry] this.runWindow: ",
        this.runWindow, "this.windowState:", this.windowState);
      // Update run level selectors
      this.createSlider(windowUId, "batch-idx-selector", "Batch index",
        "batch-idx-selector-div-" + windowUId, 0,
        message.current.batch_size - 1, message.current.batch_idx);
      this.createSlider(windowUId, "step-selector", "Step",
        "step-selector-div-" + windowUId, message.current.step_limits[0],
        message.current.step_limits[1], message.current.step);
      // Init with miliseconds
      const wallTime = new Date(message.current.wall_time * 1000);
      document.getElementById("video_" + windowUId).title = message.current.run
        + " at " + wallTime.toLocaleString();
      // Update app level selectors
      this.tagLabelsNames = new Map([...this.tagLabelsNames,
        ...Object.entries(message.tag_label_to_names)]);
      for (const tag of message.current.tags) {
        this.renderState.set(tag, message.current.render_state[tag]);
        const cmap = this.renderState.get(tag).colormap;  // Object
        this.renderState.get(tag).colormap = new Map(Object.entries(cmap));  // Map
        this.createPropertyPanel(tag);
      }
      console.log("After update_geometry:", this.renderState);
      // remove busy indicator
      document.getElementById("loader_video_" +
        windowUId).classList.remove("loader");
    } else if (message.class_name.endsWith("update_rendering")) {
      this.renderState.set(message.tag, message.render_state);
      const cmap = this.renderState.get(message.tag).colormap;  // Object
      this.renderState.get(message.tag).colormap = new Map(Object.entries(cmap));  // Map
      this.createPropertyPanel(message.tag);
      console.log("After update_rendering:", this.renderState);
      // remove busy indicator
      document.getElementById("loader_video_" + windowUId).classList.remove("loader");
    }
  };

}

/**
 * The main entry point of any TensorBoard iframe plugin.
 */
export async function render() {
  const o3dclient = new TensorboardOpen3DPluginClient();
}
