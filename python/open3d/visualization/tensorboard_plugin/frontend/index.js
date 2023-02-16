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

import './adapter.min.js';
import './webrtcstreamer.js';

/* [128, 128, 128, 255] -> "#808080"
 * @param {Array[Number]} rgb RGB or RGBA UInt8 color
 * @return {String} Hex color (no alpha)
 */
function rgbToHex(rgb) {
    return '#' + (rgb[0] >= 16 ? '' : '0') + rgb[0].toString(16).slice(0, 2) +
            (rgb[1] >= 16 ? '' : '0') + rgb[1].toString(16).slice(0, 2) +
            (rgb[2] >= 16 ? '' : '0') + rgb[2].toString(16).slice(0, 2);
}

/* "#808080" -> [128, 128, 128]
 * @param {String} Hex color
 * @return {Array[Number]} rgb RGB UInt8 color
 */
function hexToRgb(hex) {
    if (hex[0] !== '#') {
        console.error('Bad hex value:', hex);
        return [0, 0, 0];
    }
    const rgb = [
        parseInt(hex.slice(1, 3), 16), parseInt(hex.slice(3, 5), 16),
        parseInt(hex.slice(5, 7), 16)
    ];
    return rgb;
}

/* Set tooltip for color selector.
 * @param {Event} evt
 * @listens document#change
 */
function setColorTitle(evt) {
    const color = hexToRgb(evt.target.value);
    evt.target.title = `R:${color[0]}, G:${color[1]}, B:${color[2]}`;
}

/**
 * Show / hide an element. Calling this function with a target id "toggle-ID"
 * will show / hide an element with id ID.
 * @param {MouseEvent} evt
 * @listens document#click
 */
function showHideDiv(evt) {
    let elt = document.getElementById(evt.currentTarget.id.slice(7));
    elt.style.display = (elt.style.display === 'none') ? 'block' : 'none';
}

/**
 * User interface for Open3D for TensorBoard.
 */
class TensorboardOpen3DPluginClient {
    /** @const {String} prefix for HTTP request URLs */
    URL_ROUTE_PREFIX = '/data/plugin/Open3D';
    /** @const {String} Options for WebRTC session */
    webRtcOptions = 'rtptransport=tcp&timeout=60';
    /** @const {Number} WebRTC video stream width */
    fullWidth = 1280;
    /** @const {Number} WebRTC video stream height */
    fullHeight = 960;
    /** @const {Number} Small widget width */
    width = 640;
    /** @const {Number} Small widget height */
    height = 480;
    /**
     * @var {Number} Sequentially increasing message number sent to the server.
     */
    messageId = 0;
    /**
     * @var {Map} windowUId {String} -> {webRtcStreamer}. The windowUId is a
     * unique identifier obtained form the Open3D server.
     */
    webRtcClientList = new Map();
    /** @var {Map} windowUId -> Geometry state (run, tags, batch_idx, step) */
    windowState = new Map();
    /** @var {Map} run -> windowUId */
    runWindow = new Map();
    /** @var {Set}  Currently selected tags */
    selectedTags = new Set();
    /**
     * @var {Map} tag -> Object
     *                  {geometry or custom property {String}: shape {Array}}
     */
    tagsPropertiesShapes = new Map();
    /** @var {Map} tag -> Object {class label {Number}: class name {String} }*/
    tagLabelsNames = new Map();
    /**
     * @var {Object} Default colormaps (currently RAINBOW and GRAYSCALE). Get
     * as response to "get_run_tags" message.
     */
    COLORMAPS = null;
    /**
     * @var {Array} Default LabelLUT colormap. Get as response to
     * "get_run_tags" message
     */
    labelLUTColors = null;
    /**
     * @var {Map} Current rendering state for a tag.
     * tag -> Object {"property":__, "index": 0, "shader": __, "colormap": []}}
     */
    renderState = new Map();
    /**
     * @var {Number} Common step for all runs, if they are in sync. Else null.
     */
    commonStep = null;
    /**
     * @var {Number} Common batch index for all runs, if they are in sync. Else
     *            null.
     */
    commonBatchIdx = null;
    /**
     * @var {Array} List of valid shaders, based on the selected geometry
     *              property.
     */
    validShaders = [
        'unlitSolidColor', 'unlitGradient.GRADIENT.RAINBOW',
        'unlitGradient.GRADIENT.GREYSCALE', 'defaultUnlit', 'defaultLit'
    ];

    /**
     * Entry point for the TensorBoard Open3D plugin client
     * @constructor
     */
    constructor() {
        /** @const {String} HTML for the document structure */
        const DASHBOARD_HTML = `<link  href="style.css" rel="stylesheet">
            <div id="open3d-dashboard">
                <div id="options-selector">
                  <div class="sel-1button">
                      <h3>Options</h3>
                      <div class="selector">
                        <input type="checkbox" id="ui-options-view">
                        <label for="ui-options-view" title=
                        "Show all current runs from the same view point"> Sync view </label>
                        <input type="checkbox" id="ui-options-step">
                        <label for="ui-options-step" title=
                        "Use a common step for all runs"> Sync step </label>
                        <input type="checkbox" id="ui-options-bidx">
                        <label for="ui-options-bidx" title=
                        "Use a common batch index for all runs"> Sync batch index </label>
                        <input type="checkbox" id="ui-options-axes">
                        <label for="ui-options-axes" title=
                        "Show coordinate axes X (red), Y (green), Z (blue) in all current runs">
                        Show axes </label>
                        <input type="checkbox" id="ui-options-ground">
                        <label for="ui-options-ground" title=
                        "Show  a ground plane grid. Change the plane assigned as
ground from the Open3D GUI settings for each widget."> Show ground </label>
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
        document.body.insertAdjacentHTML('beforeend', DASHBOARD_HTML);
        /**
         * @const {String} SVG icons are inserted in the DOM here and used as
         *                 required later.
         */
        const ICONS = `<svg style="display: none">
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
      </svg>`;
        document.body.insertAdjacentHTML('beforeend', ICONS);
        this.requestNewWindow();
    }

    /**
     * Request Open3D server for a new window.
     *  @param {String} run Populate the window with data from this run. If
     *                      undefined, the first run is chosen by default.
     */
    requestNewWindow = (run) => {
        console.info(
                'Requesting window with size: (' + this.fullWidth + ',' +
                this.fullHeight + ')');
        fetch(this.URL_ROUTE_PREFIX + '/new_window?width=' + this.fullWidth +
                      '&height=' + this.fullHeight,
              null)
                .then((response) => response.json())
                .then((response) => this.addConnection(
                              response.window_id, response.logdir, run))
                .then(this.addAppEventListeners)
                .catch((err) => console.error(
                               'Error: /new_window failed:' + err));
    };

    /**
     * Send message to Open3D server to toggle synchronized views for all
     * currently open run widgets. To synchronize run widgets opened after
     * wards, the user has to repeat this step.
     * @callback
     * @param {Event} evt
     * @listens document#change
     */
    onSyncView = (evt) => {
        if (evt.target.checked) {  // sync future viewpoint changes
            for (let [windowUId, webRtcClient] of this.webRtcClientList) {
                webRtcClient.sendJsonData = (jsonData) => {
                    if (jsonData.class_name === 'MouseEvent') {
                        jsonData.class_name = 'SyncMouseEvent';
                        jsonData.window_uid_list =
                                Array.from(this.webRtcClientList.keys());
                    }
                    webRtcClient.dataChannel.send(JSON.stringify(jsonData));
                };
            }
            this.messageId += 1;
            // sync current viewpoint
            const syncViewMessage = {
                messageId: this.messageId,
                window_uid_list: Array.from(this.windowState.keys()),
                class_name: 'tensorboard/sync_view',
            };
            console.info('Sending syncViewMessage:', syncViewMessage);
            this.webRtcClientList.values().next().value.sendJsonData(
                    syncViewMessage);
        } else {
            for (let [windowUId, webRtcClient] of this.webRtcClientList) {
                webRtcClient.sendJsonData =
                        WebRtcStreamer.prototype.sendJsonData;
            }
        }
    };

    /**
     * Synchronize step / epoch for all displayed runs (currently open as well
     * as future).
     * @callback
     * @param {Event} evt
     * @listens document#change
     */
    onSyncStep = (evt) => {
        if (evt.target.checked) {
            this.commonStep = this.windowState.values().next().value.step;
            for (let [windowUId, currentState] of this.windowState) {
                currentState.step = this.commonStep;
                this.requestGeometryUpdate(windowUId);
            }
        } else {
            this.commonStep = null;
        }
        console.debug('this.commonStep = ', this.commonStep);
    };

    /**
     * Synchronize batch index for all displayed runs (currently open as well
     * as future).
     * @callback
     * @param {Event} evt
     * @listens document#change
     */
    onSyncBIdx = (evt) => {
        if (evt.target.checked) {
            this.commonBatchIdx =
                    this.windowState.values().next().value.batch_idx;
            for (let [windowUId, currentState] of this.windowState) {
                currentState.batch_idx = this.commonBatchIdx;
                this.requestGeometryUpdate(windowUId);
            }
        } else {
            this.commonBatchIdx = null;
        }
        console.debug('this.commonBatchIdx = ', this.commonBatchIdx);
    };

    /**
     * Send message to the Open3D server to toggle show / hide axes in all
     * currently open windows.
     * @callback
     * @param {Event} evt
     * @listens document#change
     */
    onToggleAxes = (evt) => {
        this.messageId += 1;
        const showHideAxesMessage = {
            messageId: this.messageId,
            window_uid_list: Array.from(this.windowState.keys()),
            class_name: 'tensorboard/show_hide_axes',
            show: evt.target.checked
        };
        console.info('Sending showHideAxesMessage:', showHideAxesMessage);
        this.webRtcClientList.values().next().value.sendJsonData(
                showHideAxesMessage);
    };

    /**
     * Send message to the Open3D server to toggle show / hide the procedural
     * ground plane grid in all currently open windows.
     * @callback
     * @param {Event} evt
     * @listens document#change
     */
    onToggleGround = (evt) => {
        this.messageId += 1;
        const showHideGroundMessage = {
            messageId: this.messageId,
            window_uid_list: Array.from(this.windowState.keys()),
            class_name: 'tensorboard/show_hide_ground',
            show: evt.target.checked
        };
        console.info('Sending showHideGroundMessage:', showHideGroundMessage);
        this.webRtcClientList.values().next().value.sendJsonData(
                showHideGroundMessage);
    };


    /**
     * Add App level event listeners for UI options.
     */
    addAppEventListeners = () => {
        window.addEventListener('beforeunload', this.closeWindow);
        // Listen for the user clicking on the main TB reload button
        let tbReloadButton = parent.document.querySelector('.reload-button');
        if (tbReloadButton != null) {
            tbReloadButton.addEventListener('click', this.reloadRunTags);
        }
        document.getElementById('ui-options-view')
                .addEventListener('change', this.onSyncView);
        document.getElementById('ui-options-step')
                .addEventListener('change', this.onSyncStep);
        document.getElementById('ui-options-bidx')
                .addEventListener('change', this.onSyncBIdx);
        document.getElementById('ui-options-axes')
                .addEventListener('change', this.onToggleAxes);
        document.getElementById('ui-options-ground')
                .addEventListener('change', this.onToggleGround);
    };

    /**
     * Show message asking the user to reload the page if network connection is
     * lost.
     */
    onCloseWebRTC = () => {
        let widgetView = document.getElementById('widget-view');
        widgetView.insertAdjacentHTML('afterbegin', `
            <div class="no-webrtc">
                <h3>Network connection to Open3D server lost.</h3>
                <p>If TensorBoard is still running, please reload the page.</p>
            </div>`);
    };

    /**
     * Create a video element to display geometry and initiate WebRTC connection
     * with server. Attach listeners to process data channel messages.
     *
     * @param {String} windowUId  Window ID from the server, e.g.: "window_1"
     * @param {String} logdir TensorBoard log directory to be displayed to the
     *                        user.
     * @param {String} run TB run to display here. Can be "undefined" in which
     *                     case the server will assign a run.
     */
    addConnection = (windowUId, logdir, run) => {
        if (windowUId === -1) {
            let widgetView = document.getElementById('widget-view');
            widgetView.insertAdjacentHTML('beforeend', `
                <div class="no-data-warning">
                <h3>No 3D data was found.</h3>
                <p>Probable causes:</p>
                <ul>
                  <li>You haven’t written any 3D data to your event files.</li>
                  <li>TensorBoard can’t find your event files.</li>
                </ul>

                <p>
                  If you’re new to using TensorBoard, and want to find out how to
                  add data and set up your event files, check out the
                  <a href="https://github.com/tensorflow/tensorboard/blob/master/README.md">README</a>
                  and perhaps the
                  <a href="https://www.tensorflow.org/get_started/summaries_and_tensorboard">TensorBoard tutorial</a>.
                </p>

                <p>
                  If you think TensorBoard is configured properly, please see
                  <a href="https://github.com/tensorflow/tensorboard/blob/master/README.md#my-tensorboard-isnt-showing-any-data-whats-wrong">the section of the README devoted to missing data problems</a>
                  and consider filing an issue on GitHub.
                </p>
              </div>`);
            return;
        }
        const videoId = 'video_' + windowUId;
        let logdirElt = document.getElementById('logdir');
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
        let widgetView = document.getElementById('widget-view');
        widgetView.insertAdjacentHTML('beforeend', widgetTemplate);

        let videoElt = document.getElementById(videoId);
        let client = new WebRtcStreamer(
                videoElt, this.URL_ROUTE_PREFIX, this.onCloseWebRTC, null);
        console.info('[addConnection] videoId: ' + videoId);

        client.connect(windowUId, /*audio*/ null, this.webRtcOptions);
        this.webRtcClientList.set(windowUId, client);
        console.info('[addConnection] windowUId: ' + windowUId);
        videoElt.addEventListener('LocalDataChannelOpen', (evt) => {
            evt.detail.channel.addEventListener(
                    'message', this.processDCMessage.bind(this, windowUId));
        });
        if (this.webRtcClientList.size === 1) {
            // Initial Run Tag reload only needed for first window
            videoElt.addEventListener(
                    'RemoteDataChannelOpen', this.reloadRunTags);
        } else {  // Initialize state from first webrtcWindow
            // deep copy: All objects and sub-objects need to be copied with JS
            // spread syntax.
            let newWindowState = {...this.windowState.values().next().value};
            newWindowState.tags = [...newWindowState.tags];
            this.windowState.set(windowUId, newWindowState);
            if (run) {
                this.runWindow.set(run, windowUId);
                this.windowState.get(windowUId).run = run;
            }
            videoElt.addEventListener(
                    'RemoteDataChannelOpen',
                    this.requestGeometryUpdate.bind(this, windowUId));
        }
        videoElt.settingsOpen = false;
        document.getElementById('zoom-' + windowUId)
                .addEventListener(
                        'click', this.toggleZoom.bind(this, windowUId));
        document.getElementById('settings-' + windowUId)
                .addEventListener(
                        'click', this.toggleSettings.bind(this, windowUId));
    };

    /**
     * Toggle WebRTC widget size in the browser.
     * @callback
     * @param {String} windowUId e.g. "window_2"
     */
    toggleZoom = (windowUId) => {
        let videoElt = document.getElementById('video_' + windowUId);
        if (videoElt.width <= this.width) {  // zoom
            videoElt.width = this.fullWidth;
            videoElt.height = this.fullHeight;
        } else {  // original
            videoElt.width = this.width;
            videoElt.height = this.height;
            if (videoElt.settingsOpen) {  // close settings panel
                this.toggleSettings(windowUId);
            }
        }
        console.debug(`New ${windowUId} size: (${videoElt.width}, ${
                videoElt.height})`);
    };

    /**
     * Send message to toggle O3DVisualizer settings panel.
     * @callback
     * @param {String} windowUId e.g. "window_2"
     */
    toggleSettings = (windowUId) => {
        this.messageId += 1;
        const toggleSettingsMessage = {
            messageId: this.messageId,
            window_uid: windowUId,
            class_name: 'tensorboard/' + windowUId + '/toggle_settings',
        };
        console.info('Sending toggleSettingsMessage:', toggleSettingsMessage);
        this.webRtcClientList.get(windowUId).sendJsonData(
                toggleSettingsMessage);
    };

    /**
     * Create generic checkbox and radio button selectors used for Run and Tag
     * selectors respectively.
     * @param {String} name     Selector name attribute.
     * @param {String} parentId Id of parent element.
     * @param {Array} options   Array of string options to create
     * @param {String} type     "checkbox" or "checkbox-button"
     * @param {Array} initialCheckedOptions Array of options to be shown
     *                          selected initially.
     */
    createSelector(name, parentId, options, type, initialCheckedOptions) {
        console.assert(
                type === 'checkbox' || type === 'checkbox-button',
                'type must be checkbox or checkbox-button');

        let parentElement = document.getElementById(parentId);
        parentElement.replaceChildren(); /* Remove existing children */
        initialCheckedOptions = new Set(initialCheckedOptions);
        const selTypes = type.split('-');
        let BUTTON_TEMPLATE = '';
        let PROP_DIV_TEMPLATE = '';
        for (const option of options) {
            const checked = initialCheckedOptions.has(option) ? 'checked' : '';
            if (selTypes.length === 2) {
                BUTTON_TEMPLATE = `
              <button id="toggle-property-${option}" type="button" disabled
              title="Change display properties, including selecting data and adjusting colors.">
                  <svg> <use href="#settings" /> </svg>
              </button>`
                PROP_DIV_TEMPLATE = `
          <div id="property-${option}" style="display: none;"></div>`;
            }
            const OPTION_TEMPLATE = `
        <div class="selector">
            <input type="${selTypes[0]}" id="${option}" name="${name}" ${
                    checked}>
            ${BUTTON_TEMPLATE}
            <label for="${name}">${option}</label>
        </div>
        ${PROP_DIV_TEMPLATE}`;
            parentElement.insertAdjacentHTML('beforeend', OPTION_TEMPLATE);
            document.getElementById(option).addEventListener(
                    'change', this.onRunTagselect);
            if (selTypes.length == 2) {
                document.getElementById(`toggle-property-${option}`)
                        .addEventListener('click', showHideDiv);
            }
        }
    }

    /**
     * Create slider (range input element) for selecting a geometry in a batch,
     * or for selecting a step / iteration.
     * @param {Number} windowUId    Window ID associated with slider.
     * @param {String} name         Selector name/id attribute.
     * @param {String} displayName  Show this label in the UI.
     * @param {String} parentId     Id of parent element.
     * @param {Number} min          Min value of range.
     * @param {Number} max          Max value of range.
     * @param {Number} value        Initial value of range.
     */
    createSlider(windowUId, name, displayName, parentId, min, max, value) {
        let parentElement = document.getElementById(parentId);
        parentElement.replaceChildren(); /* Remove existing children */
        const nameWid = name + '-' + windowUId;
        if (max > min) {  // Don"t create slider if no choice
            const sliderTemplate = `
            <form oninput="document.getElementById("${nameWid}_output").value
                = document.getElementById("${nameWid}").valueAsNumber;">
                <label> <span> ${
                    displayName + ': [' + min.toString() + '-' +
                    max.toString() + '] '} </span>
                    <input type="range" id="${nameWid}" name="${
                    nameWid}" min="${min}"
                        max="${max}" value="${value}">
                    </input>
                </label>
                <output for="${nameWid}" id="${nameWid + '_output'}"> ${value}
                </output>
            </form>
            `;
            parentElement.insertAdjacentHTML('beforeend', sliderTemplate);
            document.getElementById(nameWid).addEventListener(
                    'change', this.onStepBIdxSelect.bind(this, windowUId));
        }
    }

    /**
     * Create settings panel for a tag. Any existing HTML is removed.
     * @param {String} tag
     */
    createPropertyPanel = (tag) => {
        let tagPropEl = document.getElementById(`property-${tag}`);
        tagPropEl.replaceChildren();  // clean up
        const propertiesShapes = this.tagsPropertiesShapes.get(tag);
        if (Object.keys(propertiesShapes).length === 0) {
            return;
        }
        const PROPERTY_TEMPLATE = `<label class="property-ui">Data
          <select name="property" id="ui-options-${tag}-property">
            <option value="">Geometry</option>
          </select>
      </label>
      <label class="property-ui">Index
          <input type="number" name="index" id="ui-options-${
                tag}-index" min="0" value="0">
      </label>
      <label class="property-ui">Shader
          <select name="shader" id="ui-options-${tag}-shader">
            <option value="unlitSolidColor" title="Uniform color for the entire geometry."
            >Solid Color</option>
            <option value="unlitGradient.LUT"
            title="Colors based on discrete labels for points or bounding boxes."
            >Label Colormap</option>
            <option value="unlitGradient.GRADIENT.RAINBOW"
            title="Render a scalar point property in full color"
            >Colormap (Rainbow)</option>
            <option value="unlitGradient.GRADIENT.GREYSCALE"
            title="Render a scalar point property in grayscale."
            >Colormap (Greyscale)</option>
            <option value="defaultUnlit"
            title="Use colors or another 3 element property of the points as RGB.\nOnly the first 3 dimensions of a custom property are used."
            >RGB</option>
            <option value="defaultLit"
            title="Default geometry visualization."
            >Default</option>
          </select>
      </label>
      <div class="property-ui" id="ui-options-${tag}-colormap"> </div>
      `;
        tagPropEl.insertAdjacentHTML('beforeend', PROPERTY_TEMPLATE);

        let propListEl = document.getElementById(`ui-options-${tag}-property`);
        let idxListEl = document.getElementById(`ui-options-${tag}-index`);
        let shaderListEl = document.getElementById(`ui-options-${tag}-shader`);

        const renderStateTag = this.renderState.get(tag);
        shaderListEl.value = renderStateTag.shader;
        const selectedProperty = renderStateTag.property;

        for (const property of Object.keys(
                     this.tagsPropertiesShapes.get(tag))) {
            const selected = (property === selectedProperty ? 'selected' : '');
            propListEl.insertAdjacentHTML(
                    'beforeend', `<option ${selected}>${property}</option>`);
        }
        idxListEl.max = propertiesShapes[selectedProperty] - 1;
        idxListEl.value = renderStateTag.index;
        idxListEl.disabled = (idxListEl.max === 0);

        propListEl.addEventListener(
                'change',
                this.onPropertyChanged.bind(this, tag, /* onCreate=*/ false));
        idxListEl.addEventListener(
                'change', this.requestRenderUpdate.bind(this, tag));
        shaderListEl.addEventListener(
                'change',
                this.onShaderChanged.bind(this, tag, /* onCreate=*/ false));
        document.getElementById(`toggle-property-${tag}`).disabled = false;
        this.onPropertyChanged(tag, /* onCreate=*/ true);
    };

    /**
     * Populate the options list with valid shaders that the user can select for
     * a pre-selected geometry / custom property.
     * @param {String} tag
     * @param {Array}  shaderList  List of valid shaders
     */
    setEnabledShaders = (tag, shaderList) => {
        const shaderChild = [
            'unlitSolidColor', 'unlitGradient.LUT',
            'unlitGradient.GRADIENT.RAINBOW',
            'unlitGradient.GRADIENT.GREYSCALE', 'defaultUnlit', 'defaultLit'
        ];
        let shaderListEl = document.getElementById(`ui-options-${tag}-shader`);
        shaderChild.forEach(function(item, index, array) {
            shaderListEl[index].hidden = !shaderList.includes(item);
        });
    };

    /**
     * Update rendering options when the user selects Geometry or custom
     * property.
     * @callback
     * @param {String} tag
     * @param {Bool} onCreate True if the property panel is being created, False
     * if based on user selection.
     * @param {Event} evt
     * @listens document#change
     */
    onPropertyChanged = (tag, onCreate, evt) => {
        // Disable incompatible options
        let propListEl = document.getElementById(`ui-options-${tag}-property`);
        let idxListEl = document.getElementById(`ui-options-${tag}-index`);
        let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
        if (propListEl.value === '') {  // Geometry
            cmapEl.previousElementSibling.previousElementSibling.style.display =
                    'none';  // Hide index
            this.validShaders =
                    ['defaultLit', 'unlitSolidColor', 'defaultUnlit'];
        } else {
            cmapEl.previousElementSibling.previousElementSibling.style.display =
                    'block';  // Show index
            idxListEl.max =
                    this.tagsPropertiesShapes.get(tag)[propListEl.value] - 1;
            idxListEl.value = Math.min(idxListEl.value, idxListEl.max);
            idxListEl.disabled = (idxListEl.max == 0);
            // custom properties
            this.validShaders = [
                'unlitGradient.GRADIENT.RAINBOW',
                'unlitGradient.GRADIENT.GREYSCALE'
            ];
            const labelNames = this.tagLabelsNames.get(tag);
            if (idxListEl.max == 0 && labelNames != null) {
                this.validShaders.unshift(
                        'unlitGradient.LUT');  // Add as first item
            }
            if (idxListEl.max >= 2) {
                this.validShaders.push('defaultUnlit');  // RGB
            }
        }
        this.setEnabledShaders(tag, this.validShaders);
        this.onShaderChanged(tag, onCreate);
    };

    /**
     * Update rendering options when the user selects a shader.
     * @callback
     * @param {String} tag
     * @param {Bool} onCreate True if the property panel is being created, False
     * if based on user selection.
     * @param {Event} evt
     * @listens document#change
     */
    onShaderChanged = (tag, onCreate, evt) => {
        let shaderListEl = document.getElementById(`ui-options-${tag}-shader`);
        let idxListEl = document.getElementById(`ui-options-${tag}-index`);
        if ([
                'unlitGradient.GRADIENT.RAINBOW',
                'unlitGradient.GRADIENT.GREYSCALE'
            ].includes(shaderListEl.value)) {
            idxListEl.parentElement.style.display = 'block';
        } else {
            idxListEl.parentElement.style.display = 'none';
        }
        let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
        let renderStateTag = this.renderState.get(tag);

        let needRenderUpdate = false;
        if (!this.validShaders.includes(shaderListEl.value)) {
            shaderListEl.value = this.validShaders[0];
            needRenderUpdate = true;  // need update with valid shader
        }

        cmapEl.replaceChildren();  // delete all children
        let idx = 0;
        if (shaderListEl.value === 'unlitSolidColor') {
            idx = 1;
            let color = [128, 128, 128, 255];
            if (onCreate) {
                color = renderStateTag.colormap.values().next().value;
            }
            cmapEl.innerHTML = `<label class="property-ui-colormap">
          <input type="checkbox" id="ui-cmap-${
                    tag}-alpha-0" checked disabled style="display:none;">
          <input type="text" id="ui-cmap-${tag}-val-0" value="" disabled>
          <input type="color" id="ui-cmap-${tag}-col-0"
            value="${rgbToHex(color)}"
            title="R:${color[0]} G:${color[1]} B:${color[2]}
Click to change">
        </label>`;
        } else if (shaderListEl.value === 'unlitGradient.LUT') {
            // create LabelLUT
            if (this.tagLabelsNames.has(tag)) {
                const labelNames = this.tagLabelsNames.get(tag);
                if (!onCreate) {
                    renderStateTag.colormap = new Map(
                            Object.keys(labelNames)
                                    .map((lab,
                                          k) => [lab, this.labelLUTColors[k]]));
                }
                for (const [label, name] of Object.entries(labelNames)
                             .sort((l1, l2) => parseInt(l1) > parseInt(l2))) {
                    const color = renderStateTag.colormap.get(label);
                    const checked = color[3] > 0 ? 'checked' : '';
                    cmapEl.insertAdjacentHTML(
                            'beforeend',
                            `<label class="property-ui-colormap">
              <input type="checkbox" id="ui-cmap-${tag}-alpha-${idx}" ${
                                    checked} title="Show / hide data for this label">
              <input type="text" id="ui-cmap-${tag}-val-${idx}" value="${
                                    label}: ${
                                    name}" title="label: class name" readonly>
              <input type="color" id="ui-cmap-${tag}-col-${idx}"
                value="${rgbToHex(color)}"
                title="R:${color[0]}, G:${color[1]}, B:${color[2]}
Click to change">
            </label>`);
                    idx = idx + 1;
                }
            }
        } else if (shaderListEl.value.startsWith('unlitGradient.GRADIENT.')) {
            if (!onCreate) {
                renderStateTag.colormap = new Map(Object.entries(
                        this.COLORMAPS[shaderListEl.value.slice(23)]));
            }
            let currentRange = renderStateTag.range;
            cmapEl.insertAdjacentHTML(
                    'beforeend',
                    `<p>Range: [${currentRange[0].toPrecision(4)}, ${
                            currentRange[1].toPrecision(4)}]</p>`);
            if (!currentRange) {
                currentRange = [0.0, 1.0];
            }
            const step = (currentRange[1] - currentRange[0]) / 16;
            for (const [value, color] of renderStateTag.colormap) {
                const valueFlt = currentRange[0] +
                        parseFloat(value) * (currentRange[1] - currentRange[0]);
                const checked = color[3] > 0 ? 'checked' : '';
                cmapEl.insertAdjacentHTML(
                        'beforeend', `<label class="property-ui-colormap">
            <span>
              <button type="button" id="ui-cmap-${tag}-add-${idx}" title=
              "Insert a new color point in the color gradient">&#xff0b;</button>
              <button type="button" id="ui-cmap-${tag}-rem-${idx}" title=
              "Delete this color point from the color gradient">&#xff0d;</button>
            </span>
            <input type="number" id="ui-cmap-${tag}-val-${idx}" title=
            "Associated data value for this color point. Values will be sorted automatically."
            value=${valueFlt.toPrecision(4)} step=${step} min=${currentRange[0]}
            max=${currentRange[1]}>
            <input type="color" id="ui-cmap-${tag}-col-${idx}"
              value="${rgbToHex(color)}"
              title="R:${color[0]}, G:${color[1]}, B:${color[2]}
Click to change">
          </label>`);
                idx = idx + 1;
            }
        } else if (shaderListEl.value === 'defaultUnlit') {
            idxListEl.value = 0;  // Index is unused in this case
        }
        // Setup colormap callbacks:
        for (let ncol = 0; ncol < idx; ++ncol) {
            document.getElementById(`ui-cmap-${tag}-col-${ncol}`)
                    .addEventListener('change', setColorTitle);
            document.getElementById(`ui-cmap-${tag}-col-${ncol}`)
                    .addEventListener(
                            'change', this.requestRenderUpdate.bind(this, tag));
            document.getElementById(`ui-cmap-${tag}-val-${ncol}`)
                    .addEventListener(
                            'change', this.requestRenderUpdate.bind(this, tag));
            let alphaCheck =
                    document.getElementById(`ui-cmap-${tag}-alpha-${ncol}`);
            if (alphaCheck != null) {
                alphaCheck.addEventListener(
                        'change', this.requestRenderUpdate.bind(this, tag));
            }
            let addBtn = document.getElementById(`ui-cmap-${tag}-add-${ncol}`);
            if (typeof addBtn != 'undefined' && addBtn != null) {
                addBtn.addEventListener(
                        'click', this.cmapAdd.bind(this, tag, ncol));
                let remBtn =
                        document.getElementById(`ui-cmap-${tag}-rem-${ncol}`);
                remBtn.addEventListener(
                        'click', this.cmapRem.bind(this, tag, ncol));
            }
        }
        if (!onCreate || needRenderUpdate) {
            this.requestRenderUpdate(tag);
        }
    };

    /**
     * Add a color to the colormap. The average of the previous and next colors
     * is added.
     * @callback
     * @param {String} tag
     * @param {Number} ncol Position of the new color.
     * @param {Event} evt
     * @listens document#change
     */
    cmapAdd = (tag, ncol, evt) => {
        let renderStateTag = this.renderState.get(tag);
        let cmap = [...renderStateTag.colormap];
        // If ncol is last, new value, color is average with 2nd last entry.
        // If colormap has only one entry, add (1.0 or 0.0, white)
        let ncol1 = (ncol === cmap.length - 1) ? ncol - 1 : ncol + 1;
        let newVal = parseFloat(cmap[ncol][0]) > 0.0 ? 0.0 : 1.0;
        let newCol = [255, 255, 255, 255];
        if (ncol1 >= 0) {
            newVal = (parseFloat(cmap[ncol][0]) + parseFloat(cmap[ncol1][0])) /
                    2;
            for (let k = 0; k < 3; ++k) {
                newCol[k] =
                        Math.round(cmap[ncol][1][k] + cmap[ncol1][1][k]) / 2;
            }
        }
        renderStateTag.colormap.set(newVal, newCol);
        let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
        // Custom property to indicate cmapEl is outdated
        cmapEl.outdated = true;
        this.requestRenderUpdate(tag);
    };

    /**
     * Remove a color from the colormap.
     * @callback
     * @param {String} tag
     * @param {Number} ncol Position of the color.
     * @param {Event} evt
     * @listens document#change
     */
    cmapRem = (tag, ncol, evt) => {
        let renderStateTag = this.renderState.get(tag);
        let cmap = [...renderStateTag.colormap];
        renderStateTag.colormap.delete(cmap[ncol][0]);
        let cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
        // Custom property to indicate cmapEl is outdated
        cmapEl.outdated = true;
        this.requestRenderUpdate(tag);
    };

    /**
     * Send message to the Open3D server to update rendering for a tag.
     * @callback
     * @param {String} tag
     * @param {Event} evt
     * @listens document#change
     */
    requestRenderUpdate = (tag, evt) => {
        const propListEl =
                document.getElementById(`ui-options-${tag}-property`);
        const idxListEl = document.getElementById(`ui-options-${tag}-index`);
        const shaderListEl =
                document.getElementById(`ui-options-${tag}-shader`);

        let updated = [];
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
        if (propListEl.value !== renderStateTag.property ||
            idxListEl.value != renderStateTag.index) {
            updated.push('property');
        }
        renderStateTag.property = propListEl.value;
        renderStateTag.index = idxListEl.value;
        if (shaderListEl.value != renderStateTag.shader) {
            updated.push('shader');
        }
        renderStateTag.shader = shaderListEl.value;

        let cmap = renderStateTag.colormap;
        let nCols = 0;
        const cmapEl = document.getElementById(`ui-options-${tag}-colormap`);
        if (cmapEl.outdated == null) {
            if (renderStateTag.shader == 'unlitSolidColor') {
                nCols = 1;
            } else if (renderStateTag.shader.startsWith('unlitGradient')) {
                nCols = cmap.size;
            }
            const currentRange = renderStateTag['range'];
            cmap.clear();
            for (let idx = 0; idx < nCols; ++idx) {
                const alphaEl =
                        document.getElementById(`ui-cmap-${tag}-alpha-${idx}`)
                const alpha = (alphaEl == null || alphaEl.checked) ? 255 : 0;
                let labelValue =
                        document.getElementById(`ui-cmap-${tag}-val-${idx}`)
                                .value;
                if (labelValue.includes(':')) {  // LabelLUT
                    labelValue = labelValue.split(':')[0];
                } else {  // Colormap
                    labelValue = (parseFloat(labelValue) - currentRange[0]) /
                            (currentRange[1] - currentRange[0]);
                }
                let color = hexToRgb(
                        document.getElementById(`ui-cmap-${tag}-col-${idx}`)
                                .value);
                color.push(alpha);
                cmap.set(labelValue, color);
            }
        }
        if (!(nCols == 0 && cmap.size == 0)) updated.push('colormap');

        this.messageId += 1;
        let updateRenderingMessage = {
            messageId: this.messageId,
            class_name: 'tensorboard/update_rendering',
            window_uid_list: Array.from(this.windowState.keys()),
            tag: tag,
            render_state: JSON.parse(JSON.stringify(renderStateTag)),
            updated: updated
        };
        // Need colormap Object for JSON
        updateRenderingMessage.render_state.colormap =
                Object.fromEntries(renderStateTag.colormap.entries());
        console.info('Sending updateRenderingMessage:', updateRenderingMessage);
        this.webRtcClientList.values().next().value.sendJsonData(
                updateRenderingMessage);
        // add busy indicator
        document.getElementById(
                        'loader_video_' +
                        updateRenderingMessage.window_uid_list[0])
                .classList.add('loader');
    };

    /**
     * Event handler for browser tab / window close. Disconnect all server data
     * channel connections and close all server windows.
     * @callback
     * @param {Event} evt
     * @listens document#beforeunload
     */
    closeWindow = (evt) => {
        for (let [windowUId, webRtcClient] of this.webRtcClientList) {
            navigator.sendBeacon(
                    this.URL_ROUTE_PREFIX +
                            '/close_window?window_id=' + windowUId,
                    null);
        }
        this.webRtcClientList.clear();
    };

    /**
     * Send a data channel message to the server to reload runs and tags from
     * the event file. Automatically run on loading.
     * @callback
     * @listens document#click, WebRtcStreamer#RemoteDataChannelOpen
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
                class_name: 'tensorboard/' + windowUId + '/get_run_tags'
            };
            console.info('Sending getRunTagsMessage: ', getRunTagsMessage);
            client.dataChannel.send(JSON.stringify(getRunTagsMessage));
        } else {
            console.warn('No webRtcStreamer object initialized!');
        }
    };

    /**
     * Send a data channel message to the server to request an update to the
     * geometry display.
     * @callback
     * @param {String} windowUId
     * @listens WebRtcStreamer#RemoteDataChannelOpen
     */
    requestGeometryUpdate = (windowUId) => {
        this.messageId += 1;
        let updateGeometryMessage = JSON.parse(JSON.stringify({
            // deep copy
            messageId: this.messageId,
            window_uid: windowUId,
            class_name: 'tensorboard/' + windowUId + '/update_geometry',
            run: this.windowState.get(windowUId).run,
            tags: Array.from(this.selectedTags),
            render_state: Object.fromEntries(
                    Array.from(this.renderState.entries())
                            .filter(([tag, rs]) => this.selectedTags.has(tag))),
            batch_idx: this.commonBatchIdx ||
                    this.windowState.get(windowUId).batch_idx,
            step: this.commonStep || this.windowState.get(windowUId).step
        }));

        // Need colormap Object for JSON (not Map)
        for (const tag of this.selectedTags) {
            let rst = updateGeometryMessage.render_state[tag];
            if (typeof rst != 'undefined' &&
                this.renderState.get(tag).colormap != null)
                rst.colormap = Object.fromEntries(
                        this.renderState.get(tag).colormap.entries());
        }
        console.info('Sending updateGeometryMessage:', updateGeometryMessage);
        this.webRtcClientList.get(windowUId).sendJsonData(
                updateGeometryMessage);
        // add busy indicator
        document.getElementById('loader_video_' + windowUId)
                .classList.add('loader');
    };

    /**
     * Event handler for Run and Tags selector update. Triggers a geometry
     * update message.
     * @callback
     * @param {Event} evt
     * @listens document#change
     */
    onRunTagselect = (evt) => {
        if (evt.target.name === 'run-selector-checkboxes') {
            if (this.runWindow.has(evt.target.id)) {
                const windowUId = this.runWindow.get(evt.target.id);
                let windowWidget =
                        document.getElementById('widget_video_' + windowUId);
                if (evt.target.checked) {  // display window
                    windowWidget.style.display = 'flex';
                    console.info(
                            'Showing window ' + windowUId + ' with run ' +
                            evt.target.id);
                } else {  // hide window
                    windowWidget.style.display = 'none';
                    console.info(
                            'Hiding window ' + windowUId + ' with run ' +
                            evt.target.id);
                }
            } else {  // create new window
                this.requestNewWindow(evt.target.id);
            }
        } else if (evt.target.name === 'tag-selector-checkboxes') {
            if (evt.target.checked) {
                this.selectedTags.add(evt.target.id);
            } else {
                this.selectedTags.delete(evt.target.id);
                document.getElementById(`property-${evt.target.id}`)
                        .style.display = 'none';
                document.getElementById(`toggle-property-${evt.target.id}`)
                        .disabled = true;
            }
            for (const windowUId of this.windowState.keys()) {
                this.requestGeometryUpdate(windowUId);
            }
        }
    };

    /**
     * Event handler for Step and Batch Index selector update. Triggers a
     * geometry update message.
     * @callback
     * @param {String} windowUId
     * @param {Event} evt
     * @listens document#change
     */
    onStepBIdxSelect = (windowUId, evt) => {
        if (evt.target.name.startsWith('batch-idx-selector')) {
            if (this.commonBatchIdx != null) {
                this.commonBatchIdx = evt.target.value;
                for (const windowUId of this.windowState.keys()) {
                    this.requestGeometryUpdate(windowUId);
                }
            } else {
                this.windowState.get(windowUId).batch_idx = evt.target.value;
                this.requestGeometryUpdate(windowUId);
            }
        } else if (evt.target.name.startsWith('step-selector')) {
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
     * @callback
     * @param {String} windowUId
     * @param {Event} evt
     * @listens WebRtcStreamer#message
     */
    processDCMessage = (windowUId, evt) => {
        let message = null;
        try {
            message = JSON.parse(evt.data);
        } catch (err) {
            if (err.name === 'SyntaxError') {
                if (evt.data.endsWith('DataChannel open')) {
                    return;
                }
                if (evt.data.startsWith('[Open3D WARNING]')) {
                    console.warn(evt.data);
                    return;
                }
                console.error(err.name, err.message, evt.data);
            }
        }
        // remove busy indicator if reply is for last message
        if (this.messageId == message.messageId) {
            document.getElementById('loader_video_' + windowUId)
                    .classList.remove('loader');
        }
        if (message.status !== 'OK') {
            console.error(message.status);
        }
        if (message.class_name.endsWith('toggle_settings')) {
            let videoElt = document.getElementById('video_' + windowUId);
            videoElt.settingsOpen = message.open;
            if (message.open == true) {
                // O3DVisualizer controls work only with full size.
                videoElt.width = this.fullWidth;
                videoElt.height = this.fullHeight;
                // Disable view sync on opening options panel
                let optViewElt = document.getElementById('ui-options-view');
                optViewElt.checked = false;
                evt = new Event('change');
                optViewElt.dispatchEvent(evt);
            }
        } else if (message.class_name.endsWith('get_run_tags')) {
            const runToTags = message.run_to_tags;
            this.createSelector(
                    'run-selector-checkboxes', 'run-selector',
                    Object.getOwnPropertyNames(runToTags), 'checkbox',
                    [message.current.run]);
            this.selectedTags = new Set(message.current.tags);
            let allTags = new Set();
            for (const run in runToTags) {
                for (const tag of runToTags[run]) {
                    allTags.add(tag);
                }
            }
            this.createSelector(
                    'tag-selector-checkboxes', 'tag-selector', allTags,
                    'checkbox-button', this.selectedTags);

            if (this.windowState.size === 0) {  // First load
                this.runWindow.set(message.current.run, windowUId);
                this.windowState.set(windowUId, message.current);
                this.COLORMAPS = message.colormaps;  // Object (not Map)
                this.labelLUTColors =
                        message.LabelLUTColors;  // Object (not Map)
            }
            for (const [windowUId, state] of this.windowState) {
                if (runToTags.hasOwnProperty(state.run)) {  // update window
                    this.requestGeometryUpdate(windowUId);
                } else {  // stale run: close window
                    fetch(this.URL_ROUTE_PREFIX +
                                  '/close_window?window_id=' + windowUId,
                          null)
                            .then((response) => response.json())
                            .then((response) => console.log(response))
                            .then(this.runWindow.delete(state.run))
                            .then(this.windowState.delete(windowUId))
                            .then(this.webRtcClientList.delete(windowUId))
                            .then(document.getElementById(
                                                  'widget_video_' + windowUId)
                                          .remove())
                            .catch((err) => console.error(
                                           'Error closing widget:', err));
                }
            }
        } else if (message.class_name.endsWith('update_geometry')) {
            // Sync state with server
            console.assert(
                    message.window_uid === windowUId,
                    `windowUId mismatch: received ${message.window_uid} !== ${
                            windowUId}`);
            this.windowState.set(windowUId, message.current);
            // Update run level selectors
            this.createSlider(
                    windowUId, 'batch-idx-selector', 'Batch index',
                    'batch-idx-selector-div-' + windowUId, 0,
                    message.current.batch_size - 1, message.current.batch_idx);
            this.createSlider(
                    windowUId, 'step-selector', 'Step',
                    'step-selector-div-' + windowUId,
                    message.current.step_limits[0],
                    message.current.step_limits[1], message.current.step);
            // Init with milliseconds
            const wallTime = new Date(message.current.wall_time * 1000);
            document.getElementById('video_' + windowUId).title =
                    message.current.run + ' at ' + wallTime.toLocaleString();
            // Update app level selectors
            this.tagLabelsNames = new Map([
                ...this.tagLabelsNames,
                ...Object.entries(message.tag_label_to_names)
            ]);
            this.tagsPropertiesShapes = new Map([
                ...this.tagsPropertiesShapes,
                ...Object.entries(message.tags_properties_shapes)
            ]);
            for (const tag of message.current.tags) {
                this.renderState.set(tag, message.current.render_state[tag]);
                const cmap = this.renderState.get(tag).colormap;  // Object
                if (typeof cmap != 'undefined' && cmap != null) {
                    this.renderState.get(tag).colormap =
                            new Map(Object.entries(cmap));  // Map
                }
                this.createPropertyPanel(tag);
            }
        } else if (message.class_name.endsWith('update_rendering')) {
            this.renderState.set(message.tag, message.render_state);
            const cmap = this.renderState.get(message.tag).colormap;  // Object
            if (typeof cmap != 'undefined' && cmap != null) {
                this.renderState.get(message.tag).colormap =
                        new Map(Object.entries(cmap));  // Map
            }
            this.createPropertyPanel(message.tag);
        }
        // Send some MouseEvents to force a redraw
        const mouseEvt = new MouseEvent('mousemove');
        let videoElt = document.getElementById('video_' + windowUId);
        for (let i = 0; i < 3; ++i) {
            videoElt.dispatchEvent(mouseEvt);
        }
    };
}

/**
 * The main entry point of any TensorBoard iframe plugin.
 */
export async function render() {
    const o3dclient = new TensorboardOpen3DPluginClient();
}
