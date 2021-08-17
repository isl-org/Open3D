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
// ----------------------------------------------------------------------------
// Contains source code from
// https://github.com/mpromonet/webrtc-streamer
//
// This software is in the public domain, furnished "as is", without technical
// support, and with no warranty, express or implied, as to its usefulness for
// any purpose.
// ----------------------------------------------------------------------------

// import "webrtcstreamer";

class TensorboardOpen3DPluginClient {

    url_route_prefix = "/data/plugin/Open3D";
    http_handshake_url = window.location.protocol + "//" + window.location.hostname +
        ":8888";
    webRtcOptions = "rtptransport=tcp&timeout=60";
    width = Math.round(window.innerWidth * 2/5 - 60);
    height = Math.round(this.width * 3/4);
    fontsize = window.getComputedStyle(document.body).fontSize;
    messageId = 0;
    webRtcClientList = new Map(); // {windowUId -> webRtcStreamer}
    window_state = new Map();     // {windowUId -> Geometry state (run, tags, batch_idx, step)}
    runWindow = new Map();  // {run -> windowUId}
    selected_tags = new Set();
    common_step = undefined;
    common_batch_idx = undefined;

    /**
     * Entry point for the TensorBoard Open3D plugin client
     * @constructor
     */
    constructor() {
        const dashboard_html =
            `<link  href="style.css" rel="stylesheet">

            <div id="open3d-dashboard">
                <div id="options-selector">
                    <h3>Options</h3>
                    <label class="container">Sync view
                        <input type="checkbox" id="ui-options-view">
                        <span class="checkmark"></span>
                    </label>
                    <label class="container">Sync step
                        <input type="checkbox" id="ui-options-step">
                        <span class="checkmark"></span>
                    </label>
                    <label class="container">Sync batch index
                        <input type="checkbox" id="ui-options-bidx">
                        <span class="checkmark"></span>
                    </label>

                    <h3>Runs</h3>
                    <p id="logdir"></p>
                    <div id="run-selector"></div>

                    <h3>Tags</h3>
                    <div id="tag-selector"></div>
                </div>

                <div id="widget-view"> </div>
            </div>
            `;
        document.body.insertAdjacentHTML("beforeend", dashboard_html);
        // Ask Open3D for a new window
        console.info("Requesting window with size: ", this.width, "x", this.height);
        fetch(this.url_route_prefix + "/new_window?width=" + this.width + "&height="
            + this.height + "&fontsize=" + this.fontsize, null)
            .then((response) => response.json())
            .then((response) => this.addConnection(response.window_id,
                response.logdir))
            .then(this.addAppEventListeners)
            .catch(err => console.error("Error: /new_window failed:" + err));
    };

    /** Add App level event listeners.
    */
    addAppEventListeners = () => {
        window.addEventListener("beforeunload", this.closeWindow);
        // Listen for the user clicking on the main TB reload button
        let tb_reload_button =
            parent.document.querySelector(".reload-button");
        if (tb_reload_button != null) {
            tb_reload_button.addEventListener("click", this.reloadRunTags);
        }
        // App option handling
        document.getElementById("ui-options-step").addEventListener("change", (evt) => {
            if (evt.target.checked) {
                this.common_step = this.window_state.values().next().value.step;
                for (let [windowUId, current_state] of this.window_state) {
                    current_state.step = this.common_step;
                    this.requestGeometryUpdate(windowUId);
                }
            } else {
                this.common_step = undefined;
            }
            console.debug("this.common_step = ", this.common_step);
        });

        document.getElementById("ui-options-bidx").addEventListener("change", (evt) => {
            if (evt.target.checked) {
                this.common_batch_idx = this.window_state.values().next().value.batch_idx;
                for (let [windowUId, current_state] of this.window_state) {
                    current_state.batch_idx = this.common_batch_idx;
                    this.requestGeometryUpdate(windowUId);
                }
            } else {
                this.common_batch_idx = undefined;
            }
            console.debug("this.common_batch_idx = ", this.common_batch_idx);
        });
    };

    /**
     * Create a video element to display geometry and initiate WebRTC connection
     * with server. Attach listeners to process data channel messages.
     *
     * @param {string} windowUId  Window ID from the server, e.g.: "window_1"
     * @param {string} logdir TensorBoard log directory to be displayed to the
     *      user.
     * @param {string} run TB run to display here. Can be 'undefined' in which
     *      case the server will assign a run.
     */
    addConnection = (windowUId, logdir, run) => {
        const videoId = "video_" + windowUId;
        let logdir_el = document.getElementById("logdir");
        logdir_el.innerText = logdir;

        // Add a video element to display WebRTC stream.
        const widget_template = `
        <div class="webrtc" id="widget_${videoId}">
            <div class="batchidx-step-selector">
                <div id="batch-idx-selector-div-${windowUId}"></div>
                <div id="step-selector-div-${windowUId}"></div>
            </div>
            <video id="${videoId}" muted="true" playsinline="true">
                Your browser does not support HTML5 video.
            </video>
        </div>
        `;
        let widgetView = document.getElementById("widget-view");
        widgetView.insertAdjacentHTML("beforeend", widget_template);

        let videoElt = document.getElementById(videoId);
        let client = new WebRtcStreamer(videoElt, this.http_handshake_url, null, null);
        console.info("[addConnection] videoId: " + videoId);

        client.connect(windowUId, /*audio*/ null, this.webRtcOptions);
        this.webRtcClientList.set(windowUId, client);
        console.info("[addConnection] windowUId: " + windowUId);
        videoElt.addEventListener('LocalDataChannelOpen', (evt) => {
            evt.detail.channel.addEventListener('message',
                this.processDCMessage.bind(this, windowUId));
        });
        videoElt.addEventListener("resize", this.ResizeEvent);
        if (this.webRtcClientList.size == 1) {
            // Initial Run Tag reload only needed for first window
            videoElt.addEventListener('RemoteDataChannelOpen', this.reloadRunTags);
        } else { // Initialize state from first webrtc_window
            // deep copy: All objects and sub-objects need to be copied with JS
            // spread syntax.
            let new_window_state = {...this.window_state.values().next().value};
            new_window_state.tags = [...new_window_state.tags];
            this.window_state.set(windowUId, new_window_state);
            if (run) {
                this.runWindow.set(run, windowUId);
                this.window_state.get(windowUId).run = run;
                console.debug("After addConnection: this.runWindow = ",
                    this.runWindow, "this.window_state = ", this.window_state);
            }
            videoElt.addEventListener('RemoteDataChannelOpen',
                this.requestGeometryUpdate.bind(this, windowUId));
        }
    };

    /**
     * Create generic checkbox and radio button selectors used for Run and Tag
     * selectors respectively.
     * @param {string} name Selector name attribute.
     * @param {string} parent_id Id of parent element.
     * @param {array} options Array of string options to create
     * @param {string} type "radio" (single selection) or "checkbox"
     * (multi-select).
     * @param {array} initial_checked_options Array of options to be shown
     * selected initially.
     */
    createSelector(name, parent_id, options, type, initial_checked_options) {
        console.assert(type == "radio" || type == "checkbox",
            "type must be radio or checkbox");

        let parent_element = document.getElementById(parent_id);
        parent_element.replaceChildren();  /* Remove existing children */
        initial_checked_options = new Set(initial_checked_options);
        for (const option of options) {
            let checked="";
            if(initial_checked_options.has(option))
                checked="checked";
            const option_template=`
            <label class="container">${option}
                <input type="${type}" id="${option}" name="${name}" ${checked}>
                <span class="checkmark"></span>
            </label>
            `;
            parent_element.insertAdjacentHTML('beforeend', option_template);
            document.getElementById(option).addEventListener("change",
                this.onRunTagselect);
        }
    };

    /**
     * Create slider (range input element) for selecting a geometry in a batch,
     * or for selecting a step / iteration.
     * @param {int} windowUId Window ID associated with slider.
     * @param {string} name Selector name/id attribute.
     * @param {string} display_name Show this label in the UI.
     * @param {string} parent_id Id of parent element.
     * @param {number} min Min value of range.
     * @param {number} max Max value of range.
     * @param {number} value Initial value of range.
     */
    createSlider(windowUId, name, display_name, parent_id, min, max, value) {

        let parent_element = document.getElementById(parent_id);
        parent_element.replaceChildren();  /* Remove existing children */
        const name_wid = name + "-" + windowUId;
        if (max > min) { // Don't create slider if no choice
            const slider_template=`
            <form oninput="document.getElementById('${name_wid}_output').value
                = document.getElementById('${name_wid}').valueAsNumber;">
                <label> ${display_name + ': [' + min.toString() + '-' +
                        max.toString() + '] '}
                    <input type="range" id="${name_wid}" name="${name_wid}" min="${min}"
                        max="${max}" value="${value}">
                    </input>
                </label>
                <output for="${name_wid}" id="${name_wid + '_output'}"> ${value}
                </output>
            </form>
            `;
            parent_element.insertAdjacentHTML('beforeend', slider_template);
            document.getElementById(name_wid).addEventListener("change",
                this.onStepBIdxSelect.bind(this, windowUId));
        }
    };

    /**
     * Event handler for window resize. Forwards request to resize the WebRTC
     * window to the server.
     */
    // arrow function binds this at time of instantiation
    ResizeEvent = (evt) => {
        return; // TODO handle resize
        const windowUID = evt.target.id.substring(6); // Remove "video_"
            const resizeEvent = {
                window_uid: windowUId,
                class_name: "ResizeEvent",
                height: webrtc_widget.scrollHeight,
                width: webrtc_widget.scrollWidth
            };
            this.webRtcClient.dataChannel.send(JSON.stringify(resizeEvent));

        if (this.webRtcClientList.size > 1) {
            let webrtc_widget = document.getElementById("video_" +
                this.windowUId);
            webrtc_widget.style.height = this.height;
            webrtc_widget.style.width = this.width;
            // const rect = webrtc_widget.getBoundingClientRect();
            const resizeEvent = {
                window_uid: this.windowUId,
                class_name: "ResizeEvent",
                height: webrtc_widget.scrollHeight,
                width: webrtc_widget.scrollWidth
            };
            this.webRtcClient.dataChannel.send(JSON.stringify(resizeEvent));
        }
    };

    /**
     * Event handler for window close. Disconnect all server data channel
     * connections and close all server windows.
     */
    closeWindow = () => {
        for (let [windowUId, webRtcClient] of this.webRtcClientList) {
            fetch(this.url_route_prefix + "/close_window?window_id=" +
                windowUId, null)
                .then(() => {
                    webRtcClient.disconnect();
                    webRtcClient = undefined;
                });
        }
        this.webRtcClientList.clear();
    };

    /**
     * Send a data channel message to the server to reload runs and tags from
     * the event file. Automatically run on loading.
     */
    reloadRunTags = () => {
        this.messageId += 1;
        // Any _open_ window_id may be used. TODO: Check
        const getRunTagsMessage = {
            messageId: this.messageId,
            class_name: "tensorboard/window_0/get_run_tags"
        }
        console.info("Sending getRunTagsMessage: ", getRunTagsMessage);
        if (this.webRtcClientList.size > 0) {
            let client = this.webRtcClientList.values().next().value;
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
        const updateGeometryMessage = {
            messageId: this.messageId,
            window_uid: windowUId,
            class_name: "tensorboard/" + windowUId + "/update_geometry",
            run: this.window_state.get(windowUId).run,
            tags: Array.from(this.selected_tags),
            batch_idx: this.common_batch_idx || this.window_state.get(windowUId).batch_idx,
            step: this.common_step || this.window_state.get(windowUId).step
        };
        console.info("Sending updateGeometryMessage:", updateGeometryMessage);
        this.webRtcClientList.get(windowUId).dataChannel.send(JSON.stringify(
            updateGeometryMessage));
    };

    /**
     * Event handler for Run and Tags selector update. Triggers a geometry
     * update message.
     */
    onRunTagselect = (evt) => {
        if (evt.target.name == "run-selector-checkboxes") {
            if (this.runWindow.has(evt.target.id)) {
                const windowUId = this.runWindow.get(evt.target.id);
                let window_widget = document.getElementById("widget_video_" + windowUId);
                if (evt.target.checked) { // display window
                    window_widget.style.display = "block";
                    console.info("Showing window " + windowUId + " with run " + evt.target.id);
                } else {    // hide window
                    window_widget.style.display = "none";
                    console.info("Hiding window " + windowUId + " with run " + evt.target.id);
                }
            } else {    // create new window
                console.info("Requesting window with size: ", this.width, "x", this.height);
                fetch(this.url_route_prefix + "/new_window?width=" + this.width + "&height="
                    + this.height + "&fontsize=" + this.fontsize, null)
                    .then((response) => response.json())
                    .then((response) => this.addConnection(response.window_id,
                        response.logdir, evt.target.id))
                    .catch(err => console.error("Error: /new_window failed:" + err));
            }
        } else if (evt.target.name == "tag-selector-checkboxes") {
            if (evt.target.checked)
                this.selected_tags.add(evt.target.id);
            else
                this.selected_tags.delete(evt.target.id);
            for (const windowUId of this.window_state.keys())
                this.requestGeometryUpdate(windowUId);
        }
    };

    /**
     * Event handler for Step and Batch Index selector update. Triggers a
     * geometry update message.
     */
    onStepBIdxSelect = (windowUId, evt) => {
        console.debug("[onStepBIdxSelect] this.window_state: ",
            this.window_state, "this.common_step", this.common_step,
            "this.common_batch_idx", this.common_batch_idx);
        if (evt.target.name.startsWith("batch-idx-selector")) {
            if (this.common_batch_idx != undefined) {
                this.common_batch_idx = evt.target.value;
                for (const windowUId of this.window_state.keys())
                    this.requestGeometryUpdate(windowUId);
            } else {
                this.window_state.get(windowUId).batch_idx = evt.target.value;
                this.requestGeometryUpdate(windowUId)
            };
        } else if (evt.target.name.startsWith("step-selector")) {
            if (this.common_step != undefined) {
                this.common_step = evt.target.value;
                for (const windowUId of this.window_state.keys())
                    this.requestGeometryUpdate(windowUId);
            } else {
                this.window_state.get(windowUId).step = evt.target.value;
                this.requestGeometryUpdate(windowUId);
            }
        }
    };


    /**
     * Data channel message handler. Updates UI controls based on server state.
     */
    processDCMessage = (windowUId, evt) => {
        let message = undefined;
        try {
            message = JSON.parse(evt.data);
        } catch (err) {
            if (err.name=="SyntaxError") {
                if (evt.data.endsWith("DataChannel open")) return;
                if (evt.data.startsWith("[Open3D WARNING]")) {
                    console.warn(evt.data);
                    return;
                }
            }
        }
        if (message.class_name.endsWith("get_run_tags")) {
            this.run_to_tags = message.run_to_tags;
            this.runWindow.set(message.current.run, windowUId);
            this.window_state.set(windowUId, message.current);
            console.debug("[After get_run_tags] this.runWindow: ",
                this.runWindow, "this.window_state:", this.window_state);
            this.createSelector("run-selector-checkboxes", "run-selector",
                Object.getOwnPropertyNames(this.run_to_tags), "checkbox",
                [message.current.run]);

            this.selected_tags = new Set(message.current.tags);
            let all_tags = new Set()
            for (const run in this.run_to_tags)
                for (const tag of this.run_to_tags[run])
                    all_tags.add(tag);
            this.createSelector("tag-selector-checkboxes", "tag-selector",
                all_tags, "checkbox", this.selected_tags);
            this.createSlider(windowUId, "batch-idx-selector", "Batch index",
                "batch-idx-selector-div-" + windowUId, 0, message.current.batch_size - 1,
                message.current.batch_idx);
            this.createSlider(windowUId, "step-selector", "Step",
                "step-selector-div-" + windowUId, message.current.step_limits[0],
                message.current.step_limits[1], message.current.step);
            this.requestGeometryUpdate(windowUId);
        } else if (message.class_name.endsWith("update_geometry")) {
            // Sync state with server
            console.assert(message.window_uid == windowUId,
                `windowUId mismatch: received ${message.window_uid} !== ${windowUId}`);
            this.window_state.set(windowUId, message.current);
            console.debug("[After update_geometry] this.runWindow: ",
                this.runWindow, "this.window_state:", this.window_state);
            this.createSlider(windowUId, "batch-idx-selector", "Batch index",
                "batch-idx-selector-div-" + windowUId, 0,
                message.current.batch_size - 1, message.current.batch_idx);
            this.createSlider(windowUId, "step-selector", "Step",
                "step-selector-div-" + windowUId, message.current.step_limits[0],
                message.current.step_limits[1], message.current.step);
            // Init with miliseconds
            const wall_time = new Date(message.current.wall_time * 1000);
            document.getElementById("video_" + windowUId).title =
                message.current.run + " at " + wall_time.toLocaleString();

        }
    };

};

/**
 * The main entry point of any TensorBoard iframe plugin.
 */
export async function render() {
    const o3dclient = new TensorboardOpen3DPluginClient();
}
