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

class TensorboardOpen3DPluginClient {

    url_route_prefix = "/data/plugin/Open3D";
    http_handshake_url = "http://localhost:8888";
    webRtcOptions = "rtptransport=tcp&timeout=60";
    messageId = 0;
    webRtcClientList = new Map(); // {windowUId -> webRtcStreamer}
    current = new Map();     // {windowUId -> Geometry state (run, tags, batch_idx, step)}
    runWindow = new Map();  // {run -> windowUId}

    /**
     * Entry point for the Tensorboard Open3D plugin client
     * @constructor
     */
    constructor() {
        const dashboard_html =
            `<link  href="style.css" rel="stylesheet">

            <div id="open3d-dashboard">
                <div id="options-selector">
                    <h3>Options</h3>
                    <label class="container">Sync view
                        <input type="checkbox" id="ui-options-view" checked>
                        <span class="checkmark"></span>
                    </label>
                    <label class="container">Sync step
                        <input type="checkbox" id="ui-options-step" checked>
                        <span class="checkmark"></span>
                    </label>
                    <label class="container">Sync batch index
                        <input type="checkbox" id="ui-options-bidx" checked>
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
        const width = Math.round(window.innerWidth * 5/6 - 60);
        const height = Math.round(window.innerHeight * 5/6 - 40);
        const fontsize = window.getComputedStyle(document.body).fontSize;
        fetch(this.url_route_prefix + "/new_window?width=" + width + "&height="
            + height + "&fontsize=" + fontsize, null)
            .then((response) => response.json())
            .then((response) => this.addConnection(response.window_id,
                response.logdir))
            .then(this.addAppEventListeners)
            .catch(err => console.error("Error: /new_window failed:" + err));
    };

    /** Add App level event listeners.
    */
    addAppEventListeners = () => {
        window.addEventListener("resize", this.ResizeEvent);
        window.addEventListener("beforeunload", this.closeWindow);
        // Listen for the user clicking on the main TB reload button
        let tb_reload_button =
            parent.document.querySelector(".reload-button");
        if (tb_reload_button != null) {
            tb_reload_button.addEventListener("click", this.reloadRunTags);
        }
    };

    /**
     * Create a video element to display geometry and initiate WebRTC connection
     * with server. Attach listeners to process data channel messages.
     */
    addConnection = (windowUId, logdir) => {
        // this.windowUId = windowUId;
        const videoId = "video_" + windowUId;
        let logdir_el = document.getElementById("logdir");
        logdir_el.innerText = logdir;

        // Add a video element to display WebRTC stream.
        const widget_template = `
        <div class="webrtc" id="widget_${videoId}">
            <div class="batchidx-step-selector">
                <div id="batch-idx-selector"></div>
                <div id="step-selector"></div>
            </div>
            <video id="${videoId}" title="${videoId}" muted="true"
                playsinline="true">
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
            evt.detail.channel.addEventListener('message', (evt2) => {
                this.processDCMessage(windowUId, evt2);
            });
        });
        if (this.webRtcClientList.size == 1) {
            // Initial Run Tag reload only needed for first window
            videoElt.addEventListener('RemoteDataChannelOpen', this.reloadRunTags);
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
        for (const option of options) {
            let checked="";
            if(initial_checked_options.includes(option)) {
                checked="checked";
            }
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
        let name_wid = name + "-" + windowUId;
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
                (evt) => { this.onStepBIdxSelect(windowUId, evt); } );
        }
    };

    /**
     * Event handler for window resize. Forwards request to resize the WebRTC
     * window to the server.
     */
    // arrow function binds this at time of instantiation
    ResizeEvent = () => {
        return; // TODO handle resize
        if (this.webRtcClientList.size > 1) {
            let webrtc_widget = document.getElementById("video_" +
                this.windowUId);
            webrtc_widget.style.height = Math.round(window.innerHeight *5/6 - 40);
            webrtc_widget.style.width = Math.round(window.innerWidth *5/6 - 60);
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
        // Any window_id may be used
        const getRunTagsMessage = {
            messageId: this.messageId,
            class_name: "tensorboard/window_0/get_run_tags"
        }
        console.log(getRunTagsMessage);
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
            run: this.current.get(windowUId).run,
            tags: this.current.get(windowUId).tags,
            batch_idx: this.current.get(windowUId).batch_idx,
            step: this.current.get(windowUId).step
        };
        console.info(updateGeometryMessage);
        this.webRtcClientList.get(windowUId).dataChannel.send(JSON.stringify(
            updateGeometryMessage));
    };

    /**
     * Event handler for Run and Tags selector update. Triggers a geoemtry
     * update message.
     */
    onRunTagselect = (evt) => {
        if (evt.target.name == "run-selector-checkboxes") { // update tags
            // this.current.get(windowUId).run = evt.target.id;
            if (this.runWindow.has(evt.target.id)) {
                let window_widget = document.getElementById("widget_video_" + windowUId);
                if (evt.target.checked) { // display window
                    window_widget.style.display = "block";
                    console.info("Showing window " + windowUId + " with run " + evt.target.id);
                } else {    // hide window
                    window_widget.style.display = "none";
                    console.info("Hiding window " + windowUId + " with run " + evt.target.id);
                }
            } else {    // create new window
                const width = Math.round(window.innerWidth * 5/6 - 60);
                const height = Math.round(window.innerHeight * 5/6 - 40);
                const fontsize = window.getComputedStyle(document.body).fontSize;
                fetch(this.url_route_prefix + "/new_window?width=" + width + "&height="
                    + height + "&fontsize=" + fontsize, null)
                    .then((response) => response.json())
                    .then((response) => this.addConnection(response.window_id,
                        response.logdir))
                    .catch(err => console.error("Error: /new_window failed:" + err));
            }
            // this.createSelector("tag-selector-checkboxes", "tag-selector",
            //     this.run_to_tags[this.current.run], "checkbox",
            //     this.current.tags);
        } else if (evt.target.name == "tag-selector-checkboxes") {
            if (evt.target.checked) {
                this.current.values().forEach( (currentState) => {
                    currentState.tags.push(evt.target.id);
                });
            } else {
                this.current.values().forEach( (currentState) => {
                    currentState.tags.splice(currentState.tags.indexOf(
                        evt.target.id), 1);
                });
            }
            this.current.keys().forEach( (windowUId) => {
                this.requestGeometryUpdate(windowUId);
            });
        }
    };

    /**
     * Event handler for Step and Batch Index selector update. Triggers a
     * geoemtry update message.
     */
    onStepBIdxSelect = (windowUId, evt) => {
        if (evt.target.name.startsWith("batch-idx-selector-slider")) {
            this.current.get(windowUId).batch_idx = evt.target.value;
            // TODO Sync batch_idx if selected
            this.requestGeometryUpdate(windowUId);
        } else if (evt.target.name.startsWith("step-selector-slider")) {
            this.current.get(windowUId).step = evt.target.value;
            // TODO Sync step if selected
            this.requestGeometryUpdate(windowUId);
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
        console.info(message);
        if (message.class_name.endsWith("get_run_tags")) {
            this.run_to_tags = message.run_to_tags;
            this.current.set(windowUId, message.current); // TODO: Only for first window?
            this.runWindow.set(message.current.run, windowUId);

            this.createSelector("run-selector-checkboxes", "run-selector",
                Object.getOwnPropertyNames(this.run_to_tags), "checkbox",
                message.current.run);
            this.createSelector("tag-selector-checkboxes", "tag-selector",
                this.run_to_tags[message.current.run], "checkbox",
                message.current.tags);
            this.createSlider(windowUId, "batch-idx-selector-slider", "Batch index",
                "batch-idx-selector", 0, message.current.batch_size - 1,
                message.current.batch_idx);
            this.createSlider(windowUId, "step-selector-slider", "Step", "step-selector",
                message.current.step_limits[0], message.current.step_limits[1],
                message.current.step);
            this.requestGeometryUpdate(windowUId);
        } else if (message.class_name.endsWith("update_geometry")) {
            // Brute force way to sync state with server.
            this.current.set(windowUId, message.current);
            this.runWindow.set(message.current.run, windowUId);
            // this.createSelector("run-selector-checkboxes", "run-selector",
            //     Object.getOwnPropertyNames(this.run_to_tags), "checkbox",
            //     message.current.run);
            // this.createSelector("tag-selector-checkboxes", "tag-selector",
            //     this.run_to_tags[message.current.run], "checkbox",
            //     message.current.tags);
            this.createSlider(windowUId, "batch-idx-selector-slider", "Batch index",
                "batch-idx-selector", 0, message.current.batch_size - 1,
                message.current.batch_idx);
            this.createSlider(windowUId, "step-selector-slider", "Step", "step-selector",
                message.current.step_limits[0], message.current.step_limits[1],
                message.current.step);
        }
    };

};

/**
 * The main entry point of any TensorBoard iframe plugin.
 */
export async function render() {
    let o3dclient = new TensorboardOpen3DPluginClient();
}
