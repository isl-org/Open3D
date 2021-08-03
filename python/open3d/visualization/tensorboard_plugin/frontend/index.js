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
    dashboard_html =
        `<link  href="style.css" rel="stylesheet">
        <div class="run-tag-selector">
            <h3>Runs</h3>
            <p id="logdir"></p>
            <div id="run-selector"></div>

            <h3>Tags</h3>
            <div id="tag-selector"></div>
        </div>

        <div class="main-view">
            <div class="batchidx-step-selector">
                <div id="batch-idx-selector"></div>
                <div id="step-selector"></div>
            </div>

            <div id="webrtc"></div>
        </div>
        `;
    messageId = 0;
    webRtcClient;

    /**
     * Entry point for the Tensorboard Open3D plugin client
     * @constructor
     */
    constructor() {
        let dashboard = document.createElement("div");
        dashboard.setAttribute("id", "open3d-dashboard");
        document.body.appendChild(dashboard);
        dashboard.innerHTML = this.dashboard_html;
        // Ask Open3D for a new window
        const width = window.innerWidth * 5/6 - 60 ;
        const height = window.innerHeight - 40;
        const fontsize = window.getComputedStyle(document.body).fontSize;
        fetch(this.url_route_prefix + "/new_window?width=" + width + "&height="
            + height + "&fontsize=" + fontsize, null)
            .then((response) => response.json())
            .then((response) => this.addConnection(response.window_id, response.logdir))
            .catch(err => console.error("Error: /new_window failed:" + err));
    }

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
        console.assert(type == "radio" || type == "checkbox", "type must be radio or checkbox");

        let parent_element = document.getElementById(parent_id);
        parent_element.replaceChildren();  /* Remove existing children */
        for (const option of options) {
            let selector = document.createElement("label");
            parent_element.appendChild(selector);
            selector.setAttribute("class", "container");
            let selector_text = document.createTextNode(option);
            selector.appendChild(selector_text);
            let selector_input = document.createElement("input");
            selector.appendChild(selector_input);
            selector_input.setAttribute("type", type);
            selector_input.setAttribute("id", option);
            if (initial_checked_options.includes(option)) {
                selector_input.setAttribute("checked", true);
            }
            selector_input.setAttribute("name", name);
            selector_input.addEventListener('change', this.onselect);
            let selector_span = document.createElement("span");
            selector.appendChild(selector_span);
            selector_span.setAttribute("class", "checkmark");
        }
    }

    /**
     * Create slider (range input element) for selecting a geometry in a batch,
     * or for selecting a step / iteration.
     * @param {string} name Selector name/id attribute.
     * @param {string} display_name Show this label in the UI.
     * @param {string} parent_id Id of parent element.
     * @param {number} min Min value of range.
     * @param {number} max Max value of range.
     * @param {number} value Initial value of range.
     */
    createSlider(name, display_name, parent_id, min, max, value) {

        let parent_element = document.getElementById(parent_id);
        parent_element.replaceChildren();  /* Remove existing children */
        let slider_form = document.createElement("form");
        parent_element.appendChild(slider_form);
        slider_form.setAttribute("oninput", name + "_output.value = " + name + ".valueAsNumber;");
        let slider_label = document.createElement("label");
        slider_form.appendChild(slider_label);
        slider_label.setAttribute("for", name);
        slider_label.innerText = display_name + ": [" + min.toString() +  "-" + max.toString() + "] ";
        let slider_input = document.createElement("input");
        slider_form.appendChild(slider_input);
        slider_input.setAttribute("type", "range");
        slider_input.setAttribute("id", name);
        slider_input.setAttribute("name", name);
        slider_input.setAttribute("min", min);
        slider_input.setAttribute("max", max);
        slider_input.setAttribute("value", value);
        slider_input.addEventListener("change", this.onselect);
        let slider_text = document.createElement("output");
        slider_text.setAttribute("for", name);
        slider_text.setAttribute("name", name + "_output");
        slider_text.innerText = value;
        slider_form.appendChild(slider_text);
    }

    /**
     * Event handler for window resize. Forwards request to resize the WebRTC
     * window to the server.
     */
    // arrow function binds this at time of instantiation
    ResizeEvent = () => {
        if (this.webRtcClient) {
            const resizeEvent = {
                window_uid: this.windowUId,
                class_name: "ResizeEvent",
                height: window.innerHeight - 40,
                width: window.innerWidth *5/6 - 60
            };
            this.webRtcClient.dataChannel.send(JSON.stringify(resizeEvent));
        }
    };

    /**
     * Event handler for window close. Disconnect server data channel
     * connection and close server window.
     */
    closeWindow = () => {
        fetch(this.url_route_prefix + "/close_window?window_id=" + this.windowUId, null)
            .then(() => {;
                if (this.webRtcClient) {
                    this.webRtcClient.disconnect();
                    this.webRtcClient = undefined;
                }
            });
    };

    /**
     * Send a data channel message to the server to reload runs and tags from
     * the event file. Automatically run on loading.
     */
    reloadRunTags = () => {
        this.messageId += 1;
        const getRunTagsMessage = {
            messageId: this.messageId,
            window_uid: this.windowUId,
            class_name: "tensorboard/" + this.windowUId + "/get_run_tags"
        }
        console.log(getRunTagsMessage);
        if (this.webRtcClient) {
            this.webRtcClient.dataChannel.send(JSON.stringify(getRunTagsMessage));
        } else {
            console.warn("webRtcClient not initialized!");
        }
    };

    /**
     * Send a data channel message to the server to request an update to the
     * geometry display.
     */
    requestGeometryUpdate = () => {
        this.messageId += 1;
        const updateGeometryMessage = {
            messageId: this.messageId,
            window_uid: this.windowUId,
            class_name: "tensorboard/" + this.windowUId + "/update_geometry",
            run: this.current.run,
            tags: this.current.tags,
            batch_idx: this.current.batch_idx,
            step: this.current.step
        };
        console.info(updateGeometryMessage);
        this.webRtcClient.dataChannel.send(JSON.stringify(updateGeometryMessage));
    };

    /**
     * Event handler for selector update. Triggers a geoemtry update message.
     */
    onselect = (evt) => {
        if (evt.target.name == "run-selector-radio-buttons") { // update tags
            this.current.run = evt.target.id;
            this.createSelector("tag-selector-checkboxes", "tag-selector",
                this.run_to_tags[this.current.run], "checkbox", this.current.tags);
        } else if (evt.target.name == "tag-selector-checkboxes") {
            if (evt.target.checked) {
                this.current.tags.push(evt.target.id);
            } else {
                this.current.tags.splice(this.current.tags.indexOf(evt.target.id),
                    1);
            }
        } else if (evt.target.name == "batch-idx-selector-slider") {
            this.current.batch_idx = evt.target.value;
        } else if (evt.target.name == "step-selector-slider") {
            this.current.step = evt.target.value;
        }
        this.requestGeometryUpdate();
    };


    /**
     * Data channel message handler. Updates UI controls based on server state.
     */
    processDCMessage = (evt) => {
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
        console.info(message)
        if (message.class_name.endsWith("get_run_tags")) {
            this.run_to_tags = message.run_to_tags;
            this.current = message.current;

            this.createSelector("run-selector-radio-buttons", "run-selector",
                Object.getOwnPropertyNames(this.run_to_tags), "radio", this.current.run);
            this.createSelector("tag-selector-checkboxes", "tag-selector",
                this.run_to_tags[this.current.run], "checkbox", this.current.tags);
            this.createSlider("batch-idx-selector-slider", "Batch index", "batch-idx-selector",
                0, this.current.batch_size - 1, this.current.batch_idx);
            this.createSlider("step-selector-slider", "Step", "step-selector",
                this.current.step_limits[0], this.current.step_limits[1], this.current.step);
            this.requestGeometryUpdate();
        }
        if (message.class_name.endsWith("update_geometry")) {
            // Brute force way to sync state with server.
            this.current = message.current;
            this.createSelector("run-selector-radio-buttons", "run-selector",
                Object.getOwnPropertyNames(this.run_to_tags), "radio", this.current.run);
            this.createSelector("tag-selector-checkboxes", "tag-selector",
                this.run_to_tags[this.current.run], "checkbox", this.current.tags);
            this.createSlider("batch-idx-selector-slider", "Batch index", "batch-idx-selector",
                0, this.current.batch_size - 1, this.current.batch_idx);
            this.createSlider("step-selector-slider", "Step", "step-selector",
                this.current.step_limits[0], this.current.step_limits[1], this.current.step);
        }
    };

    /**
     * Create a video element to display geometry and initiate WebRTC connection
     * with server. Attach listeners to process data channel messages.
     */
    addConnection = (windowUId, logdir) => {
        this.windowUId = "window_" + windowUId;
        const videoId = "video_" + this.windowUId;
        let logdir_el = document.getElementById("logdir");
        logdir_el.innerText = logdir;

        // Add a video element to display WebRTC stream.
        if (document.getElementById(videoId) == null) {
            let webrtcDiv = document.getElementById("webrtc");
            if (webrtcDiv) {
                let divElt = document.createElement("div");
                divElt.id = "div_" + videoId;

                let videoElt = document.createElement("video");
                videoElt.id = videoId;
                videoElt.title = this.windowUId;
                videoElt.muted = true;
                videoElt.controls = false;
                videoElt.playsinline = true;
                videoElt.innerText = "Your browser does not support HTML5 video.";

                divElt.appendChild(videoElt);
                webrtcDiv.appendChild(divElt);
            }
        }

        let videoElt = document.getElementById(videoId);
        if (videoElt) {
            this.webRtcClient = new WebRtcStreamer(videoElt, this.http_handshake_url, null, null);
            console.info("[addConnection] videoId: " + videoId);

            this.webRtcClient.connect(this.windowUId, /*audio*/ null, this.webRtcOptions);
            console.info("[addConnection] windowUId: " + this.windowUId);
            console.info("[addConnection] options: " + this.webRtcOptions);
            window.onresize = this.ResizeEvent;
            window.onbeforeunload = this.closeWindow;
            videoElt.addEventListener('LocalDataChannelOpen', (evt) => {
                evt.detail.channel.addEventListener('message', this.processDCMessage);
            });
            videoElt.addEventListener('RemoteDataChannelOpen', this.reloadRunTags);
            // Listen for the user clicking on the main TB reload button
            let tb_reload_button = parent.document.querySelector(".reload-button");
            if (tb_reload_button != null) {
                tb_reload_button.addEventListener("click", this.reloadRunTags);
            }
        }
    };

}

/**
 * The main entry point of any TensorBoard iframe plugin.
 */
export async function render() {
    let o3dclient = new TensorboardOpen3DPluginClient();
}
