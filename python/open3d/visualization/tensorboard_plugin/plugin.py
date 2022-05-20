# ----------------------------------------------------------------------------
# -                        Open3D: www.open3d.org                            -
# ----------------------------------------------------------------------------
# The MIT License (MIT)
#
# Copyright (c) 2018-2021 www.open3d.org
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
# ----------------------------------------------------------------------------
"""Open3D visualization plugin for TensorBoard."""
import os
import sys
import threading
import json

from tensorboard.plugins import base_plugin
import werkzeug
from werkzeug import wrappers

if sys.platform == 'darwin':
    raise NotImplementedError("Open3D for TensorBoard does not run on macOS.")
# TODO: Check for GPU / EGL else TensorBoard will crash.
from open3d.visualization import O3DVisualizer
from open3d.visualization import gui
from open3d.visualization import webrtc_server
from . import plugin_data_pb2
from . import metadata
from .util import Open3DPluginDataReader
from .util import RenderUpdate
from .util import _log


class Open3DPluginWindow:
    """Create and manage a single Open3D WebRTC GUI window."""

    def __init__(self,
                 data_reader,
                 title="Open3D for Tensorboard",
                 width=1024,
                 height=768):
        """
        Args:
            data_reader: Open3DPluginDataReader object to read Tensorboard event
                files and Open3D geometry files.
            title (str): Window title. [Unused in WebRTC]
            width (int): Window width (px).
            height (int): Window height (px).
        """
        self.data_reader = data_reader
        self.run = "."  # fixed for a window once set
        self.tags = []
        self.batch_idx = 0
        self.batch_size = 1
        self.step = 0
        self.step_limits = [0, 0]
        self.wall_time = 0
        self.idx = 0
        self.step_to_idx = dict()
        # self.all_tensor_events[self.tags[0]][prop][self.idx].step == self.step
        self.all_tensor_events = dict()

        self.window = None  # Access only through async_event_loop
        self.geometry_list = []
        self.init_done = threading.Event()  # Notify when WebRTC is ready

        from open3d.visualization.async_event_loop import async_event_loop
        self._gui = async_event_loop
        self._gui.run_sync(self._create_ui, title, width, height)
        self._update_scene()

    def _get_run_tags(self, message):
        """Process message ``get_run_tags`` from the frontend (JS). Reload event
        files, set default state and send response with updated run-tag mapping
        and current state.

        JSON message format:: json

            {
              "messageId": 0,
              "class_name": "tensorboard/window_0/get_run_tags",
            }

        Response:: json

            {
              "messageId": 0,
              "class_name": "tensorboard/window_0/get_run_tags",
              "run_to_tags": {
                "run_0" : ["tag_0", "tag_1", ...],
                "run_1" : ["tag_0", "tag_1", ...],
                ...
              }
              "current": {
                "run": "run_0",
                "tags": ["tag_0", "tag_1", ...],
                "step_limits": [0, 100],
                "step": 0
                "batch_size": 8,
                "batch_idx": 0,
                "wall_time": wall_time
              }
              "colormaps": {
                "RAINBOW" : { value0: rgba0, value1: rgba1, ...},
                "GREYSCALE": { value0: rgba0, value1: rgba1, ...}
              },
              "LabelLUT.Colors": [[0, 0, 0, 255], ..., [224, 224, 224, 255]],
              "status": 'OK'
            }
        """
        _log.debug(f"[DC message recv] {message}")
        self.data_reader.reload_events()
        self._validate_run(self.run)
        self._validate_tags(self.tags, non_empty=True)
        self._validate_step(self.step)
        self._validate_batch_idx(self.batch_idx)
        # Compose reply
        message = json.loads(message)
        message["run_to_tags"] = self.data_reader.run_to_tags
        message["current"] = {
            "run": self.run,
            "tags": self.tags,
            "step_limits": self.step_limits,
            "step": self.step,
            "batch_size": self.batch_size,
            "batch_idx": self.batch_idx,
            "wall_time": self.wall_time
        }
        message["colormaps"] = RenderUpdate.DICT_COLORMAPS
        message["LabelLUTColors"] = RenderUpdate.LABELLUT_COLORS
        message["status"] = 'OK'
        return json.dumps(message)

    def _toggle_settings(self, message):
        """
        JSON message format:: json

            {
              "messageId": 2,
              "window_uid": "window_2",
              "class_name": "tensorboard/window_2/toggle_settings",
            }

        Response:: json

            {
              "messageId": 2,
              "window_uid": "window_2",
              "class_name": "tensorboard/window_2/toggle_settings",
              "open": "true" | "false",
              "status": "OK"
            }
        """
        _log.debug(f"[DC message recv] {message}")
        message = json.loads(message)
        self.window.show_settings = not self.window.show_settings
        self._gui.run_sync(self.window.post_redraw)
        message["status"] = "OK"
        message["open"] = self.window.show_settings
        return json.dumps(message)

    def _validate_run(self, selected_run):
        """Validate selected_run. Use self.run or the first valid run in case
        selected run is invalid. Clear cached events.
        """
        if selected_run not in self.data_reader.run_to_tags:
            selected_run = self.run
        if selected_run not in self.data_reader.run_to_tags:
            selected_run = next(iter(self.data_reader.run_to_tags))
        self.run = selected_run
        self.all_tensor_events = self.data_reader.tensor_events(self.run)

    def _validate_tags(self, selected_tags, non_empty=False):
        """Validate tags assuming self.run is valid. If non_empty is requested,
        validated tags will have the first available tag added if selected_tags
        is empty or invalid. Also loads all tensor data for validated run-tags
        combination and unloads data for unselected tags.
        """
        selected_tags = [
            t for t in selected_tags
            if t in self.data_reader.run_to_tags[self.run]
        ]
        if non_empty and len(selected_tags) == 0 and len(
                self.data_reader.run_to_tags[self.run]) > 0:
            selected_tags = self.data_reader.run_to_tags[
                self.run][:1]  # Only first tag default
        self.tags = selected_tags
        if len(selected_tags) == 0:  # No tags in this run
            return
        self.step_to_idx = {
            tevt.step: idx
            for idx, tevt in enumerate(self.all_tensor_events[self.tags[0]])
        }
        self.step_limits = [min(self.step_to_idx), max(self.step_to_idx)]

    def _validate_step(self, selected_step):
        """Validate step assuming self.run and self.tags are valid. Use
        self.step or first valid step if selected_step is invalid.
        """
        if len(self.tags) == 0:  # No tags in this run
            return
        if selected_step not in self.step_to_idx:
            selected_step = self.step
        if selected_step not in self.step_to_idx:
            selected_step = self.step_limits[0]  # Set to first step
        self.step = selected_step
        self.idx = self.step_to_idx[self.step]
        self.wall_time = self.all_tensor_events[self.tags[0]][
            self.idx].wall_time

        metadata_proto = plugin_data_pb2.Open3DPluginData()
        metadata_proto.ParseFromString(self.all_tensor_events[self.tags[0]][
            self.idx].tensor_proto.string_val[0])
        self.batch_size = len(metadata_proto.batch_index.start_size)

    def _validate_batch_idx(self, selected_batch_idx):
        """Validate batch_idx assuming self.run, self.tags and self.step are
        valid. Use self.batch_idx or 0 if selected_batch_idx is invalid.
        """
        if len(self.tags) == 0:  # No tags in this run
            return
        if selected_batch_idx < 0 or selected_batch_idx >= self.batch_size:
            selected_batch_idx = self.batch_idx
        if selected_batch_idx < 0 or selected_batch_idx >= self.batch_size:
            selected_batch_idx = 0
        self.batch_idx = selected_batch_idx

    def _update_geometry(self, message):
        """Process an update_geometry message from the frontend (JS). Validate
        message, update state, update scene and send response with validated
        state.

        JSON message format:: json

            {
              "messageId": 2,
              "class_name": "tensorboard/window_0/update_geometry",
              "run": "run_0",
              "tags": ["tag_0", "tag_1"],
              "render_state": {
                "tag_0": null,
                "tag_1": {
                      "property": Any vertex_*PROPERTY*
                      "index": 0,
                      "shader": , "color" (defaultUnlit), "solid" (unlitSolidColor),
                      "labels", (unlitGradient + LUT), + unlitLine for BB
                      (unlitGradient + GRADIENT) "rainbow" , "greyscale",
                      "colormap": [
                        label (LUT) / value (GRADIENT):  [r, g, b, a] (all uint8),
                        ...
                      ]
                    }
                },
              "batch_idx": 0,
              "step": 0
            }

        Response:: json

            {
              "messageId": 2,
              "class_name": "tensorboard/window_0/update_geometry",
              "current": {
                "run": "run_0",
                "tags": ["tag_0", "tag_1"],
                "render_state": {
                  "tag_0": {
                        "property": Any vertex_*PROPERTY*
                        "index": 0,
                        "shader": , "color" (defaultUnlit), "solid" (unlitSolidColor),
                        "labels", (unlitGradient + LUT), + unlitLine for BB
                        (unlitGradient + GRADIENT) "rainbow" , "greyscale",
                        "colormap": [
                          label (LUT) / value (GRADIENT):  [r, g, b, a] (all uint8),
                  },
                  "tag_1": {
                        "property": Any vertex_*PROPERTY*
                        "index": 0,
                        "shader": , "color" (defaultUnlit), "solid" (unlitSolidColor),
                        "labels", (unlitGradient + LUT), + unlitLine for BB
                        (unlitGradient + GRADIENT) "rainbow" , "greyscale",
                        "colormap": [
                          label (LUT) / value (GRADIENT):  [r, g, b, a] (all uint8),
                          ...
                        ]
                      }
                  },
                "step_limits": [0, 100],
                "step": 0
                "batch_size": 8,
                "batch_idx": 0,
                "wall_time": wall_time
              }
              "tags_properties_shapes": {
                  "tag_0": { "prop0": 3, "prop1": 1, ...},
                  "tag_1": { "prop0": 3, "prop2": 2, ...},
              ...},
              "tag_label_to_names": {
                  "tag_1": { label (int): name (str), ...}
                  ...
              }
              "status": OK
            }
        """
        _log.debug(f"[DC message recv] {message}")
        message = json.loads(message)
        self._validate_run(message["run"])
        self._validate_tags(message["tags"])
        self._validate_step(int(message["step"]))
        self._validate_batch_idx(int(message["batch_idx"]))

        status = self._update_scene(message)
        message["status"] = status
        message["current"] = {
            "run": self.run,
            "tags": self.tags,
            "render_state": message["render_state"],  # from _update_scene()
            "step_limits": self.step_limits,
            "step": self.step,
            "batch_size": self.batch_size,
            "batch_idx": self.batch_idx,
            "wall_time": self.wall_time,
        }
        # Compose reply
        message["tags_properties_shapes"] = {
            tag: self.data_reader.runtag_prop_shape[self.run][tag]
            for tag in self.tags
        }
        for key in ("run", "tags", "batch_idx", "step", "render_state"):
            message.pop(key, None)
        return json.dumps(message)

    def _update_scene(self, message=None):
        """Update scene by adding / removing geometry elements and redraw.
        message["render_state"][tag] if present, is the initial render state for
        the tag to be added. If not provided, this will be filled with the
        render state after the update.
        """
        if message is None:
            message = {"render_state": {}}
        status = ""
        new_geometry_list = []
        tag_label_to_names = message.setdefault("tag_label_to_names", dict())
        for tag in self.tags:
            if tag not in tag_label_to_names:
                tag_label_to_names[tag] = self.data_reader.get_label_to_names(
                    self.run, tag)
            message_tag = dict()
            if tag in message["render_state"]:
                message_tag["render_state"] = message["render_state"][tag]
            render_update = RenderUpdate(self.window.scaling, message_tag,
                                         tag_label_to_names[tag])
            geometry_name = ", ".join(
                str(x) for x in (self.run, tag, self.batch_idx, self.step))
            new_geometry_list.append(geometry_name)
            if geometry_name not in self.geometry_list:
                try:
                    geometry, inference_data_proto = self.data_reader.read_geometry(
                        self.run, tag, self.step, self.batch_idx,
                        self.step_to_idx)
                    _log.debug(
                        f"Displaying geometry {geometry_name}:{geometry}")
                    render_update.apply(self.window, geometry_name, geometry,
                                        inference_data_proto)
                    message["render_state"][
                        tag] = render_update.get_render_state()
                except IOError as err:
                    new_geometry_list.pop()
                    err_msg = f"Error reading {geometry_name}: {err}"
                    status = '\n'.join((status, err_msg))
                    _log.warning(__name__, err_msg)

        for current_item in self.geometry_list:
            if current_item not in new_geometry_list:
                _log.debug(f"Removing geometry {current_item}")
                self._gui.run_sync(self.window.remove_geometry, current_item)
        # Reset view only if scene changed from empty -> not empty
        if len(self.geometry_list) == 0 and len(new_geometry_list) > 0:
            self._gui.run_sync(self.window.reset_camera_to_default)
        else:
            self._gui.run_sync(self.window.post_redraw)
        self.geometry_list = new_geometry_list

        if not self.init_done.is_set():
            self.init_done.set()
        _log.debug("Displaying complete!")
        return "OK" if len(status) == 0 else status[1:]

    def _create_ui(self, title, width, height):
        """Create new Open3D application window and rendering widgets. Must run
        in the GUI thread.

        Args:
            title (str): Window title (unused).
            width (int): Window width.
            height (int): Window height.
        """
        self.window = O3DVisualizer(title, width, height)
        self.window.show_menu(False)
        self.window.scene.downsample_threshold = 400000
        self.window.set_background((1, 1, 1, 1), None)  # White background
        self.window.show_skybox(False)
        self.window.line_width = int(3 * self.window.scaling)
        # Register frontend callbacks
        class_name_base = "tensorboard/" + self.window.uid
        webrtc_server.register_data_channel_message_callback(
            class_name_base + "/get_run_tags", self._get_run_tags)
        webrtc_server.register_data_channel_message_callback(
            class_name_base + "/update_geometry", self._update_geometry)
        webrtc_server.register_data_channel_message_callback(
            class_name_base + "/toggle_settings", self._toggle_settings)
        gui.Application.instance.add_window(self.window)


class Open3DPlugin(base_plugin.TBPlugin):
    """Open3D plugin for TensorBoard.

    Subclasses should have a trivial constructor that takes a TBContext
    argument. Any operation that might throw an exception should either be
    done lazily or made safe with a TBLoader subclass, so the plugin won't
    negatively impact the rest of TensorBoard.

    Fields:
      plugin_name: The plugin_name will also be a prefix in the http
        handlers, e.g. `data/plugins/$PLUGIN_NAME/$HANDLER` The plugin
        name must be unique for each registered plugin, or a ValueError
        will be thrown when the application is constructed. The plugin
        name must only contain characters among [A-Za-z0-9_.-], and must
        be nonempty, or a ValueError will similarly be thrown.
    """
    plugin_name = metadata.PLUGIN_NAME
    _RESOURCE_PATH = os.path.join(os.path.dirname(__file__), "..", "..",
                                  "resources")
    _PLUGIN_DIRECTORY_PATH_PART = "/data/plugin/" + metadata.PLUGIN_NAME + "/"
    # Browser security: Do not guess response content type by inspection.
    _HEADERS = [("X-Content-Type-Options", "nosniff")]
    _ERROR_RESPONSE = werkzeug.Response(headers=_HEADERS)

    def __init__(self, context):
        """Instantiates Open3D plugin.

        Args:
            context: A `base_plugin.TBContext` instance.
        """
        self._logdir = context.logdir
        self.data_reader = Open3DPluginDataReader(self._logdir)
        self.window_lock = threading.Lock()  # protect self._windows
        self._http_api_lock = threading.Lock()
        self._windows = {}
        self._dummy_window = None
        self._gui = None

    def _start_gui(self):
        webrtc_server.disable_http_handshake()
        webrtc_server.enable_webrtc()
        from open3d.visualization.async_event_loop import async_event_loop
        self._gui = async_event_loop
        # Dummy window to ensure GUI remains active even if all user windows are
        # closed.
        self._dummy_window = self._gui.run_sync(
            gui.Application.instance.create_window, "Open3D Dummy Window", 32,
            32)
        webrtc_server.register_data_channel_message_callback(
            "tensorboard/show_hide_axes", self._show_hide)
        webrtc_server.register_data_channel_message_callback(
            "tensorboard/show_hide_ground", self._show_hide)
        webrtc_server.register_data_channel_message_callback(
            "tensorboard/sync_view", self._sync_view)
        webrtc_server.register_data_channel_message_callback(
            "tensorboard/update_rendering", self._update_rendering)

    def _show_hide(self, message):
        """
        JSON message format:: json

            {
              "messageId": 3,
              "class_name": "tensorboard/show_hide_axes",
              "window_uid_list": ["window_4", "window_5"],
              show: false
            }

        All correct window_ids will have their axes shown / hidden. A return
        status message is provided.
        Response:: json

            {
              "messageId": 3,
              "class_name": "tensorboard/show_hide_axes",
              "window_uid_list": ["window_5"],
              show: false,
              status: "Bad window"   # or "OK"
            }
        """
        _log.debug(f"[DC message recv] {message}")
        message = json.loads(message)
        show = bool(message["show"])
        message["status"] = 'OK'
        window_uid_list = [
            str(w)
            for w in message["window_uid_list"]
            if str(w) in self._windows
        ]
        for window_uid in window_uid_list:
            window = self._windows[window_uid].window
            if message["class_name"] == "tensorboard/show_hide_axes":
                window.show_axes = show
            elif message["class_name"] == "tensorboard/show_hide_ground":
                window.show_ground = show
            self._gui.run_sync(window.post_redraw)
        message["window_uid_list"] = window_uid_list
        return json.dumps(message)

    def _sync_view(self, message):
        """
        JSON message format:: json

            {
              "messageId": 0,
              "window_uid_list": ["window_2", "window_3"], ,
              "class_name": "tensorboard/sync_view",
            }

        Response:: json

            {
              "messageId": 0,
              "window_uid_list": ["window_2", "window_3"], ,
              "class_name": "tensorboard/sync_view",
              "status": "OK"
            }
        """
        _log.debug(f"[DC message recv] {message}")
        message = json.loads(message)
        message["status"] = 'OK'
        window_uid_list = [
            str(w)
            for w in message["window_uid_list"]
            if str(w) in self._windows
        ]
        if len(window_uid_list) < 2:
            return json.dumps(message)
        target_camera = self._windows[window_uid_list[0]].window.scene.camera

        def copy_target_camera():
            for window_uid in window_uid_list[1:]:
                o3dvis = self._windows[window_uid].window
                o3dvis.scene.camera.copy_from(target_camera)
                o3dvis.post_redraw()

        self._gui.run_sync(copy_target_camera)
        message["window_uid_list"] = window_uid_list
        return json.dumps(message)

    def _update_rendering(self, message):
        """Process an update_rendering message from the frontend (JS). Validate
        message, update state, update scene and send response with validated
        state.

        JSON message format:: json

            {
              "messageId": 3,
              "class_name": "tensorboard/update_rendering",
              "window_uid_list": ["window_2",  "window_3"],
              "tag": "tag_1",
              "render_state": {
                  "property": Any vertex_*PROPERTY*
                  "index": 0,
                  "shader": , "color" (defaultUnlit), "solid" (unlitSolidColor),
                  "labels", (unlitGradient + LUT), + unlitLine for BB
                  (unlitGradient + GRADIENT) "rainbow" , "greyscale",
                  "colormap": [
                    label (LUT) / value (GRADIENT):  [r, g, b, a] (all uint8),
                    ...
                  ]
              },
              "updated": ["property", "shader", "colormap"]
            }

        Response:: json

            {
              "messageId": 3,
              "class_name": "tensorboard/update_rendering",
              "window_uid_list": ["window_2",  "window_3"],
              "tag": "tag_1",
              "all_properties": [All properties from all tags in window_uid_list],
              "render_state": {
                  "property": Any vertex_*PROPERTY* or None,
                  "index": 0,
                  "range": [min, max],
                  "shader": , "color" (defaultUnlit), "solid" (unlitSolidColor),
                  "labels", (unlitGradient + LUT),
                  (unlitGradient + GRADIENT) "rainbow" , "greyscale",
                  "colormap": [
                    label (LUT) / value (GRADIENT):  [r, g, b, a],
                    ...
                  ]
              }
              "updated": ["property", "shader", "colormap"],
              "status": OK
            }
        """
        _log.debug(f"[DC message recv] {message}")
        message = json.loads(message)
        window_uid_list = [
            str(w)
            for w in message["window_uid_list"]
            if str(w) in self._windows
        ]
        render_update = None  # Initialize on first use

        for window_uid in window_uid_list:
            plugin_window = self._windows[window_uid]
            o3dvis = plugin_window.window
            for geometry_name in plugin_window.geometry_list:
                *run, tag, batch_idx, step = geometry_name.split(', ')
                run = ', '.join(run)
                if tag != message["tag"]:
                    continue
                if render_update is None:
                    render_update = RenderUpdate(
                        o3dvis.scaling, message,
                        self.data_reader.get_label_to_names(run, tag))
                geometry, inference_data_proto = self.data_reader.read_geometry(
                    run, tag, int(step), int(batch_idx),
                    plugin_window.step_to_idx)
                render_update.apply(o3dvis, geometry_name, geometry,
                                    inference_data_proto)

        message["window_uid_list"] = window_uid_list
        message["render_state"] = render_update.get_render_state()
        message["status"] = 'OK'
        return json.dumps(message)

    def get_plugin_apps(self):
        """Returns a set of WSGI applications that the plugin implements.

        Each application gets registered with the tensorboard app and is served
        under a prefix path that includes the name of the plugin.

        Returns:
          A dict mapping route paths to WSGI applications. Each route path
          should include a leading slash.
        """
        return {
            "/index.js": self._serve_js,
            "/webrtcstreamer.js": self._serve_js,
            "/adapter.min.js": self._serve_js,
            "/style.css": self._serve_css,
            "/new_window": self._new_window,
            "/close_window": self._close_window,
            "/api/*": self._webrtc_http_api
        }

    def is_active(self):
        """Determines whether this plugin is active.

        A plugin may not be active for instance if it lacks relevant data. If a
        plugin is inactive, the frontend may avoid issuing requests to its
        routes.

        Returns:
          A boolean value. Whether this plugin is active.
        """
        return self.data_reader.is_active()

    def frontend_metadata(self):
        """Defines how the plugin will be displayed on the frontend.

        The base implementation returns a default value. Subclasses
        should override this and specify either an `es_module_path` or
        (for legacy plugins) an `element_name`, and are encouraged to
        set any other relevant attributes.
        """
        return base_plugin.FrontendMetadata(es_module_path="/index.js")
        # es_module_path: ES module to use as an entry point to this plugin.
        #     A `str` that is a key in the result of `get_plugin_apps()`, or
        #     `None` for legacy plugins bundled with TensorBoard as part of
        #     `webfiles.zip`. Mutually exclusive with legacy `element_name`

    @wrappers.Request.application
    def _new_window(self, request):
        """Create a new WebRTC window on request."""
        if not self.data_reader.is_active():  # no data
            response = json.dumps({"window_id": -1, "logdir": self._logdir})
            return werkzeug.Response(response,
                                     content_type="application/json",
                                     headers=self._HEADERS)

        if self._gui is None:
            self._start_gui()

        win_width = min(3840,
                        max(640, int(float(request.args.get('width', 1024)))))
        win_height = min(2400,
                         max(480, int(float(request.args.get('height', 768)))))

        this_window = Open3DPluginWindow(self.data_reader,
                                         "Open3D for Tensorboard", win_width,
                                         win_height)
        with self.window_lock:
            self._windows[this_window.window.uid] = this_window

        response = json.dumps({
            "window_id": this_window.window.uid,
            "logdir": self._logdir
        })
        this_window.init_done.wait()  # Wait for WebRTC initialization
        return werkzeug.Response(response,
                                 content_type="application/json",
                                 headers=self._HEADERS)

    @wrappers.Request.application
    def _close_window(self, request):
        """Close a WebRTC window on request."""
        this_window_id = request.args.get('window_id', "")
        if this_window_id not in self._windows.keys():
            _log.warning(f"Invalid Window ID {this_window_id}")
            return werkzeug.exceptions.NotFound(
                f"Invalid Window ID {this_window_id}",
                response=self._ERROR_RESPONSE)

        self._gui.run_sync(self._windows[this_window_id].window.close)
        with self.window_lock:
            del self._windows[this_window_id]
        _log.debug(f"Window {this_window_id} closed.")
        return werkzeug.Response(f"Closed window {this_window_id}",
                                 content_type="text/plain",
                                 headers=self._HEADERS)

    @wrappers.Request.application
    def _webrtc_http_api(self, request):
        """Relay WebRTC connection setup messages coming as HTTP requests."""
        try:
            entry_point = request.path[(len(self._PLUGIN_DIRECTORY_PATH_PART) -
                                        1):]
            query_string = (b'?' + request.query_string
                            if request.query_string else b'')
            data = request.get_data()
            if len(self._windows) == 0:
                raise werkzeug.exceptions.BadRequest(
                    description="No windows exist to service this request: "
                    f"{request}",
                    response=self._ERROR_RESPONSE)

            with self._http_api_lock:
                response = webrtc_server.call_http_api(entry_point,
                                                       query_string, data)

        except RuntimeError:
            raise werkzeug.exceptions.BadRequest(
                description="Request is not a function call, ignored: "
                f"{request}",
                response=self._ERROR_RESPONSE)
        else:
            return werkzeug.Response(response,
                                     content_type="application/json",
                                     headers=self._HEADERS)

    @wrappers.Request.application
    def _serve_js(self, request):
        """Serve frontend JS files on request."""
        if request.is_multiprocess:
            return werkzeug.exceptions.ExpectationFailed(
                "Open3D plugin does not run on a multi-process web server.",
                response=self._ERROR_RESPONSE)

        js_file = request.path.split('/')[-1]
        if js_file == "index.js":
            js_file = os.path.join(os.path.dirname(__file__), "frontend",
                                   js_file)
        elif js_file == "webrtcstreamer.js":
            js_file = os.path.join(self._RESOURCE_PATH, "html", js_file)
        elif js_file == "adapter.min.js":
            js_file = os.path.join(self._RESOURCE_PATH, "html", "libs", js_file)
        else:
            raise werkzeug.exceptions.NotFound(
                description=f"JS file {request.path} does not exist.",
                response=self._ERROR_RESPONSE)

        with open(js_file) as infile:
            return werkzeug.Response(infile.read(),
                                     content_type="application/javascript",
                                     headers=self._HEADERS)

    @wrappers.Request.application
    def _serve_css(self, unused_request):
        """Serve frontend stylesheet on request."""
        with open(
                os.path.join(os.path.dirname(__file__), "frontend",
                             "style.css")) as cssfile:
            return werkzeug.Response(cssfile.read(),
                                     content_type="text/css",
                                     headers=self._HEADERS)
