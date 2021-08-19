"""Open3D visualization plugin for Tensorboard"""
import os
import sys
import threading
from collections import OrderedDict
import json
from functools import partial

import numpy as np
from tensorboard.plugins import base_plugin
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.plugin_asset_util import PluginDirectory
import werkzeug
from werkzeug import wrappers
# TODO(ssheorey) Enable operation without TF, but with PyTorch + Tensorboard
import tensorflow.compat.v2 as tf

if sys.platform == 'darwin':
    raise NotImplementedError("Open3D for TensorBoard does not run on macOS.")
import open3d as o3d
# TODO: Check for GPU / EGL else TensorBoard will crash.
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.tensorboard_plugin import metadata
_log = metadata.log
from open3d.visualization.tensorboard_plugin import plugin_data_pb2
# Set window system before the GUI event loop
o3d.visualization.webrtc_server.enable_webrtc()
from open3d.visualization._async_event_loop import _async_event_loop


class LRUCache:
    """Cache storing recent items, i.e. least recently used will be ejected when
    a new item needs to be added to a full cache.
    """

    def __init__(self, max_items=128):
        """
        Args:
            max_items (int): Max items in cache.
        """
        self.cache = OrderedDict()
        self.max_items = max_items
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Retrieve value corresponding to ``key`` from the cache.

        Return:
            Value if ``key`` is found, else None.
        """
        if key not in self.cache:
            self.misses += 1
            _log.debug(str(self))
            return None
        self.cache.move_to_end(key)
        self.hits += 1
        _log.debug(str(self))
        return self.cache[key]

    def put(self, key, value):
        """Add (key, value) pair to the cache. If cache limits are exceeded,
        eject key-value pairs till the cache is within limits."""
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        _log.debug(str(self))

    def clear(self):
        """Invalidate cache."""
        for key in self.cache:
            self.cache.popitem(key)

    def __str__(self):
        return (f"Items: {len(self.cache)}/{self.max_items}, "
                f"Hits: {self.hits}, Misses: {self.misses}")


class Open3DPluginDataReader:
    """Manage TB event data and geometry data for common use by all
    Open3DPluginWindow instances. This is thread safe for simultaneous use by
    multiple browser clients with a multi-threaded web server. Read geometry
    data is cached in memory.

    Args:
        logdir (str): TensorBoard logs directory.
        cache_max_items (int): Max geometry elements to be cached in memory.
    """

    def __init__(self, logdir, cache_max_items=128):
        self.logdir = logdir
        self.event_mux = EventMultiplexer(tensor_size_guidance={
            metadata.PLUGIN_NAME: 0  # Store all metadata in RAM
        }).AddRunsFromDirectory(logdir)
        self._run_to_tags = {}
        self._lock = threading.Lock()
        # Geometry data reading
        self._tensor_events = dict()
        self.geometry_cache = LRUCache(max_items=cache_max_items)
        self._file_handles = {}
        self.reload_events()

    def reload_events(self):
        """Reload event file"""
        self.event_mux.Reload()
        run_tags = self.event_mux.PluginRunToTagToContent(metadata.PLUGIN_NAME)
        with self._lock:
            self._run_to_tags = {
                run: list(tagdict.keys()) for run, tagdict in run_tags.items()
            }
            self._tensor_events = dict()  # Invalidate index
        _log.debug(f"Event data reloaded: {self._run_to_tags}")

    def is_active(self):
        """Do we have any Open3D data to display?"""
        return any(len(tags) > 0 for tags in self._run_to_tags.values())

    @property
    def run_to_tags(self):
        """Locked access to the run_to_tags map."""
        with self._lock:
            return self._run_to_tags

    def tensor_events(self, run):
        with self._lock:
            if run not in self._tensor_events:
                self._tensor_events[run] = {
                    tag: self.event_mux.Tensors(run, tag)
                    for tag in self._run_to_tags[run]
                }
            return self._tensor_events[run]

    def read_geometry(self, run, tag, step, batch_idx, step_to_idx):
        """Geometry reader from msgpack files.
        TODO(ssheorey): Add CRC-32C
        """
        idx = step_to_idx[step]
        metadata_proto = plugin_data_pb2.Open3DPluginData()
        run_tensor_events = self.tensor_events(run)
        metadata_proto.ParseFromString(
            run_tensor_events[tag][idx].tensor_proto.string_val[0])
        filename = metadata_proto.batch_index.filename
        read_location = metadata_proto.batch_index.start_size[batch_idx].start
        read_size = metadata_proto.batch_index.start_size[batch_idx].size
        cache_key = (filename, read_location, read_size, run, tag, step,
                     batch_idx)
        geometry = self.geometry_cache.get(cache_key)
        if geometry is None:  # Read from storage
            data_dir = PluginDirectory(os.path.join(self.logdir, run),
                                       metadata.PLUGIN_NAME)
            # TODO(ssheorey): Make this a bounded LRU dict. Close files if too many open
            if filename not in self._file_handles:
                self._file_handles[filename] = tf.io.gfile.GFile(
                    os.path.join(data_dir, filename), "rb")
                if not self._file_handles[filename].seekable():
                    raise RuntimeError(
                        os.path.join(data_dir, filename) +
                        " does not support seeking. This storage is not supported."
                    )
            self._file_handles[filename].seek(offset=read_location)
            buf = self._file_handles[filename].read(read_size)
            msg_tag, msg_step, geometry = o3d.io.rpc.get_data_from_set_mesh_data_buffer(
                buf)
            if geometry is None:
                raise IOError(f"Geometry {cache_key} reading failed! Possible "
                              "msgpack or TF event file corruption.")
            if tag != msg_tag or step != msg_step:
                _log.warning(
                    f"Mismatch between TF event (tag={tag}, step={step}) and "
                    f"mesgpack (tag={msg_tag}, step={msg_step}) data. Possible data"
                    " corruption.")
            _log.debug(f"Geometry {cache_key} reading successful!")
            self.geometry_cache.put(cache_key, geometry)

        # Fill in properties by reference
        for prop_ref in metadata_proto.property_references:
            prop = plugin_data_pb2.Open3DPluginData.GeometryProperty.Name(
                prop_ref.geometry_property)
            if prop_ref.step_ref >= step:
                _log.warning(
                    f"Incorrect future step reference {prop_ref.step_ref} for"
                    f" property {prop} of geometry at step {step}. Ignoring.")
                continue
            geometry_ref = self.read_geometry(run, tag, prop_ref.step_ref,
                                              batch_idx, step_to_idx)
            # "vertex_normals" -> ["vertex", "normals"]
            prop_map, prop_attribute = prop.split("_")
            # geometry.vertex["normals" = geometry_ref.vertex["normals"]
            getattr(geometry, prop_map)[prop_attribute] = getattr(
                geometry_ref, prop_map)[prop_attribute]

        return geometry


def _postprocess(geometry):
    """Post process geometry before displaying to account for WIP
    Tensor API in Open3D.
    """

    if isinstance(geometry, o3d.t.geometry.PointCloud):
        return geometry
    legacy = geometry.to_legacy()
    # color is FLoat64 but range is [0,255]!
    if isinstance(geometry, o3d.t.geometry.TriangleMesh):
        if legacy.has_vertex_colors():
            legacy.vertex_colors = o3d.utility.Vector3dVector(
                np.asarray(legacy.vertex_colors) / 255)
    elif isinstance(geometry, o3d.t.geometry.TriangleMesh):
        if legacy.has_colors():
            legacy.colors = o3d.utility.Vector3dVector(
                np.asarray(legacy.colors) / 255)
    return legacy


class Open3DPluginWindow:
    """Create and manage a single Open3D WebRTC GUI window.
    """

    # Settings to match tensorboard
    BG_COLOR = gui.Color(0.95, 0.95, 0.95)
    FONT_COLOR = gui.Color(0.5, 0.5, 0.5)

    def __init__(self,
                 window_id,
                 data_reader,
                 title="Open3D for Tensorboard",
                 width=1024,
                 height=768,
                 font_size=12):
        """
        Args:
            window_id (str): Open3D window id. e.g.: "window_0"
            data_reader: Open3DPluginDataReader object to read Tensorboard event
                files and Open3D geometry files.
            title (str): Window title. [Unused in WebRTC]
            width (int): Window width (px).
            height (int): Window height (px).
        """
        self.data_reader = data_reader
        self.run = "."
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

        self.window_id = window_id
        self.scene_widget = None  # Access only through _async_event_loop
        self.geometry_list = []
        self.init_done = threading.Event()  # Notify when WebRTC is ready

        # Register frontend callbacks
        class_name_base = "tensorboard/" + window_id
        o3d.visualization.webrtc_server.register_data_channel_message_callback(
            class_name_base + "/get_run_tags", self._get_run_tags)
        o3d.visualization.webrtc_server.register_data_channel_message_callback(
            class_name_base + "/update_geometry", self._update_geometry)

        _async_event_loop.run_sync(
            partial(self._create_ui, title, width, height))
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
            }
        """
        _log.debug(f"[DC message recv] {message}")
        self.data_reader.reload_events()
        self._validate_run(self.run)
        self._validate_tags(self.tags)
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

    def _validate_tags(self, selected_tags):
        """Validate tags assuming self.run is valid. Use self.tags or first
        valid tag in case selected tags are invalid. Also loads all tensor
        data for validated run-tags combination and unloads data for unselected
        tags.
        """
        selected_tags = [
            t for t in selected_tags
            if t in self.data_reader.run_to_tags[self.run]
        ]
        if len(selected_tags) == 0:
            selected_tags = [
                t for t in self.tags
                if t in self.data_reader.run_to_tags[self.run]
            ]
        if len(selected_tags) == 0 and len(
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
        self.step or first valid step if selected_step is invalid."""
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
        valid. Use self.batch_idx or 0 if selected_batch_idx is invalud.
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
              "messageId": 0,
              "class_name": "tensorboard/window_0/update_geometry",
              "run": "run_0",
              "tags": ["tag_0", "tag_1"],
              "batch_idx": 0,
              "step": 0
            }

        Response:: json

            {
              "messageId": 0,
              "class_name": "tensorboard/window_0/update_geometry",
              "current": {
                "run": "run_0",
                "tags": ["tag_0", "tag_1", ...],
                "step_limits": [0, 100],
                "step": 0
                "batch_size": 8,
                "batch_idx": 0,
                "wall_time": wall_time
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

        self._update_scene()

        # Compose reply
        message["current"] = {
            "run": self.run,
            "tags": self.tags,
            "step_limits": self.step_limits,
            "step": self.step,
            "batch_size": self.batch_size,
            "batch_idx": self.batch_idx,
            "wall_time": self.wall_time,
            "status": "OK"
        }
        return json.dumps(message)

    def _update_scene(self):
        """Update scene by adding / removing geometry elements and redraw.
        """
        new_geometry_list = []
        for tag in self.tags:
            geometry_name = f"{self.run}/{tag}/b{self.batch_idx}/s{self.step}"
            new_geometry_list.append(geometry_name)
            if geometry_name not in self.geometry_list:
                geometry = self.data_reader.read_geometry(
                    self.run, tag, self.step, self.batch_idx, self.step_to_idx)
                _log.debug(f"Displaying geometry {geometry_name}:{geometry}")
                pp_geometry = _postprocess(geometry)
                _async_event_loop.run_sync(
                    partial(self.scene_widget.scene.add_geometry, geometry_name,
                            pp_geometry, self.material))
        for current_item in self.geometry_list:
            if current_item not in new_geometry_list:
                _log.debug(f"Removing geometry {current_item}")
                _async_event_loop.run_sync(
                    partial(self.scene_widget.scene.remove_geometry,
                            current_item))
        self.geometry_list = new_geometry_list

        _async_event_loop.run_sync(self.scene_widget.force_redraw)

        if not self.init_done.is_set():
            self.init_done.set()
        _log.debug(f"Displaying complete!")

    def _create_ui(self, title, width, height):
        """Create new Open3D application window and rendering widgets.

        Args:
            title (str): Window title (unused).
            width (int): Window width.
            height (int): Window height.
        """
        self.window = gui.Application.instance.create_window(
            title, width, height)
        # self.window_uid = o3d.webrtc_server.get_window_uid()
        # Add 3D scene
        self.material = o3d.visualization.rendering.Material()
        self.material.shader = "defaultLit"
        # Set n_pixels displayed for each 3D point, accounting for HiDPI scaling
        self.material.point_size = 2 * self.window.scaling
        self.scene_widget = gui.SceneWidget()
        # sequence_ui.add_child(self.scene_widget)
        self.window.add_child(self.scene_widget)
        self.scene_widget.enable_scene_caching(True)
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])  # White background
        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])


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
    _DEFAULT_DOWNSAMPLING = 100  # meshes per time series
    _PLUGIN_DIRECTORY_PATH_PART = "/data/plugin/" + metadata.PLUGIN_NAME + "/"

    def __init__(self, context):
        """Instantiates Open3D plugin.

        Args:
            context: A `base_plugin.TBContext` instance.
        """
        self._logdir = context.logdir
        self.data_reader = Open3DPluginDataReader(self._logdir)
        self.window_lock = threading.Lock()  # protect _windows and _next_wid
        self._windows = {}
        self._next_wid = 0
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
        # o3d.visualization.webrtc_server.disable_http_handshake()

    @wrappers.Request.application
    def _new_window(self, request):

        if request.is_multiprocess:
            return werkzeug.exceptions.ExpectationFailed(
                "Open3D plugin does not run on a multi-process web server.")

        win_width = min(3840,
                        max(640, int(float(request.args.get('width', 1024)))))
        win_height = min(2400,
                         max(480, int(float(request.args.get('height', 768)))))
        font_size = min(
            20, max(6, int(float(request.args.get('fontsize', '12px')[:-2]))))
        with self.window_lock:
            this_window_id = f"window_{self._next_wid}"
            self._next_wid += 1
            # TODO: Use GetWebRTCUID() instead

        this_window = Open3DPluginWindow(this_window_id, self.data_reader,
                                         "Open3D for Tensorboard", win_width,
                                         win_height, font_size)
        with self.window_lock:
            self._windows[this_window_id] = this_window

        response = f'{{"window_id": "{this_window_id}", "logdir": "{self._logdir}"}}'
        this_window.init_done.wait()  # Wait for WebRTC initialization
        return werkzeug.Response(response, content_type="application/json")

    @wrappers.Request.application
    def _close_window(self, request):

        this_window_id = request.args.get('window_id', "")
        if this_window_id not in self._windows.keys():
            _log.warning(f"Invalid Window ID {this_window_id}")
            return werkzeug.exceptions.NotFound(
                f"Invalid Window ID {this_window_id}")

        self._windows[this_window_id].window.close()
        with self.window_lock:
            del self._windows[this_window_id]
        _log.debug(f"Window {this_window_id} closed.")
        return werkzeug.Response(f"Closed window_{this_window_id}")

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
    def _webrtc_http_api(self, request):
        try:
            entry_point = request.path[(len(self._PLUGIN_DIRECTORY_PATH_PART) -
                                        1):]
            query_string = (b'?' + request.query_string
                            if request.query_string else b'')
            data = request.get_data()
            _log.debug("Request:{}|{}|{}".format(entry_point, query_string,
                                                 data))
            response = o3d.visualization.webrtc_server.call_http_api(
                entry_point, query_string, data)
            _log.debug("Response: {}".format(response))
        except RuntimeError:
            raise werkzeug.exceptions.BadRequest(
                description=f"request is not a function call, ignored: {request}"
            )
        else:
            return werkzeug.Response(response, content_type="application/json")

    @wrappers.Request.application
    def _serve_js(self, request):
        contents = ""
        for js_lib in (os.path.join(self._RESOURCE_PATH, "html", "libs",
                                    "adapter.min.js"),
                       os.path.join(self._RESOURCE_PATH, "html",
                                    "webrtcstreamer.js"),
                       os.path.join(os.path.dirname(__file__), "frontend",
                                    "index.js")):
            with open(js_lib) as infile:
                contents += '\n' + infile.read()
        return werkzeug.Response(contents,
                                 content_type="application/javascript")

    @wrappers.Request.application
    def _serve_css(self, unused_request):
        with open(
                os.path.join(os.path.dirname(__file__), "frontend",
                             "style.css")) as cssfile:
            return werkzeug.Response(cssfile.read(), content_type="text/css")
