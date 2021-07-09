"""Open3D visualization plugin for Tensorboard"""
import os
import threading
import time
from collections import namedtuple
import logging as log

from tensorboard import errors
from tensorboard.plugins import base_plugin
from tensorboard.util import tensor_util
from tensorboard.data import provider
from tensorboard import plugin_util
from tensorboard.backend import http_util
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
import werkzeug
from werkzeug import wrappers

import open3d as o3d
import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
from open3d.visualization.tensorboard_plugin import metadata
# Note: the _AsyncEventLoop is started whenever this module is imported.
from open3d.visualization._async_event_loop import in_async_event_loop, _async_event_loop

import ipdb

WindowData = namedtuple('WindowData', ['run', 'tag', 'step', 'batch_idx'])


class Open3DPluginWindow(object):

    def __init__(self,
                 window_id,
                 event_mux,
                 title="Open3D",
                 width=1024,
                 height=768):
        self.event_mux = event_mux
        self.run = None
        self.tag = None
        self.batch_idx = 0
        self.step = 0
        self.wall_time = 0
        self.idx = 0  # self.all_tensor_events[self.idx].step == self.step
        self.all_tensor_events = {
            prop: None for prop in metadata._MESH_PROPERTIES
        }
        self.mesh = o3d.geometry.TriangleMesh()

        self.window_id = window_id
        self.event_mux.Reload()
        self.run_to_tags = self.event_mux.PluginRunToTagToContent(
            metadata.PLUGIN_NAME)

        self.cv_init_done = threading.Condition()  # Notify when WebRTC is ready
        self.init_done = False

        self._create_ui(title, width, height)
        self._update_scene()

    @in_async_event_loop
    def _create_ui(self, title, width, height):
        self.window = gui.Application.instance.create_window(
            title, width, height)

        layout = gui.Horiz()
        self.window.add_child(layout)
        layout.background_color = gui.Color(0.95, 0.95, 0.95)

        # Setup the run selector
        run_selector = gui.Vert()
        layout.add_child(run_selector)
        run_selector.add_child(gui.Label('Runs'))
        self.run_list = gui.ListView()
        run_selector.add_child(self.run_list)
        run_list = list(self.run_to_tags.keys())
        self.run_list.set_items(run_list)
        try:
            self.run_list.selected_index = run_list.index(self.run)
        except ValueError:  # self.run is None or not in list
            self.run_list.selected_index = 0
            self.run = run_list[0]
        self.run_list.set_on_selection_changed(self._on_run_select)
        logdir = gui.Label(self._logdir)
        logdir.text_color = gui.Color(0.5, 0.5, 0.5)
        run_selector.add_child(logdir)

        # Setup the tag selector
        tag_selector = gui.Vert()
        layout.add_child(tag_selector)
        tag_selector.add_child(gui.Label('Tags'))
        self.tag_list = gui.ListView()
        tag_selector.add_child(self.tag_list)
        self.tags = [
            tag[:-9]
            for tag in self.run_to_tags[self.run]
            if tag.endswith('_vertices')
        ]
        self.tag_list.set_items(self.tags)
        try:
            self.tag_list.selected_index = self.tags.index(self.tag)
        except ValueError:  # self.tag is None or not in list
            self.tag_list.selected_index = 0
            self.tag = self.tags[0]
        self.tag_list.set_on_selection_changed(self._on_tag_select)

        for prop in metadata._MESH_PROPERTIES:
            if self.tag + '_' + prop in self.run_to_tags[self.run]:
                self.all_tensor_events[prop] = self.event_mux.Tensors(
                    self.run, self.tag + '_' + prop)  # slow
        self.step_to_idx = {
            vtx_te.step: idx
            for idx, vtx_te in enumerate(self.all_tensor_events['vertices'])
        }

        self.step = self.all_tensor_events['vertices'][0].step
        self.wall_time = self.all_tensor_events['vertices'][0].wall_time
        self.batch_size = self.all_tensor_events['vertices'][
            0].tensor_shape.dim[0].size

        # Setup the step selector
        sequence_ui = gui.Vert()
        layout.add_child(sequence_ui)
        step = gui.Horiz()
        sequence_ui.add_child(step)
        step.add_child(gui.Label("Step"))
        self.step_slider = gui.Slider(gui.Slider.INT)
        step.add_child(self.step_slider)
        # Assume steps are monotonic
        self.step_limits = (self.all_tensor_events['vertices'][0].step,
                            self.all_tensor_events['vertices'][-1].step)
        self.step_slider.set_limits(*self.step_limits)
        self.step_slider.set_on_value_changed(self._on_step_changed)

        # Setup the batch index selector
        self.batch_idx = 0
        batch_idx = gui.Horiz()
        sequence_ui.add_child(batch_idx)
        batch_idx.add_child(gui.Label("Batch index"))
        self.batch_idx_slider = gui.Slider(gui.Slider.INT)
        batch_idx.add_child(self.batch_idx_slider)
        self.batch_idx_slider.set_limits(0, self.batch_size)
        self.batch_idx_slider.set_on_value_changed(self._on_batch_idx_changed)

        # Add 3D scene
        self.scene_widget = gui.SceneWidget()
        sequence_ui.add_child(self.scene_widget)
        self.scene_widget.enable_scene_caching()
        self.scene_widget.scene = rendering.Open3DScene(self.window.renderer)
        self.scene_widget.scene.set_background([1, 1, 1, 1])  # White background
        self.scene_widget.scene.set_lighting(
            rendering.Open3DScene.LightingProfile.SOFT_SHADOWS, [0, -6, 0])

    def _on_run_select(self, selected_run, is_double_click):
        self.run = selected_run
        self.tags = [
            tag[:-9]
            for tag in self.run_to_tags[self.run]
            if tag.endswith('_vertices')
        ]

        # Update tags, steps, batch, validate and update scene
        self._on_tag_select(self.tag, False)

    def _on_tag_select(self, selected_tag, unused_is_double_click):
        if selected_tag not in self.tags:
            log.warn(f"Tag {selected_tag} not in current run. Skipping.")
            self._update_selectors()
            return
        self.tag = selected_tag
        for prop in metadata._MESH_PROPERTIES:
            if self.tag + '_' + prop in self.run_to_tags[self.run]:
                self.all_tensor_events[prop] = self.event_mux.Tensors(
                    self.run, self.tag + '_' + prop)  # slow
            else:
                self.all_tensor_events[prop] = None

        self.step_to_idx = {
            vtx_te.step: idx
            for idx, vtx_te in enumerate(self.all_tensor_events['vertices'])
        }
        # Assume steps are monotonic
        self.step_limits = (self.all_tensor_events['vertices'][0].step,
                            self.all_tensor_events['vertices'][-1].step)

        # Validate step, batch_idx and update scene
        self._on_step_changed(self.step)

    def _on_step_changed(self, new_step):
        if new_step not in self.step_to_idx:
            log.warn(
                f"Step {new_step} missing in event file for run {self.run}, tag {self.tag}. Skipping."
            )
            self._update_selectors()
            return
        self.step = new_step
        self.idx = self.step_to_idx[self.step]
        self.wall_time = self.all_tensor_events['vertices'][self.idx].wall_time
        self.batch_size = self.all_tensor_events['vertices'][
            self.idx].tensor_shape.dim[0].size
        # Check if batch_idx is still valid and update scene
        self._on_batch_idx_changed(self.batch_idx)

    def _on_batch_idx_changed(self, new_batch_idx):
        this_batch_size = self.all_tensor_events['vertices'][
            self.idx].tensor_shape.dim[0].size
        self.batch_idx = max(0, min(new_batch_idx, this_batch_size - 1))
        self._update_selectors()
        self._update_scene()

    @in_async_event_loop
    def _update_selectors(self):
        self.tag_list.set_items(self.tags)
        self.step_slider.set_limits(*self.step_limits)
        self.batch_idx_slider.set_limits(0, self.batch_size)
        self.window.post_redraw()

    @in_async_event_loop
    def _update_scene(self):

        self.mesh.vertices = o3d.utility.Vector3dVector(
            tensor_util.make_ndarray(self.all_tensor_events['vertices'][
                self.idx].tensor_proto)[self.batch_idx, ...])
        for prop in metadata._MESH_PROPERTIES:
            if self.all_tensor_events[prop] is not None:
                prop_value = tensor_util.make_ndarray(
                    self.all_tensor_events[prop][self.idx].tensor_proto)[
                        self.batch_idx, ...]
                setattr(
                    self.mesh, prop,
                    o3d.utility.Vector3iVector(prop_value.astype(int)) if prop
                    == 'triangles' else o3d.utility.Vector3dVector(prop_value))
        if self.scene_widget.has_geometry(self.tag):
            self.scene_widget.remove_geometry(self.tag)
        self.scene_widget.add_geometry(self.tag, self.mesh)
        self.scene_widget.force_redraw()

        if not self.init_done:
            with self.cv_init_done:
                self.init_done = True
                self.cv_init_done.notify_all()


class Open3DPlugin(base_plugin.TBPlugin):
    """Open3D plugin for TensorBoard

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
    _MAX_SAMPLES_IN_RAM = 10
    _PLUGIN_DIRECTORY_PATH_PART = "/data/plugin/" + metadata.PLUGIN_NAME + "/"

    def __init__(self, context):
        """Instantiates Open3D plugin.

        Args:
            context: A `base_plugin.TBContext` instance.
        """
        self._data_provider = context.data_provider
        self._downsample_to = (context.sampling_hints or
                               {}).get(self.plugin_name,
                                       self._DEFAULT_DOWNSAMPLING)
        self._logdir = context.logdir
        self.event_mux = EventMultiplexer(tensor_size_guidance={
            metadata.PLUGIN_NAME: self._MAX_SAMPLES_IN_RAM
        }).AddRunsFromDirectory(self._logdir)
        self._windows = []
        self._next_wid = 0
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        # o3d.visualization.webrtc_server.disable_http_handshake()
        o3d.visualization.webrtc_server.enable_webrtc()

    @wrappers.Request.application
    def _new_window(self, request):

        assert not request.is_multithread, "Open3D plugin is not currently multi-thread safe."
        assert not request.is_multiprocess, "Open3D plugin is not currently multi-process safe."

        win_width = min(3840, max(640, request.args.get('width', 1024)))
        win_height = min(2400, max(480, request.args.get('height', 768)))
        self._windows.append(
            Open3DPluginWindow(self._next_wid, self.event_mux,
                               "Open3D for Tensorboard", win_width, win_height))

        response = str(self._next_wid)
        self._next_wid += 1

        # Wait for WebRTC initialization
        with self._windows[-1].cv_init_done:
            self._windows[-1].cv_init_done.wait_for(
                predicate=lambda: self._windows[-1].init_done)

        print("--WebRTC setup should be complete by now-->\n")
        return werkzeug.Response(response, content_type="text/plain")

    # def _instance_tag_content(self, ctx, experiment, run, instance_tag):
    #     """Gets the `MeshPluginData` proto for an instance tag."""
    #     results = self._data_provider.list_tensors(
    #         ctx,
    #         experiment_id=experiment,
    #         plugin_name=metadata.PLUGIN_NAME,
    #         run_tag_filter=provider.RunTagFilter(runs=[run],
    #                                              tags=[instance_tag]),
    #     )
    #     return results[run][instance_tag].plugin_content

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
            "/window": self._new_window,
            "/tags": self._serve_tags,
            "/api/*": self._webrtc_http_api,
        }

    def is_active(self):
        """Determines whether this plugin is active.

        A plugin may not be active for instance if it lacks relevant data. If a
        plugin is inactive, the frontend may avoid issuing requests to its routes.

        Returns:
          A boolean value. Whether this plugin is active.
        """
        return True
        # return bool(self._multiplexer.PluginRunToTagToContent(self.plugin_name))

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
            print("Request:{}|{}|{}".format(entry_point, query_string, data))
            response = o3d.visualization.webrtc_server.call_http_api(
                entry_point, query_string, data)
            print("Response: {}", response)
        except:
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
        log.info(contents)
        return werkzeug.Response(contents,
                                 content_type="application/javascript")

    @wrappers.Request.application
    def _serve_tags(self, request):
        """Serves run to tag info.

        Frontend clients can use the Multiplexer's run+tag structure to request
        data for a specific run+tag. Responds with a map of the form:
        {runName: [tagName, tagName, ...]}
        """
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        lp = self._data_provider.list_plugins(ctx, experiment_id=experiment)
        em = self._data_provider.experiment_metadata(ctx,
                                                     experiment_id=experiment)
        log.info(lp)
        log.info(em)
        ipdb.set_trace()
        run_tag_mapping = self._data_provider.list_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )
        run_info = {run: list(tags) for (run, tags) in run_tag_mapping.items()}
        log.info(run_info)
        blob_run_tag_mapping = self._data_provider.list_blob_sequences(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
        )
        blob_run_info = {
            run: list(tags) for (run, tags) in blob_run_tag_mapping.items()
        }
        log.info(blob_run_info)
        ipdb.set_trace()

        return http_util.Respond(request, run_info, "application/json")

    # @wrappers.Request.application
    # def _serve_static_file(self, request):
    #     """Returns a resource file from the static asset directory.

    #     Requests from the frontend have a path in this form:
    #     /data/plugin/open3d/resources/foo
    #     This serves the appropriate asset: __file__/../../resources/foo.

    #     Checks the normpath to guard against path traversal attacks.
    #     """
    #     static_path_part = request.path[len(self._PLUGIN_DIRECTORY_PATH_PART):]
    #     resource_name = os.path.normpath(
    #         os.path.join(*static_path_part.split("/")))
    #     if not resource_name.startswith("resources" + os.path.sep):
    #         return http_util.Respond(request,
    #                                  "Not found",
    #                                  "text/plain",
    #                                  code=404)

    #     resource_path = os.path.join(self._RESOURCE_PATH,                                  resource_name)
    #     with open(resource_path, "rb") as read_file:
    #         mimetype = mimetypes.guess_type(resource_path)[0]
    #         return http_util.Respond(request,
    #                                  read_file.read(),
    #                                  content_type=mimetype)

    def tensors_impl(self, ctx, experiment, tag, run):
        """Returns tensor data for the specified tag and run.

        For details on how to use tags and runs, see
        https://github.com/tensorflow/tensorboard#tags-giving-names-to-data

        Args:
          tag: string
          run: string

        Returns:
          A list of TensorEvents - tuples containing 3 numbers describing
          entries in the data series.

        Raises:
          NotFoundError if there are no tensors data for provided `run` and
          `tag`.
        """
        all_tensors = self._data_provider.read_tensors(
            ctx,
            experiment_id=experiment,
            plugin_name=metadata.PLUGIN_NAME,
            downsample=self._DEFAULT_DOWNSAMPLING,
            run_tag_filter=provider.RunTagFilter(runs=[run], tags=[tag]),
        )
        tensors = all_tensors.get(run, {}).get(tag, None)
        log.info(tensors)
        if tensors is None:
            raise errors.NotFoundError("No tensors data for run=%r, tag=%r" %
                                       (run, tag))
        return [(x.wall_time, x.step, x.value) for x in tensors]

    @wrappers.Request.application
    def _serve_greetings(self, request):
        """Given a tag and single run, return array of TensorEvents."""
        tag = request.args.get("tag")
        run = request.args.get("run")
        ctx = plugin_util.context(request.environ)
        experiment = plugin_util.experiment_id(request.environ)
        body = self.tensors_impl(ctx, experiment, tag, run)
        return http_util.Respond(request, body, "application/json")
