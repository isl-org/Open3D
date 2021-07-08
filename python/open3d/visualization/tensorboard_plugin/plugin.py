"""Open3D visualization plugin for Tensorboard"""
import json
import os
import threading
import time
from collections import namedtuple

import mimetypes
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
from open3d.visualization.tensorboard_plugin import metadata
# Note: the _AsyncEventLoop is started whenever this module is imported.
# from open3d.visualization._async_event_loop import _async_event_loop

import ipdb
import logging as log

Window = namedtuple('Window', ['run', 'tags', 'step'])


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
        self._events = EventMultiplexer(tensor_size_guidance={
            metadata.PLUGIN_NAME: self._MAX_SAMPLES_IN_RAM
        }).AddRunsFromDirectory(self._logdir)
        self._windows = {}
        self._next_wid = 0
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Debug)
        # o3d.visualization.webrtc_server.disable_http_handshake()
        o3d.visualization.webrtc_server.enable_webrtc()

    @wrappers.Request.application
    def _new_window(self, request):

        self._events.Reload()
        run_to_tags = self._events.PluginRunToTagToContent(metadata.PLUGIN_NAME)
        # Pick first run for now
        run = next(iter(run_to_tags))

        tags = [
            tag[:-9] for tag in run_to_tags[run] if tag.endswith('_vertices')
        ]
        # Pick first tag for now
        tag = tags[0]

        # Pick step 0
        step = 0

        # Pick first in the batch
        batch_idx = 0

        mesh = o3d.geometry.TriangleMesh()

        mesh.vertices = o3d.utility.Vector3dVector(
            tensor_util.make_ndarray(
                self._events.Tensors(
                    run, tag + '_vertices')[step].tensor_proto)[batch_idx, ...])
        for prop in metadata._MESH_PROPERTIES:
            if tag + '_' + prop in run_to_tags[run]:
                prop_value = tensor_util.make_ndarray(
                    self._events.Tensors(run, tag + '_' +
                                         prop)[step].tensor_proto)[batch_idx,
                                                                   ...]
                setattr(
                    mesh, prop,
                    o3d.utility.Vector3iVector(prop_value.astype(int)) if prop
                    == 'triangles' else o3d.utility.Vector3dVector(prop_value))

        threading.Thread(target=lambda: o3d.visualization.draw(mesh)).start()

        self._windows[self._next_wid] = Window(run, tags, step)
        response = str(self._next_wid)
        self._next_wid += 1

        time.sleep(0.4)
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
