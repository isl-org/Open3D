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
"""Utility functions for the Open3D TensorBoard plugin."""

import os
from copy import deepcopy
from collections import OrderedDict
import logging
import threading

import numpy as np
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.plugin_asset_util import PluginDirectory
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c

import open3d as o3d
from open3d.visualization import rendering
from open3d.ml.vis import Colormap
from open3d.ml.vis import LabelLUT
from . import plugin_data_pb2
from . import metadata

try:
    from tensorflow.io.gfile import GFile as _fileopen
except ImportError:
    _fileopen = open

# Setup Python logger to emulate Open3D C++ logger.
_log = logging.getLogger("Open3D")
_log.propagate = False
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(
    logging.Formatter('[%(name)s %(levelname)s T:%(threadName)s] %(message)s'))
_stream_handler.setLevel(logging.WARNING)
_log.setLevel(logging.WARNING)
_log.addHandler(_stream_handler)


class ReadWriteLock:
    """A lock object that allows many simultaneous "read locks", but
    only one "write lock."

    Implmentation from Python Cookbook (O'Reilly)
    https://www.oreilly.com/library/view/python-cookbook/0596001673/ch06s04.html
    Credit: Sami Hangaslammi
    """

    def __init__(self):
        self._read_ready = threading.Condition(threading.Lock())
        self._readers = 0

    def acquire_read(self):
        """Acquire a read lock. Blocks only if a thread has acquired the write
        lock.
        """
        self._read_ready.acquire()
        try:
            self._readers += 1
        finally:
            self._read_ready.release()

    def release_read(self):
        """Release a read lock."""
        self._read_ready.acquire()
        try:
            self._readers -= 1
            if not self._readers:
                self._read_ready.notifyAll()
        finally:
            self._read_ready.release()

    def acquire_write(self):
        """Acquire a write lock. Blocks until there are no acquired read or
        write locks.
        """
        self._read_ready.acquire()
        while self._readers > 0:
            self._read_ready.wait()

    def release_write(self):
        """Release a write lock."""
        self._read_ready.release()


class LRUCache:
    """Cache storing recent items, i.e. least recently used will be ejected when
    a new item needs to be added to a full cache. This is thread safe for
    concurrent access.
    """

    def __init__(self, max_items=128):
        """
        Args:
            max_items (int): Max items in cache.
        """
        self.cache = OrderedDict()
        self.max_items = max_items
        self.rwlock = ReadWriteLock()
        # hits, misses are not protected against concurrent access for
        # performance.
        self.hits = 0
        self.misses = 0

    def get(self, key):
        """Retrieve value corresponding to ``key`` from the cache.

        Return:
            Value if ``key`` is found, else None.
        """
        self.rwlock.acquire_read()
        value = self.cache.get(key)  # None if not found
        self.rwlock.release_read()
        if value is None:
            self.misses += 1
            _log.debug(str(self))
            return None
        self.rwlock.acquire_write()
        self.cache.move_to_end(key)
        self.rwlock.release_write()
        self.hits += 1
        _log.debug(str(self))
        return value

    def put(self, key, value):
        """Add (key, value) pair to the cache. If cache limits are exceeded,
        eject key-value pairs till the cache is within limits.
        """
        self.rwlock.acquire_write()
        self.cache[key] = value
        self.cache.move_to_end(key)
        if len(self.cache) > self.max_items:
            self.cache.popitem(last=False)
        self.rwlock.release_write()
        _log.debug(str(self))

    def clear(self):
        """Invalidate cache."""
        self.rwlock.acquire_write()
        self.cache.clear()
        self.rwlock.release_write()

    def __str__(self):
        return (f"Items: {len(self.cache)}/{self.max_items}, "
                f"Hits: {self.hits}, Misses: {self.misses}")


def classify_properties(tensormap):
    """Classify custom geometry properties (other than positions, indices,
    colors, normals) into labels and other custom properties.

    Args:
        tensormap: geometry TensorMap (e.g.: `point` or `vertex`)

    Returns:
        label_prop (List): properties treated as labels (1D Int values).
        custom_prop (Dict): properties treated as custom properties (Float
            values, possibly multidimensional).
    """
    label_prop = {}
    custom_prop = {}
    exp_size = (tensormap['positions'].shape[0]
                if 'positions' in tensormap else tensormap['indices'].shape[0])
    for name, tensor in tensormap.items():
        if name in ('positions', 'colors', 'normals',
                    'indices') or name.startswith('_'):
            continue
        if (tensor.shape == (exp_size, 1) and
                tensor.dtype not in (o3d.core.float32, o3d.core.float64,
                                     o3d.core.undefined)):
            label_prop.update({name: 1})
        else:
            custom_prop.update({name: tensor.shape[1]})
    return label_prop, custom_prop


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
        })
        self._run_to_tags = {}
        self._event_lock = threading.Lock()  # Protect TB event file data
        # Geometry data reading
        self._tensor_events = dict()
        self.geometry_cache = LRUCache(max_items=cache_max_items)
        self.runtag_prop_shape = dict()
        self._file_handles = {}  # {filename, (open_handle, read_lock)}
        self._file_handles_lock = threading.Lock()
        self.reload_events()

    def reload_events(self):
        """Reload event file"""
        self.event_mux.AddRunsFromDirectory(self.logdir)
        self.event_mux.Reload()
        run_tags = self.event_mux.PluginRunToTagToContent(metadata.PLUGIN_NAME)
        with self._event_lock:
            self._run_to_tags = {
                run: sorted(tagdict.keys())
                for run, tagdict in sorted(run_tags.items())
            }
            self._tensor_events = dict()  # Invalidate index
        # Close all open files
        with self._file_handles_lock:
            while len(self._file_handles) > 0:
                unused_filename, file_handle = self._file_handles.popitem()
                with file_handle[1]:
                    file_handle[0].close()

        _log.debug(f"Event data reloaded: {self._run_to_tags}")

    def is_active(self):
        """Do we have any Open3D data to display?"""
        with self._event_lock:
            return any(len(tags) > 0 for tags in self._run_to_tags.values())

    @property
    def run_to_tags(self):
        """Locked access to the run_to_tags map."""
        with self._event_lock:
            return self._run_to_tags

    def tensor_events(self, run):
        """Locked access to tensor events for a run."""
        with self._event_lock:
            if run not in self._tensor_events:
                self._tensor_events[run] = {
                    tag: self.event_mux.Tensors(run, tag)
                    for tag in self._run_to_tags[run]
                }
            return self._tensor_events[run]

    def get_label_to_names(self, run, tag):
        """Get label (id) to name (category) mapping for a tag."""
        md_proto = self.event_mux.SummaryMetadata(run, tag)
        lab2name = metadata.parse_plugin_metadata(md_proto.plugin_data.content)
        return dict(sorted(lab2name.items()))

    def read_from_file(self, filename, read_location, read_size,
                       read_masked_crc32c):
        """Read data from the file ``filename`` from a given offset
        ``read_location`` and size ``read_size``. Data is validated with the provided
        ``masked_crc32c``. This is thread safe and manages a list of open files.
        """
        with self._file_handles_lock:
            if filename not in self._file_handles:
                self._file_handles[filename] = (_fileopen(filename, "rb"),
                                                threading.Lock())
                if not self._file_handles[filename][0].seekable():
                    raise RuntimeError(filename + " does not support seeking."
                                       " This storage is not supported.")
            # lock to seek + read
            file_handle = self._file_handles[filename]
            file_handle[1].acquire()

        file_handle[0].seek(read_location)
        buf = file_handle[0].read(read_size)
        file_handle[1].release()
        if masked_crc32c(buf) == read_masked_crc32c:
            return buf
        else:
            return None

    def update_runtag_prop_shape(self, run, tag, geometry,
                                 inference_data_proto):
        """Update list of custom properties and their shapes for different runs
        and tags.
        """
        tag_prop_shape = self.runtag_prop_shape.setdefault(run, dict())
        prop_shape = tag_prop_shape.setdefault(tag, dict())
        if len(prop_shape) == 0 and not geometry.is_empty():
            for prop_type in ('point', 'vertex'):  # exclude 'line'
                if hasattr(geometry, prop_type):
                    label_props, custom_props = classify_properties(
                        getattr(geometry, prop_type))
                    prop_shape.update(custom_props)
                    prop_shape.update(label_props)
            if len(inference_data_proto.inference_result) > 0:
                prop_shape.update({'labels': 1, 'confidence': 1})

    def read_geometry(self, run, tag, step, batch_idx, step_to_idx):
        """Geometry reader from msgpack files."""
        idx = step_to_idx[step]
        metadata_proto = plugin_data_pb2.Open3DPluginData()
        run_tensor_events = self.tensor_events(run)
        metadata_proto.ParseFromString(
            run_tensor_events[tag][idx].tensor_proto.string_val[0])
        data_dir = PluginDirectory(os.path.join(self.logdir, run),
                                   metadata.PLUGIN_NAME)
        filename = os.path.join(data_dir, metadata_proto.batch_index.filename)
        read_location = metadata_proto.batch_index.start_size[batch_idx].start
        read_size = metadata_proto.batch_index.start_size[batch_idx].size
        read_masked_crc32c = metadata_proto.batch_index.start_size[
            batch_idx].masked_crc32c
        cache_key = (filename, read_location, read_size, run, tag, step,
                     batch_idx)
        geometry = self.geometry_cache.get(cache_key)
        if geometry is None:  # Read from storage
            buf = self.read_from_file(filename, read_location, read_size,
                                      read_masked_crc32c)
            if buf is None:
                raise IOError(f"Geometry {cache_key} reading failed! CRC "
                              "mismatch in msgpack data.")
            msg_tag, msg_step, geometry = o3d.io.rpc.data_buffer_to_meta_geometry(
                buf)
            if geometry is None:
                raise IOError(f"Geometry {cache_key} reading failed! Possible "
                              "msgpack or TensorFlow event file corruption.")
            if tag != msg_tag or step != msg_step:
                _log.warning(
                    f"Mismatch between TensorFlow event (tag={tag}, step={step})"
                    f" and msgpack (tag={msg_tag}, step={msg_step}) data. "
                    "Possible data corruption.")
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
                                              batch_idx, step_to_idx)[0]
            # "vertex_normals" -> ["vertex", "normals"]
            prop_map, prop_attribute = prop.split("_")
            if prop_map == "vertex" and not isinstance(
                    geometry, o3d.t.geometry.TriangleMesh):
                prop_map = "point"
            # geometry.vertex["normals"] = geometry_ref.vertex["normals"]
            getattr(geometry, prop_map)[prop_attribute] = getattr(
                geometry_ref, prop_map)[prop_attribute]

        # Aux data (e.g. labels, confidences) -> not cached
        aux_read_location = metadata_proto.batch_index.start_size[
            batch_idx].aux_start
        aux_read_size = metadata_proto.batch_index.start_size[
            batch_idx].aux_size
        aux_read_masked_crc32c = metadata_proto.batch_index.start_size[
            batch_idx].aux_masked_crc32c
        data_bbox_proto = plugin_data_pb2.InferenceData()
        if aux_read_size > 0:
            data_bbox_serial = self.read_from_file(filename, aux_read_location,
                                                   aux_read_size,
                                                   aux_read_masked_crc32c)
            if data_bbox_serial is None:
                raise IOError(f"Aux data for {cache_key} reading failed! CRC "
                              "mismatch in protobuf data.")
            data_bbox_proto.ParseFromString(data_bbox_serial)

        self.update_runtag_prop_shape(run, tag, geometry, data_bbox_proto)
        return geometry, data_bbox_proto


def _normalize(tensor):
    """Normalize tensor by scaling and shifting to range [0, 1].

    Args:
        tensor: Open3D float tensor. Int tensors are returned unchanged.

    Return:
        tuple: (Normalized tensor, min value of tensor, max value of tensor) min
        and max are stretched to include 0 or 1 for degenerate tensors.
    """
    if tensor.dtype in (o3d.core.float32, o3d.core.float64, np.float32,
                        np.float64):
        m, M = tensor.min().item(), tensor.max().item()
        if M <= m + 1e-6:  # stretch degenerate range to include [0, 1]
            m, M = min(m, 0), max(M, 1)
        return (tensor - m) * (1. / (M - m)), m, M
    return tensor, 0, 1


def _u8_to_float(color):
    return tuple(float(c) / 255 for c in color)


def _float_to_u8(color):
    return tuple(round(255 * c) for c in color)


class RenderUpdate:
    """Update rendering for a geometry, based on an incoming
    "tensorboard/update_rendering" message. Default rendering is used with an
    empty message.

    We use uint8 colors everywhere, since they are more compact to store during
    serialization. Colormaps are converted to float here, since rendering uses float
    colormaps.
    """

    _LINES_PER_BBOX = 17  # BoundingBox3D
    ALL_UPDATED = ["property", "shader", "colormap"]
    _CMAPS = {
        "RAINBOW": Colormap.make_rainbow(),
        "GREYSCALE": Colormap.make_greyscale(),
    }
    DICT_COLORMAPS = {
        name: {
            # float -> uint8, and RGB -> RGBA
            point.value: _float_to_u8(point.color) + (255,)
            for point in cmap.points
        } for name, cmap in _CMAPS.items()
    }
    LABELLUT_COLORS = {
        label: _float_to_u8(color + [1.0])
        for label, color in enumerate(LabelLUT.Colors)
    }

    def __init__(self, window_scaling, message, label_to_names):
        from open3d.visualization.async_event_loop import async_event_loop
        self._gui = async_event_loop
        self._window_scaling = window_scaling
        self._label_to_names = label_to_names
        render_state = message.get("render_state", {
            "property": "",
            "index": 0,
            "shader": "",
            "colormap": None,
        })
        self._updated = message.get("updated", deepcopy(self.ALL_UPDATED))
        self._property = render_state["property"]
        if int(render_state["index"]) >= 0:
            self._index = int(render_state["index"])
        # For custom scalar / 3-vector visualization
        self.data_range = None
        self._shader = render_state["shader"]

        # { label or value : float tuple (r,g,b,a) in range [0., 1.] }
        self._colormap = None  # Initialize in first apply()
        if render_state["colormap"] is not None:
            to_number = int if self._shader == "unlitGradient.LUT" else float
            self._colormap = {
                to_number(label_value): tuple(int(c) for c in rgba) for
                label_value, rgba in sorted(render_state["colormap"].items(),
                                            key=lambda l_c: to_number(l_c[0]))
            }

    def get_render_state(self):
        """Return current render state."""
        return {
            "property": self._property,
            "index": self._index,
            "shader": self._shader,
            "colormap": self._colormap,
            "range": self.data_range or [0, 1]
        }

    def _update_range(self, min_val, max_val):
        if self.data_range is None:
            self.data_range = [min_val, max_val]
        else:
            self.data_range = [
                min(min_val, self.data_range[0]),
                max(max_val, self.data_range[1])
            ]

    def _set_vis_minmax(self, geometry_vertex, material):
        m, M = 0., 1.
        if "__visualization_scalar" in geometry_vertex:
            m = geometry_vertex["__visualization_scalar"].min().item()
            M = geometry_vertex["__visualization_scalar"].max().item()
            if M <= m + 1e-6:  # stretch degenerate range to include [0, 1]
                m, M = min(m, 0), max(M, 1)
            self._update_range(m, M)
        material.scalar_min, material.scalar_max = m, M
        _log.debug("material colormap range range set to "
                   f"{material.scalar_min, material.scalar_max}")

    def apply(self, o3dvis, geometry_name, geometry, inference_data_proto=None):
        """Apply the RenderUpdate to a geometry.

        Args:
            o3dvis (O3DVisualizer): Window containing the geometry.
            geometry_name (str): Geometry name in the window.
            geometry (o3d.t.geometry): Geometry whose rendering is to be
                updated.
            inference_data_proto : BoundingBox labels and confidences.
        """

        def swap__(tensormap, prop):
            """If backup of property prop exists, restore it. Else save prop to
            a backup (__prop).  prop can now be safely modified for
            visualization. Returns True if prop exists, else False.
            """
            if "__" + prop in tensormap:
                tensormap[prop][:] = tensormap["__" + prop][:]  # restore
                return True
            if prop in tensormap:
                tensormap["__" + prop] = tensormap[prop].clone()  # backup
                return True
            return False

        if (len(self._updated) == 0 or geometry.is_empty()):
            _log.debug("No updates, or empty geometry.")
            return

        geometry_vertex = (geometry.point
                           if hasattr(geometry, 'point') else geometry.vertex)
        have_colors = ("colors" in geometry.line if hasattr(geometry, 'line')
                       else "colors" in geometry_vertex)

        if inference_data_proto is not None:
            inference_result = inference_data_proto.inference_result
        else:
            inference_result = []
        # Set default shader and property
        label_props, custom_props = classify_properties(geometry_vertex)
        if self._shader == "":
            if (len(inference_result) > 0  # Have BB labels
                    or len(label_props) > 0):  # Have vertex labels
                self._shader = "unlitGradient.LUT"
                if self._property == "":
                    self._property = (next(iter(label_props))
                                      if len(label_props) > 0 else "labels")
                    self._index = 0
            elif isinstance(geometry, o3d.t.geometry.LineSet):
                self._shader = "unlitLine"
            elif len(custom_props) > 0:
                self._shader = "unlitGradient.GRADIENT.RAINBOW"
                if self._property == "":
                    self._property = next(iter(custom_props))
                    self._index = 0
            elif "normals" in geometry_vertex:
                self._shader = "defaultLit"
            elif "colors" in geometry_vertex:
                self._shader = "defaultUnlit"
            else:  # Only XYZ
                self._shader = "unlitSolidColor"

        o3dscene = o3dvis.scene
        if o3dscene.has_geometry(geometry_name):
            updated = self._updated
            # update_geometry() only accepts tPointCloud
            if not isinstance(geometry, o3d.t.geometry.PointCloud):
                o3dvis.remove_geometry(geometry_name)
        else:
            updated = deepcopy(self.ALL_UPDATED)

        geometry_update_flag = 0
        material_update_flag = 0
        material = o3dvis.get_geometry_material(geometry_name)

        # Visualize scalar / 3-vector property with color map
        if "property" in updated:
            # Float Scalar with colormap
            if ((self._property in custom_props or
                 self._property in label_props) and
                    self._shader.startswith("unlitGradient") and
                    geometry_vertex[self._property].shape[1] > self._index):
                geometry_vertex["__visualization_scalar"] = geometry_vertex[
                    self._property][:, self._index].to(o3d.core.float32)
                geometry_update_flag |= rendering.Scene.UPDATE_UV0_FLAG
            # 3-vector as RGB
            elif (self._property in custom_props and
                  self._shader == "defaultUnlit" and
                  geometry_vertex[self._property].shape[1] >= 3):
                swap__(geometry_vertex, "colors")
                geometry_vertex["colors"], min_val, max_val = _normalize(
                    geometry_vertex[self._property][:, :3])
                self._update_range(min_val, max_val)
                geometry_update_flag |= rendering.Scene.UPDATE_COLORS_FLAG

        # Bounding boxes
        if isinstance(geometry, o3d.t.geometry.LineSet):
            if 'shader' in updated:  # shader is always "unlitLine"
                updated.remove("shader")
            material.shader = "unlitLine"
            material.line_width = 2 * self._window_scaling
            if not swap__(geometry.line, "colors"):
                geometry.line["colors"] = o3d.core.Tensor.full(
                    (len(geometry.line["indices"]), 3),
                    128,
                    dtype=o3d.core.uint8)
            if self._property == "" and self._shader == "unlitSolidColor":
                geometry.line["colors"][:] = _u8_to_float(
                    next(iter(self._colormap.values())))[:3]
            elif self._property != "" and len(inference_result) > 0:  # labels
                if self._colormap is None:
                    self._colormap = deepcopy(self.LABELLUT_COLORS)
                    for label in list(self._colormap):
                        if label not in self._label_to_names:
                            self._colormap.pop(label)
                if "colormap" in updated:
                    swap__(geometry.line, "indices")
                    t_lines = geometry.line["indices"]
                    t_colors = geometry.line["colors"]
                    idx = 0
                    for k in range(len(inference_result)):
                        label = inference_result[k].label
                        if label in self._colormap:
                            col = self._colormap[label]
                            col, alpha = col[:3], col[3]
                        else:
                            col, alpha = (128, 128, 128), 255
                        t_colors[idx:idx + self._LINES_PER_BBOX] = col
                        if alpha == 0:
                            t_lines[idx:idx + self._LINES_PER_BBOX] = 0
                        idx += self._LINES_PER_BBOX

        # PointCloud, Mesh, LineSet with colors
        elif "shader" in updated or "colormap" in updated:
            material_update_flag = 1
            if self._shader == "defaultLit":
                material.shader = "defaultLit"
            elif self._shader == "defaultUnlit":
                material.shader = "defaultUnlit"
                material.base_color = [0.5, 0.5, 0.5, 1.0]
            elif self._shader == "unlitSolidColor":
                material.shader = "unlitSolidColor"
                if self._colormap is None:
                    self._colormap = {0.0: [128, 128, 128, 255]}
                material.base_color = _u8_to_float(
                    next(iter(self._colormap.values())))
            elif self._shader == "unlitGradient.LUT":  # Label Colormap
                if self._colormap is None:
                    self._colormap = deepcopy(self.LABELLUT_COLORS)
                    for label in list(self._colormap):
                        if label not in self._label_to_names:
                            self._colormap.pop(label)
                material.shader = "unlitGradient"
                material.gradient = rendering.Gradient()
                if len(self._colormap) > 1:
                    material.gradient.points = list(
                        rendering.Gradient.Point(
                            float(i) /
                            (len(self._colormap) - 1), _u8_to_float(color))
                        for i, color in enumerate(self._colormap.values()))
                else:
                    material.gradient.points = [
                        rendering.Gradient.Point(0.0, [1.0, 0.0, 1.0, 1.0])
                    ]
                material.gradient.mode = rendering.Gradient.LUT
                material.scalar_min, material.scalar_max = 0, max(
                    self._label_to_names.keys())
            # Colormap (RAINBOW / GREYSCALE): continuous data
            elif self._shader.startswith("unlitGradient.GRADIENT."):
                if self._colormap is None:
                    self._colormap = deepcopy(self.DICT_COLORMAPS["RAINBOW"])
                material.shader = "unlitGradient"
                material.gradient = rendering.Gradient()
                material.gradient.points = list(
                    rendering.Gradient.Point(value,
                                             _u8_to_float(color[:3]) + (1.,))
                    for value, color in self._colormap.items())
                material.gradient.mode = rendering.Gradient.GRADIENT
                self._set_vis_minmax(geometry_vertex, material)

        if o3dscene.has_geometry(geometry_name):
            if material_update_flag > 0:
                self._gui.run_sync(o3dvis.modify_geometry_material,
                                   geometry_name, material)
                # does not do force_redraw(), so also need update_geometry()
            self._gui.run_sync(o3dvis.update_geometry, geometry_name, geometry,
                               geometry_update_flag)
            _log.debug(
                f"Geometry {geometry_name} updated with flags "
                f"Geo:{geometry_update_flag:b}, Mat:{material_update_flag:b}")
        else:
            self._gui.run_sync(o3dvis.add_geometry, geometry_name, geometry,
                               material if self._shader == "" else None)
        self._gui.run_sync(o3dvis.post_redraw)
        if not have_colors:  # reset colors
            if hasattr(geometry, "line") and "colors" in geometry.line:
                del geometry.line["colors"]
            elif "colors" in geometry_vertex:
                del geometry_vertex["colors"]
        _log.debug(f"apply complete: {geometry_name} with "
                   f"{self._property}[{self._index}]/{self._shader}")


def to_dict_batch(o3d_geometry_list):
    """Convert sequence of identical Open3D Eigen geometry types to
    attribute-tensor dictionary. The geometry sequence forms a batch of data.
    Custom attributes are not supported.

    Args:
        o3d_geometry_list (Iterable): Iterable (list / tuple / sequence
            generator) of Open3D Eigen geometry types.

    Returns:
        Dict[str: numpy.array]: Dictionary of property names and corresponding
        Numpy arrays.
    """
    if len(o3d_geometry_list) == 0:
        return {}
    vertex_positions = []
    vertex_colors = []
    vertex_normals = []
    triangle_indices = []
    triangle_uvs = []
    line_colors = []
    line_indices = []
    if isinstance(o3d_geometry_list[0], o3d.geometry.PointCloud):
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.points))
            vertex_colors.append(np.asarray(geometry.colors))
            vertex_normals.append(np.asarray(geometry.normals))

        geo_dict = {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
        }

    elif isinstance(o3d_geometry_list[0], o3d.geometry.TriangleMesh):
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.vertices))
            vertex_colors.append(np.asarray(geometry.vertex_colors))
            vertex_normals.append(np.asarray(geometry.vertex_normals))
            triangle_indices.append(np.asarray(geometry.triangles))
            if geometry.has_triangle_uvs():
                triangle_uvs.append(
                    np.asarray(geometry.triangle_uvs).reshape((-1, 3, 2)))

        geo_dict = {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
            'triangle_indices': np.stack(triangle_indices, axis=0),
        }
        if len(triangle_uvs) > 0:
            geo_dict.update(triangle_texture_uvs=np.stack(triangle_uvs, axis=0))

    elif isinstance(o3d_geometry_list[0], o3d.geometry.LineSet):
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.points))
            line_colors.append(np.asarray(geometry.colors))
            line_indices.append(np.asarray(geometry.lines))

        geo_dict = {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'line_colors': np.stack(line_colors, axis=0),
            'line_indices': np.stack(line_indices, axis=0),
        }

    else:
        raise NotImplementedError(
            f"Geometry type {type(o3d_geometry_list[0])} is not supported yet.")

    # remove empty arrays
    for prop in tuple(geo_dict.keys()):
        if geo_dict[prop].size == 0:
            del geo_dict[prop]
    return geo_dict
