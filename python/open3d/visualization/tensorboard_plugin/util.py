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
from collections import OrderedDict
import logging
import threading
import numpy as np
from tensorboard.backend.event_processing.plugin_event_multiplexer import EventMultiplexer
from tensorboard.backend.event_processing.plugin_asset_util import PluginDirectory
from tensorboard.compat.tensorflow_stub.pywrap_tensorflow import masked_crc32c
import open3d as o3d
from open3d.visualization.tensorboard_plugin import plugin_data_pb2
from open3d.visualization.tensorboard_plugin import metadata

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
    """ A lock object that allows many simultaneous "read locks", but
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
        lock."""
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
        write locks."""
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
        eject key-value pairs till the cache is within limits."""
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
        self._event_lock = threading.Lock()  # Protect TB event file data
        # Geometry data reading
        self._tensor_events = dict()
        self.geometry_cache = LRUCache(max_items=cache_max_items)
        self._file_handles = {}  # {filename, (open_handle, read_lock)}
        self._file_handles_lock = threading.Lock()
        self.reload_events()

    def reload_events(self):
        """Reload event file"""
        self.event_mux.Reload()
        run_tags = self.event_mux.PluginRunToTagToContent(metadata.PLUGIN_NAME)
        with self._event_lock:
            self._run_to_tags = {
                run: list(tagdict.keys()) for run, tagdict in run_tags.items()
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
        with self._event_lock:
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
            with self._file_handles_lock:
                if filename not in self._file_handles:
                    self._file_handles[filename] = (_fileopen(filename, "rb"),
                                                    threading.Lock())
                    if not self._file_handles[filename][0].seekable():
                        raise RuntimeError(filename +
                                           " does not support seeking."
                                           " This storage is not supported.")
                # lock to seek + read
                file_handle = self._file_handles[filename]
                file_handle[1].acquire()

            file_handle[0].seek(read_location)
            buf = file_handle[0].read(read_size)
            file_handle[1].release()
            if not read_masked_crc32c == masked_crc32c(buf):
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
                                              batch_idx, step_to_idx)
            # "vertex_normals" -> ["vertex", "normals"]
            prop_map, prop_attribute = prop.split("_")
            if prop_map == "vertex" and not isinstance(
                    geometry, o3d.t.geometry.TriangleMesh):
                prop_map = "point"
            # geometry.vertex["normals"] = geometry_ref.vertex["normals"]
            getattr(geometry, prop_map)[prop_attribute] = getattr(
                geometry_ref, prop_map)[prop_attribute]

        return geometry


def to_dict_batch(o3d_geometry_list):
    """Convert sequence of identical (legacy) Open3D geometry types to
    attribute-tensor dictionary. The geometry seequence forms a batch of data.
    Custom attributes are not supported.

    TODO: This involves a data copy. Add support for List[Open3D geometry]
    directly to add_3d() if needed.

    Args:
        o3d_geometry_list (Iterable): Iterable (list / tuple / sequence
            generator) of Open3D Tensor geometry types.

    Returns:
        Dict[str: numpy.array]: Dictionary of property names and corresponding
        Numpy arrays.
    """
    if len(o3d_geometry_list) == 0:
        return {}
    if isinstance(o3d_geometry_list[0], o3d.geometry.PointCloud):
        vertex_positions = []
        vertex_colors = []
        vertex_normals = []
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
        vertex_positions = []
        vertex_colors = []
        vertex_normals = []
        triangle_indices = []
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.vertices))
            vertex_colors.append(np.asarray(geometry.vertex_colors))
            vertex_normals.append(np.asarray(geometry.vertex_normals))
            triangle_indices.append(np.asarray(geometry.triangles))

        geo_dict = {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
            'triangle_indices': np.stack(triangle_indices, axis=0),
        }

    elif isinstance(o3d_geometry_list[0], o3d.geometry.LineSet):
        vertex_positions = []
        line_colors = []
        line_indices = []
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
            f"Geometry type {type(o3d_geometry_list[0])} is not suported yet.")

    # remove empty arrays
    for prop in tuple(geo_dict.keys()):
        if geo_dict[prop].size == 0:
            del geo_dict[prop]
    return geo_dict
