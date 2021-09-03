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
# ----------------------------------------------------------------------------
"""Utility functions for the Open3D TensorBoard plugin."""

from collections import OrderedDict
import logging
import threading
import numpy as np
import open3d as o3d

# Setup Python logger to emulate Open3D C++ logger.
_log = logging.getLogger("Open3D")
_log.propagate = False
_stream_handler = logging.StreamHandler()
_stream_handler.setFormatter(
    logging.Formatter('[%(name)s %(levelname)s T:%(threadName)s] %(message)s'))
# TODO(@ssheorey): Change to WARNING before merge
_stream_handler.setLevel(logging.DEBUG)
_log.setLevel(logging.DEBUG)
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


def to_dict_batch(o3d_geometry_list):
    """
    Convert sequence of identical Open3D geometry types to attribute-tensor
    dictionary. The geometry seequence forms a batch of data. Only common
    attributes are supported.

    TODO: This involves a data copy. Add support for List[Open3D geometry]
    directly to add_3d() if needed.

    Args:
        o3d_geometry_list (Iterable): Iterable (list / tuple / sequence
            generator) of Open3D Tensor geometry types.
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

        return {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
        }
    if isinstance(o3d_geometry_list[0], o3d.geometry.TriangleMesh):
        vertex_positions = []
        vertex_colors = []
        vertex_normals = []
        triangle_indices = []
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.vertices))
            vertex_colors.append(np.asarray(geometry.vertex_colors))
            vertex_normals.append(np.asarray(geometry.vertex_normals))
            triangle_indices.append(np.asarray(geometry.triangles))

        return {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'vertex_colors': np.stack(vertex_colors, axis=0),
            'vertex_normals': np.stack(vertex_normals, axis=0),
            'triangle_indices': np.stack(triangle_indices, axis=0),
        }

    if isinstance(o3d_geometry_list[0], o3d.geometry.LineSet):
        vertex_positions = []
        line_colors = []
        line_indices = []
        for geometry in o3d_geometry_list:
            vertex_positions.append(np.asarray(geometry.points))
            line_colors.append(np.asarray(geometry.colors))
            line_indices.append(np.asarray(geometry.lines))

        return {
            'vertex_positions': np.stack(vertex_positions, axis=0),
            'line_colors': np.stack(line_colors, axis=0),
            'line_indices': np.stack(line_indices, axis=0),
        }

    raise NotImplementedError(
        f"Geometry type {type(o3d_geometry_list[0])} is not suported yet.")
