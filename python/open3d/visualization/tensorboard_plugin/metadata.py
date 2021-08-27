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
"""Internal information about the Open3D plugin."""

import json
from tensorboard.compat.proto.summary_pb2 import SummaryMetadata
from open3d.visualization.tensorboard_plugin import plugin_data_pb2
import open3d as o3d
import numpy as np

# Setup Python logger to emulate Open3D C++ logger.
import logging as _logging
log = _logging.getLogger("Open3D")
log.propagate = False
_stream_handler = _logging.StreamHandler()
_stream_handler.setFormatter(
    _logging.Formatter('[%(name)s %(levelname)s T:%(threadName)s] %(message)s'))
_stream_handler.setLevel(_logging.DEBUG)
log.setLevel(_logging.DEBUG)
log.addHandler(_stream_handler)

PLUGIN_NAME = "Open3D"

# The most recent value for the `version` field of the
# `Open3DPluginData` proto. Sync with Open3D version (MAJOR*100 + MINOR)
_VERSION = 14

SUPPORTED_FILEFOPRMAT_VERSIONS = [14]

GEOMETRY_PROPERTY_DIMS = {
    'vertex_positions': 3,
    'vertex_normals': 3,
    'vertex_colors': 3,
    'vertex_texture_uvs': 2,
    'triangle_indices': 3,
    'line_indices': 2,
    'material': 0
}
VERTEX_PROPERTIES = ('vertex_normals', 'vertex_colors', 'vertex_texture_uvs')
TRIANGLE_PROPERTIES = ()
LINE_PROPERTIES = ('line_colors',)


def create_summary_metadata(description):
    """Creates summary metadata. Reserved for future use. Required by
    TensorBoard.

    Arguments:
      description: The description to show in TensorBoard.

    Returns:
      A `SummaryMetadata` protobuf object.
    """
    return SummaryMetadata(
        summary_description=description,
        plugin_data=SummaryMetadata.PluginData(plugin_name=PLUGIN_NAME,
                                               content=b''),
    )


def parse_plugin_metadata(unused_content):
    """Parse summary metadata to a Python object. Reserved for future use.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the Open3D plugin.
    """
    return b''


# Utility functions


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
