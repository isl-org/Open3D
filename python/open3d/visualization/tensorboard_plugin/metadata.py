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
from tensorboard.compat.proto import summary_pb2
from open3d.visualization.tensorboard_plugin import plugin_data_pb2

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
    # 'vertex_uvs': 2,
    'triangle_indices': 3,
    'line_indices': 2
}
VERTEX_PROPERTIES = (
    'vertex_normals',
    'vertex_colors',
    # 'vertex_uvs'
)
TRIANGLE_PROPERTIES = ()
LINE_PROPERTIES = ()


def create_summary_metadata(description):
    """Creates summary metadata. Reserved for future use. Required by
    TensorBoard.

    Arguments:
      description: The description to show in TensorBoard.

    Returns:
      A `summary_pb2.SummaryMetadata` protobuf object.
    """
    return summary_pb2.SummaryMetadata(
        summary_description=description,
        plugin_data=summary_pb2.SummaryMetadata.PluginData(
            plugin_name=PLUGIN_NAME, content=b''),
    )


def parse_plugin_metadata(unused_content):
    """Parse summary metadata to a Python object. Reserved for future use.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the Open3D plugin.
    """
    return b''
