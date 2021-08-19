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
    """Creates summary metadata which defined at ??? proto.

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


def parse_plugin_metadata(content):
    """Parse summary metadata to a Python object.

    Arguments:
      content: The `content` field of a `SummaryMetadata` proto
        corresponding to the Open3D plugin.

    Returns:
      A `Open3DPluginData` protobuf object.

    Raises:
      Error if the version of the plugin is not supported.
    """
    return b''
    # if not isinstance(content, bytes):
    #     raise TypeError("Content type must be bytes.")
    # result = plugin_data_pb2.Open3DPluginData.FromString(content)

    # if result.version not in SUPPORTED_FILEFOPRMAT_VERSIONS:
    #     raise RuntimeError(
    #         f"Open3D plugin fileformat version {result.version} is not " +
    #         f"supported. Supported versions are {SUPPORTED_FILEFOPRMAT_VERSIONS}."
    #     )
    # return result
