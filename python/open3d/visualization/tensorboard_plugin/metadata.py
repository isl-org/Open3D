"""Internal information about the Open3D plugin."""

from tensorboard.compat.proto import summary_pb2

PLUGIN_NAME = "Open3D"

# The most recent value for the `version` field of the
# `Open3DPluginData` proto. Sync with Open3D version (MAJOR*100 + MINOR)
_PROTO_VERSION = 14

MESH_PROPERTIES = ('vertices', 'vertex_normals', 'vertex_colors', 'triangles',
                   'triangle_normals', 'triangle_material_ids', 'triangle_uvs',
                   'adjacency_list', 'textures')


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
            plugin_name=PLUGIN_NAME,
            content=b"",  # no need for summary-specific metadata
        ),
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
    return b""
